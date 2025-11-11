# step3_stream_features_parallel.py
# Fast, causal stream features + labels (parallel, cached)
# -------------------------------------------------------

import os, sys, time, zlib, uuid
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd

# ======================= CONFIG =======================
DATA_ROOT    = Path(r"C:\Users\Nilufar\Desktop\BINANCE")   # aggTrades/<ASSET>/<ASSET>-aggTrades-YYYY-MM-DD.csv
EVENTS_CSV   = Path("./events_refined_with_trades.csv")    # from Step 2
MANIFEST     = Path("./agg_download_manifest.csv")         # pos / near_miss / quiet

# Output as a partitioned dataset (recommended for scale)
OUT_DATASET_DIR = Path("./features_stream_dataset")        # writes asset=.../date=YYYY-MM-DD/part-*.parquet

# Optional single Parquet (turn on only if dataset fits in RAM)
WRITE_SINGLE_PARQUET = False
OUT_SINGLE_PARQUET   = Path("./features_stream.parquet")

# Labeling & sampling
BUCKET_MS   = 200                      # 200ms buckets (set to 1000 for 1s while prototyping)
H_PRE       = pd.Timedelta(seconds=120) # positive window before trough
# at top
NEG_PER_DAY_POS   = 100
NEG_PER_DAY_NEAR  = 300
NEG_PER_DAY_QUIET = 300
POS_MAX_PER_DAY = 200                 # e.g., 300 to cap positives/day; None disables

# Ring windows (left,right] in seconds
SLICE_CUTS_S = [0, 5, 20, 80, 160]     # → (0,5], (5,20], (20,80], (80,160]

# Parallelism
MAX_WORKERS = min(8, max(1, (os.cpu_count() or 4) - 1))

# RNG
RNG_SEED = 42                           # base seed for reproducibility

# Parquet cache for daily aggTrades (speeds up subsequent runs)
CACHE_DIR = DATA_ROOT / "_cache_parquet"
# ======================================================


# ----------------------- Utils ------------------------
def ensure_utc(s: pd.Series) -> pd.Series:
    s = pd.to_datetime(s, errors="coerce")
    return s.dt.tz_convert("UTC") if getattr(s.dtype, "tz", None) is not None else s.dt.tz_localize("UTC")

def _to_bool_series(s: pd.Series) -> pd.Series:
    ss = s.astype(str).str.strip().str.lower()
    return ss.isin(["1","true","t","y","yes"])

def _detect_ts_unit(sample_int: pd.Series) -> str:
    v = pd.to_numeric(sample_int, errors="coerce").dropna()
    if v.empty: return "ms"
    return "ms" if v.iloc[: min(10000, len(v))].median() > 1e11 else "s"


# --------------- Loaders (fast + cache) ---------------
def load_agg_day_csv_fast(asset: str, date_utc: pd.Timestamp) -> pd.DataFrame | None:
    """Fast CSV read (price, qty, ts, isBuyerMaker if present). Avoid global sort if already time-ordered."""
    ds = pd.Timestamp(date_utc).strftime("%Y-%m-%d")
    p  = DATA_ROOT / "aggTrades" / asset / f"{asset}-aggTrades-{ds}.csv"
    if not p.exists():
        return None

    usecols = [1, 2, 5, 6]  # price, qty, ts, isBuyerMaker (col 6 may not exist in older dumps)
    names   = ["price", "quantity", "ts", "is_buyer_maker"]

    # Try pyarrow engine (fast). Fallback to C engine and drop isBuyerMaker if missing.
    try:
        df = pd.read_csv(p, header=None, usecols=usecols, names=names, engine="pyarrow")
    except Exception:
        df = pd.read_csv(p, header=None, usecols=[1, 2, 5], names=["price","quantity","ts"], engine="c")
        df["is_buyer_maker"] = pd.NA

    ts_unit = _detect_ts_unit(df["ts"].iloc[:10000])

    df["price"]    = pd.to_numeric(df["price"], errors="coerce")
    df["quantity"] = pd.to_numeric(df["quantity"], errors="coerce")
    df["ts"]       = pd.to_datetime(pd.to_numeric(df["ts"], errors="coerce"), unit=ts_unit, utc=True)
    if "is_buyer_maker" in df.columns:
        df["is_buyer_maker"] = _to_bool_series(df["is_buyer_maker"])

    df = df.dropna(subset=["price","quantity","ts"]).set_index("ts")
    # Binance dumps are typically time-ordered; if not, sort once
    if not df.index.is_monotonic_increasing:
        df = df.sort_index()
    return df


def load_agg_day(asset: str, date_utc: pd.Timestamp) -> pd.DataFrame | None:
    """Parquet cache -> CSV fast loader."""
    ds = pd.Timestamp(date_utc).strftime("%Y-%m-%d")
    outp = CACHE_DIR / asset / f"{asset}-{ds}.parquet"
    outp.parent.mkdir(parents=True, exist_ok=True)
    if outp.exists():
        try:
            return pd.read_parquet(outp)
        except Exception:
            pass  # fall through to CSV if cache corrupted

    df = load_agg_day_csv_fast(asset, date_utc)
    if df is not None and not df.empty:
        try:
            df.to_parquet(outp, engine="pyarrow", compression="snappy")
        except Exception:
            pass
    return df


# ---- Large trade threshold (fast, causal via 1s as-of) ----
def set_large_trade_rolling_fast(agg: pd.DataFrame,
                                 lookback: str = "3h",
                                 q: float = 0.975,
                                 min_samp: int = 200,
                                 hard_floor: float = 1e4,
                                 hard_cap: float = 5e5) -> None:
    """Compute rolling quantile on 1-second notional, then as-of join to ticks (strictly past-only)."""
    if not agg.index.is_monotonic_increasing:
        agg.sort_index(inplace=True)

    notional_tick = (agg["price"] * agg["quantity"]).astype("float32")

    # 1) Resample to 1s notional
    sec = notional_tick.rename("notional").to_frame().resample("1s").sum().fillna(0.0)

    # 2) Rolling past-only quantile at 1Hz (closed='left'), fallback expanding shifted
    roll_q = sec["notional"].rolling(lookback, min_periods=min_samp, closed="left").quantile(q)
    exp_q  = sec["notional"].expanding(min_periods=min_samp).quantile(q).shift(1)
    thr_s  = roll_q.fillna(exp_q).fillna(hard_floor).clip(hard_floor, hard_cap).astype("float32")

    # 3) As-of join (pad) back to ticks
    thr_ticks = thr_s.reindex(agg.index, method="pad").fillna(hard_floor)
    agg["large_trade"] = (notional_tick >= thr_ticks).astype("int8")


# -------------------- Preprocess trades --------------------
def preprocess_trades(df_tr: pd.DataFrame) -> pd.DataFrame:
    """Lee–Ready with buyer_maker tie-break, signed flow, fast rolling large-trade flag."""
    agg = df_tr.copy()
    if agg.index.name != "ts":
        agg = agg.set_index("ts")
    if not agg.index.is_monotonic_increasing:
        agg = agg.sort_index()

    # Lee–Ready (tick rule); tie-break with is_buyer_maker (buyer_maker=True => sell-initiated => -1)
    dP   = agg["price"].diff()
    tick = np.sign(dP).replace([np.inf, -np.inf], 0).fillna(0).astype("int8")
    zero = (tick == 0)
    if "is_buyer_maker" in agg.columns:
        ibm = agg["is_buyer_maker"].fillna(False).astype(bool)
        tick.loc[zero] = np.where(ibm.loc[zero], -1, 1).astype("int8")
    tick.replace(0, np.nan, inplace=True)
    tick.ffill(inplace=True)
    tick.fillna(0, inplace=True)

    agg["dir_lr"]     = tick
    agg["signed_qty"] = (agg["quantity"] * agg["dir_lr"]).astype("float32")

    # large trade (fast, causal)
    set_large_trade_rolling_fast(agg)

    return agg


# ----------------------- Bucketize -------------------------
def bucketize(agg: pd.DataFrame, bucket_ms: int) -> pd.DataFrame:
    rule = f"{int(bucket_ms)}ms"

    agg = agg.copy()
    # always present
    agg["notional"] = (agg["price"] * agg["quantity"]).astype("float32")
    # present even if 'large_trade' missing (zeros then)
    if "large_trade" in agg.columns:
        agg["large_notional"] = np.where(agg["large_trade"] == 1, agg["notional"], 0.0).astype("float32")
    else:
        agg["large_notional"] = 0.0

    g = agg.groupby(pd.Grouper(freq=rule))

    # build columns one by one and align on the same datetime index
    idx = g.size().index
    out = pd.DataFrame(index=idx)
    out["price_last"]       = g["price"].last()
    out["vol_sum"]          = g["quantity"].sum()
    out["signed_qty_sum"]   = g["signed_qty"].sum()
    out["n_trades"]         = g.size()
    out["dollar_sum"]       = g["notional"].sum()
    out["large_dollar_sum"] = g["large_notional"].sum()
    out["large_cnt"]        = g["large_trade"].sum() if "large_trade" in agg.columns else 0.0

    # expand to a complete grid at the chosen frequency
    if len(out.index):
        idx_full = pd.date_range(out.index.min(), out.index.max(), freq=rule, tz="UTC")
        out = out.reindex(idx_full)

    # fill
    out["price_last"] = out["price_last"].ffill()
    for c in ["vol_sum","signed_qty_sum","n_trades","large_cnt","dollar_sum","large_dollar_sum"]:
        out[c] = out[c].fillna(0.0)

    out["ret_1bkt"] = out["price_last"].pct_change().fillna(0.0).astype("float32")
    return out


# ------------------ Ring features (fast) -------------------
def add_slice_features_fast(df: pd.DataFrame, cuts_s=None) -> pd.DataFrame:
    if df.empty: return df
    if cuts_s is None:
        cuts_s = SLICE_CUTS_S

    # rings in ns
    cuts_ns = np.array([int(s * 1e9) for s in cuts_s], dtype="int64")
    pairs   = list(zip(cuts_ns[:-1], cuts_ns[1:]))

    idx = df.index.view("int64")
    n   = len(df)

    price   = df["price_last"].astype("float64").to_numpy()
    vol     = df["vol_sum"].astype("float32").to_numpy()
    flow    = df["signed_qty_sum"].astype("float32").to_numpy()
    ntr     = df["n_trades"].astype("float32").to_numpy()
    lgc     = df["large_cnt"].astype("float32").to_numpy()
    dol_all = df["dollar_sum"].astype("float32").to_numpy()
    dol_big = df["large_dollar_sum"].astype("float32").to_numpy()

    logp = np.log(price, where=price>0, out=np.zeros_like(price)).astype("float32")
    dlog = np.empty_like(logp); dlog[0]=0.0; dlog[1:] = np.diff(logp).astype("float32")
    absr = np.abs(dlog).astype("float32")

    def csum(a): return np.cumsum(np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)).astype("float64")
    cs_ntr, cs_vol, cs_flow, cs_lgc = csum(ntr), csum(vol), csum(flow), csum(lgc)
    cs_absr, cs_dlog                = csum(absr), csum(dlog)
    cs_flow2, cs_rQ                 = csum(flow*flow), csum(dlog*flow)
    cs_ones                         = csum(np.ones(n, dtype="float32"))
    cs_dol_all, cs_dol_big          = csum(dol_all), csum(dol_big)

    end_idx = np.arange(n) - 1
    end_idx[end_idx < 0] = -1

    def start_idx_for(delta_ns: int) -> np.ndarray:
        return np.searchsorted(idx, idx - delta_ns, side="left")

    def window_sum(cs: np.ndarray, s: np.ndarray) -> np.ndarray:
        end_cum   = np.where(end_idx >= 0, cs[end_idx], 0.0)
        start_pre = s - 1
        start_cum = np.where(start_pre >= 0, cs[start_pre], 0.0)
        return (end_cum - start_cum).astype("float32")

    out = {}
    eps = np.float32(1e-9)

    for L, R in pairs:
        sL = start_idx_for(L); sR = start_idx_for(R)
        tag = f"{L/1e9:.0f}s_{R/1e9:.0f}s"

        breadth = window_sum(cs_ntr,  sR) - window_sum(cs_ntr,  sL)
        v_sum   = window_sum(cs_vol,  sR) - window_sum(cs_vol,  sL)
        q_flow  = window_sum(cs_flow, sR) - window_sum(cs_flow, sL)
        lg_sum  = window_sum(cs_lgc,  sR) - window_sum(cs_lgc,  sL)

        dol_all_ring = window_sum(cs_dol_all, sR) - window_sum(cs_dol_all, sL)
        dol_big_ring = window_sum(cs_dol_big, sR) - window_sum(cs_dol_big, sL)

        out[f"breadth_{tag}"]             = breadth
        out[f"volume_all_{tag}"]          = v_sum
        out[f"large_share_count_{tag}"]   = (lg_sum / (breadth + eps)).astype("float32")
        out[f"large_share_notional_{tag}"]= (dol_big_ring / (dol_all_ring + eps)).astype("float32")

        # Amihud (abs return over dollar notional)
        num_absr = window_sum(cs_absr, sR) - window_sum(cs_absr, sL)
        out[f"amihud_rs_{tag}"] = (num_absr / (dol_all_ring + eps)).astype("float32")

        # Kyle's lambda (OLS) in the ring: cov(r,flow)/var(flow)
        Sr  = window_sum(cs_dlog, sR)  - window_sum(cs_dlog, sL)
        SQ  = q_flow
        SrQ = window_sum(cs_rQ,  sR)   - window_sum(cs_rQ,  sL)
        SQ2 = window_sum(cs_flow2, sR) - window_sum(cs_flow2, sL)
        n_w = np.maximum(window_sum(cs_ones, sR) - window_sum(cs_ones, sL), 1.0)
        cov_rQ = (SrQ - (Sr * SQ) / n_w)
        var_Q  = (SQ2 - (SQ * SQ) / n_w)
        out[f"lambda_ols_{tag}"] = np.where(var_Q > 1e-12, cov_rQ / var_Q, np.nan).astype("float32")

    feat = (pd.DataFrame(out, index=df.index)
              .replace([np.inf,-np.inf], np.nan)
              .fillna(0.0)
              .astype("float32"))
    return feat


# ---------------- Build one day (worker) -------------------
def build_stream_day(asset: str, date_str: str, role: str, trough_iso_list: list[str]) -> pd.DataFrame | None:
    d = pd.Timestamp(date_str).tz_localize("UTC")
    tr = load_agg_day(asset, d)
    if tr is None or tr.empty:
        return None

    agg  = preprocess_trades(tr)
    buck = bucketize(agg, BUCKET_MS)
    feat = add_slice_features_fast(buck)
    feat.index = pd.DatetimeIndex(feat.index).tz_convert("UTC")

    # labels
    y = pd.Series(0, index=feat.index, dtype="int8")
    if role == "pos" and trough_iso_list:
        for iso in trough_iso_list:
            t_trough = pd.Timestamp(iso).tz_convert("UTC")
            t0 = t_trough - H_PRE
            y.loc[(feat.index >= t0) & (feat.index < t_trough)] = 1

    # reproducible sampling per (asset, date)
    key  = f"{asset}_{date_str}"
    seed = (zlib.crc32(key.encode("utf-8")) + RNG_SEED) % (2**32)
    rng  = np.random.default_rng(seed)

    pos_idx = np.flatnonzero(y.values == 1)
    neg_idx = np.flatnonzero(y.values == 0)

    if POS_MAX_PER_DAY is not None and len(pos_idx) > POS_MAX_PER_DAY:
        pos_idx = np.sort(rng.choice(pos_idx, size=POS_MAX_PER_DAY, replace=False))

    # in build_stream_day(...)
    if role == "pos":
        k = NEG_PER_DAY_POS
    elif role == "near_miss":
        k = NEG_PER_DAY_NEAR
    else:  # quiet
        k = NEG_PER_DAY_QUIET

    if len(neg_idx) > k:
        neg_idx = np.sort(rng.choice(neg_idx, size=k, replace=False))

    pos = feat.iloc[pos_idx]
    neg = feat.iloc[neg_idx]
    out = pd.concat([pos, neg]).sort_index()
    out["y"] = 0
    if len(pos_idx):
        out.loc[pos.index, "y"] = 1

    out["asset"]     = asset
    out["date"]      = d.normalize()
    out["role"]      = role
    out["group_key"] = f"{asset}_{date_str}"
    return out


def write_partition(df: pd.DataFrame, root: Path) -> str:
    """Write one day as asset/date partition with a unique file name. Returns file path."""
    asset    = str(df["asset"].iloc[0])
    date_str = str(pd.Timestamp(df["date"].iloc[0]).date())
    part_dir = root / f"asset={asset}" / f"date={date_str}"
    part_dir.mkdir(parents=True, exist_ok=True)

    # ensure index is a column 't'
    df2 = df.reset_index().rename(columns={"index": "t"})
    # Store as parquet (pyarrow)
    fname = f"part-{uuid.uuid4().hex}.parquet"
    fpath = part_dir / fname
    df2.to_parquet(fpath, engine="pyarrow", compression="snappy", index=False)
    return str(fpath)


def process_one_day(task: dict) -> dict:
    """Worker entry: build -> write -> return summary."""
    asset = task["asset"]
    date_str = task["date_str"]
    role  = task["role"]
    troughs = task["troughs"]

    df = build_stream_day(asset, date_str, role, troughs)
    if df is None or df.empty:
        return {"asset": asset, "date": date_str, "role": role, "rows": 0, "pos": 0, "neg": 0, "path": ""}

    pos = int(df["y"].sum())
    rows= int(len(df))
    neg = int(rows - pos)
    out_path = write_partition(df, OUT_DATASET_DIR)
    return {"asset": asset, "date": date_str, "role": role, "rows": rows, "pos": pos, "neg": neg, "path": out_path}


# -------------------------- Main --------------------------
def main():
    t0 = time.time()

    # Load events → map (asset,date) → list of trough ISO strings
    ev = pd.read_csv(EVENTS_CSV, parse_dates=["date","t_trough_final"])
    ev["date"] = ensure_utc(ev["date"])
    ev["t_trough_final"] = ensure_utc(ev["t_trough_final"])
    ev = ev.dropna(subset=["asset","date"])
    ev["date_str"] = ev["date"].dt.strftime("%Y-%m-%d")
    trough_map = {}
    for r in ev.dropna(subset=["t_trough_final"]).itertuples(index=False):
        key = (r.asset, r.date_str)
        trough_map.setdefault(key, []).append(pd.Timestamp(r.t_trough_final).isoformat())

    # Load manifest (assets/dates/roles to process)
    mf = pd.read_csv(MANIFEST)
    mf["date"] = ensure_utc(mf["date"])
    mf = mf.dropna(subset=["asset","date","role"])
    mf["date_str"] = mf["date"].dt.strftime("%Y-%m-%d")

    tasks = []
    for r in mf.itertuples(index=False):
        key = (r.asset, r.date_str)
        tasks.append({
            "asset": r.asset,
            "date_str": r.date_str,
            "role": r.role,
            "troughs": trough_map.get(key, []),
        })

    if not tasks:
        print("No tasks to process — check MANIFEST and EVENTS.")
        return

    OUT_DATASET_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Processing {len(tasks)} day-level tasks with {MAX_WORKERS} workers …")
    rows_total = pos_total = neg_total = 0
    written_files = []

    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futs = [ex.submit(process_one_day, t) for t in tasks]
        for i, fut in enumerate(as_completed(futs), 1):
            res = fut.result()
            rows_total += res["rows"]
            pos_total  += res["pos"]
            neg_total  += res["neg"]
            if res["path"]:
                written_files.append(res["path"])
            if i % 10 == 0 or i == len(futs):
                print(f"  done {i}/{len(futs)} | files: {len(written_files)} | rows: {rows_total} (pos {pos_total}, neg {neg_total})")

    print(f"\nWrote {len(written_files)} partition files to {OUT_DATASET_DIR}")
    print(f"Total rows: {rows_total} | positives: {pos_total} | negatives: {neg_total}")
    print(f"Elapsed: {time.time()-t0:.1f}s")

    # Optional: collect dataset to a single parquet (warning: RAM heavy)
    if WRITE_SINGLE_PARQUET:
        try:
            import pyarrow.dataset as ds, pyarrow.parquet as pq, pyarrow as pa
            dataset = ds.dataset(str(OUT_DATASET_DIR), format="parquet")
            table   = dataset.to_table()  # loads all partitions
            pq.write_table(table, str(OUT_SINGLE_PARQUET), compression="snappy")
            print(f"Also wrote single file -> {OUT_SINGLE_PARQUET}")
        except Exception as e:
            print(f"[warn] Could not write single parquet: {e}")


if __name__ == "__main__":
    import multiprocessing as mp
    mp.freeze_support()
    main()
