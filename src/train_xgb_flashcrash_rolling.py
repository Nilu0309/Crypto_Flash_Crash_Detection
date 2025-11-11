# Robust XGBoost training for flash-crash detection with:
# - Year split: TRAIN=2021, VALID=2022, TEST>=2023
# - Strict pre-trough evaluation window (no last-second credit)
# - VALID-only thresholding (no test peeking)
# - Events filtered & deduped to one (asset,date), date derived from trough UTC
# - FA-calibrated operating point (quiet-day median false alerts/day target)
# - K-of-N confirmations with adaptive window (no cooldown applied before K-of-N)
# - Cooldown/merge only for operational FA/day counting
# - Year-by-year test breakdown

import json
from pathlib import Path
from glob import glob

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import xgboost as xgb
from sklearn.metrics import average_precision_score, roc_auc_score, precision_recall_curve

# --- NEW: plotting & optional SHAP ---
import os
import matplotlib.pyplot as plt
from collections import OrderedDict

try:
    import shap  # optional
    _HAVE_SHAP = True
except Exception:
    _HAVE_SHAP = False

# ========= CONFIG =========
FEATURES_PARQUET   = Path("./features_stream_dataset")   # or single file .parquet
FEATURE_COLS_JSON  = Path("feature_cols.json")
EVENTS_CSV         = Path("events_refined_with_trades.csv")
ASSETS             = ["BTCUSDT", "ETHUSDT"]

# Fixed year split
YEAR_TRAIN      = (2021,)        # train only on 2021
YEAR_VALID      = (2022,)        # validate on 2022
YEAR_TEST_FROM  = 2023           # test on years >= this

# Labeling window must match your features pipeline
H_PRE = pd.Timedelta(seconds=120)
EXCLUDE_LAST_S = 30  # exclude the last 30s before trough from credit

# OPTIONAL: train-only feature caps (to tame drift). Disabled by default.
CAP_FEATURES_Q = None   # set to 0.99 to enable 99th percentile capping from TRAIN

MODEL_DIR = Path("./models_per_asset_yearsplit")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# ----- EVENT FILTERS (tighten to your definition of "flash crash") -----
EVENT_FILTERS = dict(
    require_role_values=["pos"],     # only keep these roles if 'role' exists; [] disables role filter
    require_has_trade=True,          # keep only has_trade==True if 'has_trade' exists
    min_abs_drawdown_pct=None        # e.g., 0.03 -> require >=3% absolute drawdown if 'dd_pct' exists
)

# XGBoost defaults (modest capacity to avoid silly overfit)
XGB_PARAMS = dict(
    objective="binary:logistic",
    eval_metric="aucpr",
    tree_method="hist",
    learning_rate=0.04,
    max_depth=4,
    subsample=0.9,
    colsample_bytree=0.9,
    reg_lambda=2.0,
    min_child_weight=1.0,
    seed=123,
)

# ========= Detection policy (primary operating point) =========
TARGET_FA_PER_DAY = 0.20   # target median false alerts/day on quiet days (VALID-calibrated)
MERGE_SEC         = 5      # merge close alert spikes for FA counting
COOLDOWN_SEC      = 20     # minimal spacing between counted alerts (FA counting)
K_OF_N            = (2, 20)  # base K-of-N for detections; auto-adapts to time grid

# ========= HELPERS =========
def ensure_utc(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce", utc=True)

def ensure_t_column(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    if "t" in d.columns:
        d["t"] = ensure_utc(d["t"]); return d
    for cand in ("ts","time","timestamp","bucket_time","datetime"):
        if cand in d.columns:
            ts = ensure_utc(d[cand])
            if ts.notna().any():
                d["t"] = ts; return d
    if isinstance(d.index, pd.DatetimeIndex):
        d = d.reset_index().rename(columns={"index":"t"})
        d["t"] = ensure_utc(d["t"]); return d
    dt_cols = d.select_dtypes(include=["datetime64[ns]","datetimetz"]).columns
    if len(dt_cols):
        d["t"] = ensure_utc(d[dt_cols[0]]); return d
    raise KeyError("No usable timestamp column found for 't'.")

def load_features_any(path):
    p = Path(path)
    if p.is_file():
        df = pd.read_parquet(p)
        return df

    files = sorted(glob(str(p / "asset=*/date=*/part-*.parquet")))
    if not files:
        raise FileNotFoundError(f"No parquet parts under {p}")

    parts = []
    for f in files:
        dfp = pq.read_table(f).to_pandas(use_threads=False)

        # normalize metadata
        for c in ("asset", "role", "group_key"):
            if c in dfp.columns:
                dfp[c] = dfp[c].astype(str)

        # date → UTC midnight
        if "date" in dfp.columns:
            dfp["date"] = pd.to_datetime(dfp["date"], errors="coerce", utc=True).dt.normalize()
        else:
            raise KeyError(f"'date' column missing in {f}")

        # t → UTC tz-aware if present
        if "t" in dfp.columns:
            if not pd.api.types.is_datetime64_any_dtype(dfp["t"]):
                dfp["t"] = pd.to_datetime(dfp["t"], errors="coerce", utc=True)
            elif getattr(dfp["t"].dtype, "tz", None) is None:
                dfp["t"] = dfp["t"].dt.tz_localize("UTC")
            else:
                dfp["t"] = dfp["t"].dt.tz_convert("UTC")

        parts.append(dfp)

    df = pd.concat(parts, ignore_index=True)

    if "y" not in df.columns:
        cols = list(df.columns)
        sample = df.head(2).to_dict(orient="records")
        raise KeyError(
            "'y' column not found in loaded features. "
            f"Available columns: {cols}\nSample rows: {sample}\n"
        )
    return df

def has_both_classes(y: np.ndarray) -> bool:
    if y.size == 0:
        return False
    uniq = np.unique(y)
    return (uniq.size == 2) and (y.sum() > 0) and ((y == 0).sum() > 0)

def choose_thresholds(y_true, y_prob, neg_quantile=0.99):
    pr, rc, thr = precision_recall_curve(y_true, y_prob)
    f1 = 2 * pr * rc / (pr + rc + 1e-9)
    i  = int(np.argmax(f1))
    best_f1_thr = float(thr[i]) if i < len(thr) else 0.5
    prec_f1, rec_f1 = float(pr[i]), float(rc[i])
    neg_scores = y_prob[y_true == 0]
    q_thr = float(np.quantile(neg_scores, neg_quantile)) if len(neg_scores) else best_f1_thr
    return best_f1_thr, prec_f1, rec_f1, q_thr

def get_best_iter(booster: xgb.Booster, fallback_rounds: int) -> int:
    if hasattr(booster, "best_iteration") and booster.best_iteration is not None:
        return int(booster.best_iteration)
    if hasattr(booster, "best_ntree_limit") and booster.best_ntree_limit:
        return int(booster.best_ntree_limit) - 1
    try:
        return int(booster.num_boosted_rounds()) - 1
    except Exception:
        return int(fallback_rounds) - 1

def year_split_masks(dfa_asset: pd.DataFrame,
                     train_years=YEAR_TRAIN,
                     valid_years=YEAR_VALID,
                     test_from_year=YEAR_TEST_FROM):
    dfa = dfa_asset.copy()
    dfa["date_norm"] = ensure_utc(dfa["date"]).dt.normalize()
    dfa["year"] = dfa["date_norm"].dt.year
    train_mask = dfa["year"].isin(train_years)
    val_mask   = dfa["year"].isin(valid_years)
    test_mask  = dfa["year"] >= int(test_from_year)
    return train_mask.astype(bool), val_mask.astype(bool), test_mask.astype(bool)

def fit_feature_caps(train_df, cols, q=0.99):
    caps = {}
    for c in cols:
        x = train_df[c].replace([np.inf,-np.inf], np.nan).dropna()
        if len(x):
            try:
                caps[c] = float(np.quantile(x.astype(float), q))
            except Exception:
                pass
    return caps

def apply_feature_caps(df, caps):
    if not caps:
        return df
    d = df.copy()
    for c, cap in caps.items():
        if c in d.columns:
            d[c] = np.clip(d[c].astype(float), a_min=None, a_max=cap)
    return d

# ---------- PLOTTING / IMPORTANCE UTILS ----------
PLOTS_DIR = Path("./plots"); PLOTS_DIR.mkdir(parents=True, exist_ok=True)

def _savefig(path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()

def plot_pr_roc(y, p, title_prefix, out_prefix):
    from sklearn.metrics import precision_recall_curve, roc_curve, average_precision_score, roc_auc_score
    if y is None or p is None or len(y)==0:
        return
    ap = average_precision_score(y, p) if y.sum()>0 else float('nan')
    auc= roc_auc_score(y, p) if len(np.unique(y))==2 else float('nan')

    # PR
    pr, rc, _ = precision_recall_curve(y, p)
    plt.figure(figsize=(7,4))
    plt.plot(rc, pr, lw=2)
    plt.xlabel("Recall"); plt.ylabel("Precision"); 
    plt.title(f"{title_prefix} — PR curve (AP={ap:.3f})")
    _savefig(PLOTS_DIR / f"{out_prefix}_PR.png")

    # ROC
    if len(np.unique(y))==2:
        fpr, tpr, _ = roc_curve(y, p)
        plt.figure(figsize=(7,4))
        plt.plot(fpr, tpr, lw=2)
        plt.plot([0,1],[0,1], ls="--", lw=1)
        plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
        plt.title(f"{title_prefix} — ROC curve (AUC={auc:.3f})")
        _savefig(PLOTS_DIR / f"{out_prefix}_ROC.png")

def plot_threshold_scan(scan_records, target_fa, chosen_thr, out_png, split_name):
    if not scan_records:
        return
    import pandas as pd
    df = pd.DataFrame(scan_records)
    df = df.sort_values("thr")
    plt.figure(figsize=(7.5,4.5))
    ax1 = plt.gca()
    ax2 = ax1.twinx()
    ax1.plot(df["thr"], df["det"], lw=2, label="Detection rate")
    ax2.plot(df["thr"], df["fa_med"], lw=2, ls="--", label="FA/day (median)")

    ax1.set_xlabel("Threshold"); ax1.set_ylabel("Detection rate")
    ax2.set_ylabel("FA/day (median)")
    ax2.axhline(target_fa, ls=":", lw=1.5)
    ax1.axvline(chosen_thr, ls="-.", lw=1)
    plt.title(f"Threshold scan ({split_name}) — pick at thr={chosen_thr:.4f}, target FA/day={target_fa}")
    _savefig(out_png)

def xgb_gain_importance(bst: xgb.Booster, max_k=25):
    # returns OrderedDict feature -> gain
    gain = bst.get_score(importance_type="gain") or {}
    # ensure natural feature ordering for readability
    srt = sorted(gain.items(), key=lambda kv: kv[1], reverse=True)[:max_k]
    return OrderedDict(srt)

def plot_barh_from_mapping(mapping, title, out_png, csv_out=None):
    import pandas as pd
    if not mapping: 
        return
    items = list(mapping.items())
    df = pd.DataFrame(items, columns=["feature","score"])
    if csv_out:
        Path(csv_out).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(csv_out, index=False)
    df = df.iloc[::-1]  # plot from smallest to largest upward
    plt.figure(figsize=(8, 0.35*len(df)+1.5))
    plt.barh(df["feature"], df["score"])
    plt.title(title)
    plt.xlabel("Importance")
    plt.tight_layout()
    _savefig(out_png)

def permutation_importance_ap(bst: xgb.Booster, X: pd.DataFrame, y: np.ndarray,
                              n_repeats=3, random_state=123, max_features=None):
    """Permutation importance measured by drop in Average Precision (VALID preferred)."""
    if X is None or len(X)==0 or y is None or len(y)==0:
        return OrderedDict()
    rng = np.random.default_rng(random_state)
    base = average_precision_score(y, bst.predict(xgb.DMatrix(X)))
    cols = list(X.columns)
    if max_features and max_features < len(cols):
        # screen by xgb gain first if available, else random subset
        try:
            gmap = bst.get_score(importance_type="gain") or {}
            ranked = [c for c,_ in sorted(gmap.items(), key=lambda kv: kv[1], reverse=True)]
            cols = ranked[:max_features] + [c for c in cols if c not in ranked][:max(0, max_features-len(ranked))]
            cols = cols[:max_features]
        except Exception:
            cols = list(rng.choice(cols, size=max_features, replace=False))
    drops = {}
    X_work = X.copy()
    for c in cols:
        scores = []
        col = X_work[c].to_numpy().copy()
        for _ in range(n_repeats):
            rng.shuffle(col)
            X_work[c] = col
            s = average_precision_score(y, bst.predict(xgb.DMatrix(X_work)))
            scores.append(base - s)
        X_work[c] = X[c]  # restore
        drops[c] = float(np.mean(scores))
    # sort desc by importance (bigger drop = more important)
    return OrderedDict(sorted(drops.items(), key=lambda kv: kv[1], reverse=True))

# ===== EXTRA PLOTTING HELPERS (drop-in) =====
def _closest_record_by_thr(records, thr):
    return min(records, key=lambda r: abs(float(r["thr"]) - float(thr)))

def plot_alert_budget_curve(scan_records, chosen_thr, best_f1_thr, out_png, split_name):
    import pandas as pd
    if not scan_records: return
    df = pd.DataFrame(scan_records)
    # collapse by FA/day: keep max recall for each FA/day (clean curve)
    df = (df.groupby("fa_med", as_index=False)["det"].max()
            .sort_values(["fa_med","det"]))
    plt.figure(figsize=(6.6,4.2))
    plt.plot(df["fa_med"], df["det"], marker="o", lw=2)
    ch = _closest_record_by_thr(scan_records, chosen_thr)
    plt.scatter([ch["fa_med"]], [ch["det"]], s=70, marker="D", label=f"Conservative\nthr={chosen_thr:.4f}")
    if best_f1_thr is not None:
        bf = _closest_record_by_thr(scan_records, best_f1_thr)
        plt.scatter([bf["fa_med"]], [bf["det"]], s=70, marker="s", label=f"Best-F1\nthr={best_f1_thr:.4f}")
    plt.xlabel("False alerts per day (median, quiet days)")
    plt.ylabel("Event recall")
    plt.title(f"{split_name} — Alert budget curve")
    plt.legend(frameon=False)
    _savefig(out_png)

def _fa_list_quiet(df_part: pd.DataFrame, probs: np.ndarray, thr: float,
                   merge_sec:int, cooldown_sec:int):
    if len(df_part) == 0: return []
    d = df_part.copy().sort_values(["asset","date","t"])
    d["p"] = probs; d["pred1"] = (d["p"] >= float(thr)).astype(int)
    fac = []
    for (a, dt), g in d.groupby(["asset","date"], sort=False):
        if g["y"].max() != 0:  # skip non-quiet days
            continue
        kept, last = [], None
        for row in g.itertuples(index=False):
            if row.pred1:
                tt = row.t
                if last is None or (tt - last).total_seconds() > max(merge_sec, cooldown_sec):
                    kept.append(tt); last = tt
        fac.append(len(kept))
    return fac

def plot_fa_box(df_part, probs, thr, split_name, out_png, merge_sec=5, cooldown_sec=20):
    fac = _fa_list_quiet(df_part, probs, thr, merge_sec, cooldown_sec)
    if not fac: return
    plt.figure(figsize=(4.8,4.0))
    plt.boxplot(fac, vert=True, showfliers=False)
    plt.ylabel("False alerts per day (quiet days)")
    plt.title(f"{split_name} — FA/day distribution @ thr={thr:.4f}")
    _savefig(out_png)

def plot_score_hist(df_part, probs, split_name, out_png):
    y = df_part["y"].to_numpy()
    bins = np.linspace(0, 1, 50)
    plt.figure(figsize=(6.8,4.2))
    plt.hist(probs[y==0], bins=bins, density=True, alpha=0.6, label="y=0 (quiet)")
    if y.sum() > 0:
        plt.hist(probs[y==1], bins=bins, density=True, alpha=0.6, label="y=1 (pre-trough windows)")
    plt.xlabel("Predicted score")
    plt.ylabel("Density")
    plt.title(f"{split_name} — Score distributions")
    plt.legend(frameon=False)
    _savefig(out_png)

def plot_reliability_simple(y_true, y_prob, split_name, out_png, n_bins=15):
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    bins = np.linspace(0, 1, n_bins+1)
    which = np.digitize(y_prob, bins) - 1
    xs, ys, ns = [], [], []
    for b in range(n_bins):
        m = (which == b)
        if not m.any(): continue
        xs.append(float(y_prob[m].mean()))
        ys.append(float(y_true[m].mean()))
        ns.append(int(m.sum()))
    brier = float(np.mean((y_prob - y_true)**2))
    plt.figure(figsize=(5.8,5.2))
    plt.plot([0,1],[0,1], ls="--", lw=1)
    sizes = np.clip(np.array(ns)/max(ns)*220, 24, 220)
    plt.scatter(xs, ys, s=sizes)
    plt.xlabel("Mean predicted probability")
    plt.ylabel("Empirical positive rate")
    plt.title(f"{split_name} — Reliability (Brier={brier:.3f})")
    _savefig(out_png)

def _wilson_ci(k, n, z=1.96):
    if n == 0: return (np.nan, np.nan)
    p = k / n
    denom = 1 + z*z/n
    center = (p + z*z/(2*n)) / denom
    margin = z*np.sqrt(p*(1-p)/n + z*z/(4*n*n)) / denom
    return (max(0.0, center - margin), min(1.0, center + margin))

def plot_year_detection_ci(df_part, prob_series: pd.Series, thr: float,
                           events_asset_df: pd.DataFrame, split_name, out_png):
    if len(df_part) == 0: return
    d = df_part.copy()
    d["year"] = d["date"].dt.year
    years = sorted(d["year"].dropna().unique())
    vals, los, his = [], [], []
    for yy in years:
        sub = d[d["year"]==yy]
        ev_sub = events_asset_df[events_asset_df["date"].dt.year == yy]
        p_sub = prob_series.loc[sub.index].to_numpy()
        em = event_metrics_pretrough_strict_kofn(
            sub, p_sub, thr, ev_sub, H_PRE, EXCLUDE_LAST_S,
            k_of_n=K_OF_N, merge_sec=MERGE_SEC, cooldown_sec=COOLDOWN_SEC
        )
        n = int(em["events_evaluated"]); k = len(em.get("lead_values", []))
        rec = k/n if n>0 else np.nan
        lo, hi = _wilson_ci(k, n) if n>0 else (np.nan, np.nan)
        vals.append(rec); los.append(lo); his.append(hi)
    # plot
    plt.figure(figsize=(6.6,4.0))
    x = np.arange(len(years))
    yerr = [np.array(vals)-np.array(los), np.array(his)-np.array(vals)]
    plt.bar(x, vals, yerr=yerr, capsize=3)
    plt.xticks(x, years)
    plt.ylim(0, 1)
    plt.ylabel("Event recall")
    plt.title(f"{split_name} — Year-by-year recall @ thr={thr:.4f}")
    _savefig(out_png)

def save_operating_points_table(asset, split_name, df_part, prob_series, events_df,
                                thr_cons, thr_f1, out_csv):
    rows = []
    for tag, thr in [("Conservative", thr_cons), ("Best-F1", thr_f1)]:
        if thr is None: continue
        em = event_metrics_pretrough_strict_kofn(
            df_part, prob_series.to_numpy(), thr, events_df, H_PRE, EXCLUDE_LAST_S,
            k_of_n=K_OF_N, merge_sec=MERGE_SEC, cooldown_sec=COOLDOWN_SEC
        )
        fa = false_alerts_stats_quiet(df_part, prob_series.to_numpy(), thr, MERGE_SEC, COOLDOWN_SEC)
        rows.append(dict(
            asset=asset, split=split_name, operating_point=tag, thr=float(thr),
            det=float(em["event_detection_rate"]),
            lead_med_s=float(em["lead_median_s"]),
            fa_med=float(fa["alerts_med"]), fa_p90=float(fa["alerts_p90"]),
            events=int(em["events_evaluated"])
        ))
    pd.DataFrame(rows).to_csv(out_csv, index=False)

# ----- Load & filter events, dedupe to one (asset,date), date from trough -----
def apply_event_filters(ev: pd.DataFrame) -> pd.DataFrame:
    e = ev.copy()
    rf = EVENT_FILTERS

    if rf.get("require_role_values") and "role" in e.columns and len(rf["require_role_values"]) > 0:
        e = e[e["role"].astype(str).isin(rf["require_role_values"])]

    if rf.get("require_has_trade") and "has_trade" in e.columns:
        e = e[e["has_trade"].astype(bool)]

    dd_col = "dd_pct" if "dd_pct" in e.columns else ("drawdown_pct" if "drawdown_pct" in e.columns else None)
    if rf.get("min_abs_drawdown_pct") is not None and dd_col is not None:
        e = e[e[dd_col].abs() >= float(rf["min_abs_drawdown_pct"])]

    return e

def plot_leadtime_ecdf(lead_secs, title, out_png):
    if len(lead_secs)==0: return
    x = np.sort(np.asarray(lead_secs, dtype=float))
    y = np.arange(1, len(x)+1)/len(x)
    plt.figure(figsize=(6,4))
    plt.step(x, y, where="post")
    plt.xlabel("Lead time (seconds)"); plt.ylabel("Cumulative fraction of detected events")
    plt.title(title)
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=200, bbox_inches="tight"); plt.close()


# ========= NEW: alert compression & K-of-N utilities =========
def _median_step_seconds(df_part: pd.DataFrame) -> float:
    if len(df_part) < 2:
        return 1.0
    g = df_part.sort_values("t")["t"].diff().dropna().dt.total_seconds()
    g = g[(g > 0) & (g < 3600)]
    return float(np.median(g)) if len(g) else 1.0

def _compute_alert_times(df_part: pd.DataFrame, probs: np.ndarray, thr: float,
                         merge_sec:int, cooldown_sec:int, for_detection: bool = False):
    """
    Return {(asset,date): [t1,t2,...]} of alert timestamps.
    If for_detection=True, DO NOT apply cooldown (so K-of-N can see multiple alerts).
    Always keep a small merge to reduce flicker.
    """
    if len(df_part) == 0:
        return {}
    d = df_part.copy().sort_values(["asset","date","t"])
    d["p"] = probs
    d["pred1"] = (d["p"] >= float(thr)).astype(bool)
    times = {}
    for (a, dt), g in d.groupby(["asset","date"], sort=False):
        g = g.sort_values("t")
        kept = []
        last = None
        eff_cd = 0 if for_detection else max(merge_sec, cooldown_sec)
        eff_merge = max(1, merge_sec)
        for row in g.itertuples(index=False):
            if row.pred1:
                tt = row.t
                if last is None:
                    kept.append(tt); last = tt
                else:
                    gap = (tt - last).total_seconds()
                    if gap > max(eff_merge, eff_cd):
                        kept.append(tt); last = tt
        times[(a, dt)] = kept
    return times

def _choose_kofn(df_part: pd.DataFrame, base_kofn=(2,20)):
    """
    Adapt K-of-N to the dataset’s time granularity.
    For coarse grids, relax to K=1 and enlarge N.
    """
    step = _median_step_seconds(df_part)
    K, N = int(base_kofn[0]), int(base_kofn[1])
    if step >= 10:         # very coarse grid (≥10s)
        return (1, max(N, int(4*step)))
    if step >= 5:
        return (1, max(N, int(3*step)))
    # fine grid: keep base but ensure N covers ~4 steps
    return (K, max(N, int(max(20, 4*step))))

def _k_of_n_detect(alert_list, window_start, window_end, k:int, n_sec:int):
    if not alert_list:
        return False, np.nan
    arr = [t for t in alert_list if (t >= window_start) and (t < window_end)]
    if len(arr) < k:
        return False, np.nan
    j = 0
    for i in range(len(arr)):
        while j < len(arr) and (arr[j] - arr[i]).total_seconds() <= n_sec:
            j += 1
        if (j - i) >= k:
            return True, np.nan
    return False, np.nan

def event_metrics_pretrough_strict_kofn(
    test_df: pd.DataFrame,
    probs: np.ndarray,
    thr: float,
    events_asset_df: pd.DataFrame,
    H_PRE: pd.Timedelta,
    exclude_last_s: int,
    k_of_n=(2,20),
    merge_sec=5,
    cooldown_sec=20
) -> dict:
    """Valid window: [t_trough - H_PRE, t_trough - exclude_last_s). Requires K-of-N confirmations."""
    slab = test_df.copy().reset_index(drop=True).sort_values(["asset","date","t"])
    K, N = _choose_kofn(slab, k_of_n)
    alert_times = _compute_alert_times(slab, probs, thr, merge_sec, cooldown_sec, for_detection=True)

    rows = []
    for r in events_asset_df.itertuples(index=False):
        key = (r.asset, r.date)
        at = alert_times.get(key, [])
        if not at:
            rows.append({"hit":0, "lead_s":np.nan}); continue
        t1 = r.t_trough_final
        t0 = t1 - H_PRE
        if exclude_last_s > 0:
            t1 = t1 - pd.Timedelta(seconds=int(exclude_last_s))
            if t1 <= t0:
                rows.append({"hit":0,"lead_s":np.nan}); continue
        hit, _ = _k_of_n_detect(at, t0, t1, K, N)
        if not hit:
            rows.append({"hit":0, "lead_s":np.nan})
        else:
            cand = [t for t in at if (t >= t0) and (t < t1)]
            lead = (r.t_trough_final - min(cand)).total_seconds() if len(cand) else np.nan
            rows.append({"hit":1, "lead_s":lead})

    hits = pd.DataFrame(rows)
    if hits.empty:
        return {"event_detection_rate": np.nan,
                "lead_median_s": np.nan, "lead_p10_s": np.nan, "lead_p90_s": np.nan,
                "events_evaluated": 0}
    det_rate = float(hits["hit"].mean())
    lead_vals = hits["lead_s"].dropna().values
    lead_med = float(np.nanmedian(lead_vals)) if len(lead_vals) else np.nan
    lead_p10 = float(np.nanpercentile(lead_vals, 10)) if len(lead_vals) else np.nan
    lead_p90 = float(np.nanpercentile(lead_vals, 90)) if len(lead_vals) else np.nan
    return {
    "event_detection_rate": det_rate,
    "lead_median_s": lead_med, "lead_p10_s": lead_p10, "lead_p90_s": lead_p90,
    "events_evaluated": int(len(events_asset_df)),
    "lead_values": [float(x) for x in lead_vals]  # <— add this
}


def false_alerts_stats_quiet(df_part: pd.DataFrame, probs: np.ndarray, thr: float,
                             merge_sec:int, cooldown_sec:int):
    """Count alerts/day on quiet days using merged + cooldown alerts (operational counting)."""
    if len(df_part) == 0:
        return {"zero_day_count":0,"alerts_med":np.nan,"alerts_p90":np.nan}
    d = df_part.copy().sort_values(["asset","date","t"])
    d["p"] = probs
    d["pred1"] = (d["p"] >= float(thr)).astype(int)
    quiet_keys = (d.groupby(["asset","date"])["y"].max() == 0)
    quiet_idx = quiet_keys[quiet_keys].index
    alerts = []
    for a, dt in quiet_idx:
        g = d[(d["asset"]==a) & (d["date"]==dt)].sort_values("t")
        kept = []
        last = None
        for row in g.itertuples(index=False):
            if row.pred1:
                tt = row.t
                if last is None or (tt - last).total_seconds() > max(merge_sec, cooldown_sec):
                    kept.append(tt); last = tt
        alerts.append(int(len(kept)))
    if len(alerts) == 0:
        return {"zero_day_count":0, "alerts_med":0.0, "alerts_p90":0.0}
    return {
        "zero_day_count": int(len(alerts)),
        "alerts_med": float(np.median(alerts)),
        "alerts_p90": float(np.percentile(alerts, 90)),
    }

def _best_f1_and_q99(y_true, y_prob):
    pr, rc, thr = precision_recall_curve(y_true, y_prob)
    f1 = 2 * pr * rc / (pr + rc + 1e-9)
    i  = int(np.argmax(f1))
    best_f1_thr = float(thr[i]) if i < len(thr) else 0.5
    neg = y_prob[y_true == 0]
    q99 = float(np.quantile(neg, 0.99)) if len(neg) else best_f1_thr
    return best_f1_thr, q99

def pick_thr_by_quiet_fa(df_valid: pd.DataFrame, p_valid: np.ndarray,
                         events_valid: pd.DataFrame,
                         target_fa_per_day: float,
                         H_PRE: pd.Timedelta,
                         exclude_last_s:int,
                         k_of_n=(2,20),
                         merge_sec=5, cooldown_sec=20):
    """
    Scan a wide, data-driven threshold grid.
    Choose the threshold that maximizes detection subject to FA/day <= target.
    Fallback: smallest FA, then highest detection, then higher thr.
    """
    if len(df_valid) == 0 or len(p_valid) == 0:
        return None, {}
    d = df_valid.copy().sort_values(["asset","date","t"]).reset_index(drop=True)
    yv = d["y"].to_numpy()

    # candidate thresholds: negatives/positives quantiles + uniform grid
    neg_scores = p_valid[yv == 0]
    pos_scores = p_valid[yv == 1]
    cand = set()
    if len(neg_scores):
        for q in np.linspace(0.50, 0.999, 120):
            cand.add(float(np.quantile(neg_scores, q)))
    if len(pos_scores):
        for q in np.linspace(0.50, 0.999, 60):
            cand.add(float(np.quantile(pos_scores, q)))
    for t in np.linspace(0.02, 0.99, 60):
        cand.add(float(t))
    bf1, q99 = _best_f1_and_q99(yv, p_valid)
    cand.update([bf1, q99])
    cand_thrs = sorted([float(t) for t in cand if np.isfinite(t)])

    records = []
    for thr in cand_thrs:
        fa = false_alerts_stats_quiet(d, p_valid, thr, merge_sec, cooldown_sec)
        em = event_metrics_pretrough_strict_kofn(
            d, p_valid, thr, events_valid, H_PRE, exclude_last_s,
            k_of_n=k_of_n, merge_sec=merge_sec, cooldown_sec=cooldown_sec
        )
        rec = dict(thr=float(thr),
                   fa_med=fa["alerts_med"], fa_p90=fa["alerts_p90"],
                   det=float(em["event_detection_rate"]),
                   lead_med=float(em["lead_median_s"]) if em["lead_median_s"]==em["lead_median_s"] else np.nan)
        records.append(rec)

    feasible = [r for r in records if r["fa_med"] <= target_fa_per_day + 1e-9]
    if feasible:
        feasible.sort(key=lambda r: (-r["det"], r["fa_med"], -r["thr"]))  # det first, then lower FA, then higher thr
        best = feasible[0]
    else:
        records.sort(key=lambda r: (r["fa_med"], -r["det"], -r["thr"]))
        best = records[0]
    return best["thr"], {"scan": records, "picked": best}

# ========= LOAD DATA =========
df = load_features_any(FEATURES_PARQUET)
df = ensure_t_column(df)
df["date"] = ensure_utc(df["date"]).dt.normalize()
df["y"] = df["y"].astype(int)

# numeric feature columns only (exclude metadata)
if FEATURE_COLS_JSON.exists():
    feature_cols_all = json.loads(FEATURE_COLS_JSON.read_text())
else:
    meta = {"t","asset","date","role","group_key","y"}
    num  = df.select_dtypes(include=["number","bool"]).columns
    feature_cols_all = sorted([c for c in num if c not in meta])

# Load events CSV robustly
events_raw = pd.read_csv(EVENTS_CSV)
if "asset" not in events_raw.columns or "t_trough_final" not in events_raw.columns:
    raise KeyError("events CSV must contain at least 'asset' and 't_trough_final' columns.")
events_raw["t_trough_final"] = ensure_utc(events_raw["t_trough_final"])
events_raw["date"] = events_raw["t_trough_final"].dt.normalize()
events = apply_event_filters(events_raw)
# dedupe to one trough per (asset,date): keep earliest trough
events = (events.sort_values(["asset","date","t_trough_final"])
                .drop_duplicates(["asset","date"], keep="first"))

# ========= TRAIN & EVAL (per asset) =========
for asset in ASSETS:
    print("\n" + "="*90)
    print(f"ASSET: {asset}")
    print("="*90)

    dfa = df[df["asset"] == asset].copy()
    if dfa.empty:
        print("[WARN] No rows for", asset); continue

    # Year split
    tr_mask, va_mask, te_mask = year_split_masks(dfa)
    df_tr = dfa[tr_mask].copy()
    df_va = dfa[va_mask].copy()
    df_te = dfa[te_mask].copy()

    # features
    feat_cols = [c for c in feature_cols_all if c in dfa.columns]
    if not feat_cols:
        raise ValueError("No numeric feature columns found after filtering.")

    # (optional) TRAIN-only feature caps
    caps = {}
    if CAP_FEATURES_Q is not None:
        caps = fit_feature_caps(df_tr, feat_cols, q=float(CAP_FEATURES_Q))
        df_tr = apply_feature_caps(df_tr, caps)
        df_va = apply_feature_caps(df_va, caps)
        df_te = apply_feature_caps(df_te, caps)

    X_tr, y_tr = df_tr[feat_cols], df_tr["y"].to_numpy(dtype=int)
    X_va, y_va = df_va[feat_cols], df_va["y"].to_numpy(dtype=int)
    X_te, y_te = df_te[feat_cols], df_te["y"].to_numpy(dtype=int)

    # spans and counts
    span = lambda d: (d["date"].min().date() if len(d) else "NA",
                      d["date"].max().date() if len(d) else "NA")
    print(f"Train: rows={len(df_tr)} pos={int(y_tr.sum())} neg={int((y_tr==0).sum())} | span {span(df_tr)[0]} → {span(df_tr)[1]}")
    print(f"Valid: rows={len(df_va)} pos={int(y_va.sum())} neg={int((y_va==0).sum())} | span {span(df_va)[0]} → {span(df_va)[1]}")
    print(f"Test : rows={len(df_te)} pos={int(y_te.sum())} neg={int((y_te==0).sum())} | span {span(df_te)[0]} → {span(df_te)[1]}")

    # Event-window sanity (counts from filtered CSV per window)
    ev_asset_all = events[events["asset"] == asset][["asset","date","t_trough_final"]]
    def cnt_in_range(ev, start, end):
        if start is None or end is None:
            return 0
        e = ev[(ev["date"] >= start) & (ev["date"] <= end)]
        return int(e.drop_duplicates(["asset","date"]).shape[0])
    v_start, v_end = (df_va["date"].min(), df_va["date"].max()) if len(df_va) else (None, None)
    t_start, t_end = (df_te["date"].min(), df_te["date"].max()) if len(df_te) else (None, None)
    raw_valid = cnt_in_range(ev_asset_all, v_start, v_end)
    raw_test  = cnt_in_range(ev_asset_all, t_start, t_end)
    print(f"[Events CSV after filters] unique (asset,date) — VALID window: {raw_valid} | TEST window: {raw_test}")

    # Build evaluated event lists by intersecting with slabs
    def events_present_in_slab(df_slab: pd.DataFrame, ev_asset: pd.DataFrame) -> pd.DataFrame:
        if len(df_slab) == 0:
            return ev_asset.iloc[0:0].copy()
        present = df_slab[["asset","date"]].drop_duplicates()
        return (ev_asset.merge(present, on=["asset","date"], how="inner")
                        .sort_values("t_trough_final")
                        .drop_duplicates(["asset","date"], keep="first"))

    Ev_val = events_present_in_slab(df_va, ev_asset_all)
    Ev_tst = events_present_in_slab(df_te, ev_asset_all)
    print(f"Events evaluated — VALID: {len(Ev_val)} | TEST: {len(Ev_tst)}")

    # VALID fallback if needed (derive from TRAIN via GroupKFold; no test peeking)
    used_gkf = False
    if not has_both_classes(y_va):
        from sklearn.model_selection import GroupKFold
        print("[INFO] Validation lacks both classes. Deriving a validation fold from TRAIN via GroupKFold...")
        groups = df_tr["date"].astype(str).values
        gkf = GroupKFold(n_splits=min(5, max(2, len(np.unique(groups)))))
        X_tr_full, y_tr_full = df_tr[feat_cols], df_tr["y"].to_numpy(dtype=int)
        picked = False
        for i_fold, (i_tr2, i_va2) in enumerate(gkf.split(X_tr_full, y_tr_full, groups), 1):
            if has_both_classes(y_tr_full[i_va2]):
                X_va, y_va = X_tr_full.iloc[i_va2], y_tr_full[i_va2]
                X_tr, y_tr = X_tr_full.iloc[i_tr2], y_tr_full[i_tr2]
                df_va = df_tr.iloc[i_va2].copy()
                Ev_val = events_present_in_slab(df_va, ev_asset_all)
                print(f"  Using TRAIN fold {i_fold} as VALID (rows={len(i_va2)}, pos={int(y_va.sum())}).")
                used_gkf = True
                picked = True
                break
        if not picked:
            print("[WARN] Could not find a valid fold with both classes. Proceeding without early stopping.")
            X_va, y_va = None, None

    # imbalance from TRAIN only
    neg, pos = (y_tr==0).sum(), (y_tr==1).sum()
    spw = max(neg / max(pos,1), 1.0)
    print("scale_pos_weight =", round(spw, 2))
    params = dict(XGB_PARAMS, scale_pos_weight=spw)

    # Train
    if X_va is not None:
        dtrain = xgb.DMatrix(X_tr, label=y_tr, feature_names=feat_cols)
        dvalid = xgb.DMatrix(X_va, label=y_va, feature_names=feat_cols)
        bst = xgb.train(params, dtrain,
                        num_boost_round=2000,
                        evals=[(dtrain,"train"), (dvalid,"valid")],
                        early_stopping_rounds=150,
                        verbose_eval=False)
        best_iter = get_best_iter(bst, 2000)

        # thresholds on VALID only
        p_va  = bst.predict(dvalid, iteration_range=(0, best_iter+1))
        va_ap = average_precision_score(y_va, p_va) if y_va.sum() > 0 else np.nan
        va_roc= roc_auc_score(y_va, p_va) if len(np.unique(y_va)) == 2 else np.nan
        best_f1_thr, prec_f1, rec_f1, q99_thr = choose_thresholds(y_va, p_va, neg_quantile=0.99)
        print(f"Best iteration (by VALID early stopping): {best_iter}")
        print(f"VALID — PR-AUC={va_ap:.4f}, ROC-AUC={va_roc:.4f}, "
              f"Best-F1 thr={best_f1_thr:.4f} (P={prec_f1:.3f}, R={rec_f1:.3f}), Q99(neg)={q99_thr:.4f}")

        # pick threshold by quiet FA/day target (primary operating point)
        thr_fa, fa_scan = pick_thr_by_quiet_fa(
            df_va, p_va, Ev_val,
            target_fa_per_day=TARGET_FA_PER_DAY,
            H_PRE=H_PRE, exclude_last_s=EXCLUDE_LAST_S,
            k_of_n=K_OF_N, merge_sec=MERGE_SEC, cooldown_sec=COOLDOWN_SEC
        )
        if thr_fa is None or not np.isfinite(thr_fa):
            thr_fa = q99_thr  # robust fallback
            print(f"[WARN] FA-calibration failed; falling back to Q99 thr={thr_fa:.4f}")
        else:
            picked = fa_scan["picked"]
            print(f"Chosen VALID threshold by FA/day≈{TARGET_FA_PER_DAY}: thr={thr_fa:.4f} "
                  f"(VALID quiet FA/day med={picked['fa_med']:.2f}, det={picked['det']:.3f})")

        # refit on TRAIN+VALID (no test peeking)
        dtrain_full = xgb.DMatrix(pd.concat([X_tr, X_va]), label=np.concatenate([y_tr, y_va]), feature_names=feat_cols)
        final_bst = xgb.train(params, dtrain_full, num_boost_round=best_iter+1)
        chosen_thr = float(thr_fa)
    else:
        fixed_rounds = 600
        print(f"[INFO] Training without early stopping for {fixed_rounds} rounds.")
        dtrain = xgb.DMatrix(X_tr, label=y_tr, feature_names=feat_cols)
        final_bst = xgb.train(params, dtrain, num_boost_round=fixed_rounds)
        p_tr = final_bst.predict(dtrain)
        best_f1_thr, _, _, q99_thr = choose_thresholds(y_tr, p_tr, neg_quantile=0.99)
        chosen_thr = float(q99_thr)
        best_iter = fixed_rounds - 1
        va_ap = va_roc = np.nan
        # No VALID predictions in this branch

    # TEST
    dtest = xgb.DMatrix(X_te, label=y_te, feature_names=feat_cols)
    p_te  = final_bst.predict(dtest)
    te_ap  = average_precision_score(y_te, p_te) if y_te.sum() > 0 else np.nan
    te_roc = roc_auc_score(y_te, p_te) if len(np.unique(y_te)) == 2 else np.nan
    print(f"TEST  — PR-AUC={te_ap:.4f}, ROC-AUC={te_roc:.4f}")
    print(f"Events evaluated — VALID: {len(Ev_val)} | TEST: {len(Ev_tst)}")

    # Event metrics (STRICT pre-trough with exclusion window)
    # VALID metrics
    if 'p_va' in locals():
        ev_val_f1  = event_metrics_pretrough_strict_kofn(df_va, p_va, best_f1_thr, Ev_val, H_PRE, EXCLUDE_LAST_S,
                                                         k_of_n=K_OF_N, merge_sec=MERGE_SEC, cooldown_sec=COOLDOWN_SEC)
        ev_val_q99 = event_metrics_pretrough_strict_kofn(df_va, p_va, q99_thr,    Ev_val, H_PRE, EXCLUDE_LAST_S,
                                                         k_of_n=K_OF_N, merge_sec=MERGE_SEC, cooldown_sec=COOLDOWN_SEC)
        ev_val_fa  = event_metrics_pretrough_strict_kofn(df_va, p_va, chosen_thr, Ev_val, H_PRE, EXCLUDE_LAST_S,
                                                         k_of_n=K_OF_N, merge_sec=MERGE_SEC, cooldown_sec=COOLDOWN_SEC)
        print("\nEvent metrics (VALID) @ Best-F1:",
              f"det_rate={ev_val_f1['event_detection_rate']:.3f}",
              f"| lead_med={ev_val_f1['lead_median_s']:.1f}s",
              f"[p10={ev_val_f1['lead_p10_s']:.1f}s, p90={ev_val_f1['lead_p90_s']:.1f}s]",
              f"| events={ev_val_f1['events_evaluated']}")
        print("Event metrics (VALID) @ Q99(neg):",
              f"det_rate={ev_val_q99['event_detection_rate']:.3f}",
              f"| lead_med={ev_val_q99['lead_median_s']:.1f}s",
              f"[p10={ev_val_q99['lead_p10_s']:.1f}s, p90={ev_val_q99['lead_p90_s']:.1f}s]",
              f"| events={ev_val_q99['events_evaluated']}")
        print("Event metrics (VALID) @ FA-calibrated:",
              f"det_rate={ev_val_fa['event_detection_rate']:.3f}",
              f"| lead_med={ev_val_fa['lead_median_s']:.1f}s",
              f"[p10={ev_val_fa['lead_p10_s']:.1f}s, p90={ev_val_fa['lead_p90_s']:.1f}s]",
              f"| events={ev_val_fa['events_evaluated']}")

    # TEST metrics
    ev_tst_f1  = event_metrics_pretrough_strict_kofn(df_te, p_te, best_f1_thr, Ev_tst, H_PRE, EXCLUDE_LAST_S,
                                                     k_of_n=K_OF_N, merge_sec=MERGE_SEC, cooldown_sec=COOLDOWN_SEC)
    ev_tst_q99 = event_metrics_pretrough_strict_kofn(df_te, p_te, q99_thr,    Ev_tst, H_PRE, EXCLUDE_LAST_S,
                                                     k_of_n=K_OF_N, merge_sec=MERGE_SEC, cooldown_sec=COOLDOWN_SEC)
    ev_tst_fa  = event_metrics_pretrough_strict_kofn(df_te, p_te, chosen_thr, Ev_tst, H_PRE, EXCLUDE_LAST_S,
                                                     k_of_n=K_OF_N, merge_sec=MERGE_SEC, cooldown_sec=COOLDOWN_SEC)
    lead_secs_test_detected = ev_tst_fa.get("lead_values", [])
    plot_leadtime_ecdf(
        lead_secs_test_detected,
        f"{asset} TEST — Lead-time ECDF",
        PLOTS_DIR / f"{asset}_TEST_leadtime_ECDF.png"
    )

    print("\nEvent metrics (TEST)  @ Best-F1:",
          f"det_rate={ev_tst_f1['event_detection_rate']:.3f}",
          f"| lead_med={ev_tst_f1['lead_median_s']:.1f}s",
          f"[p10={ev_tst_f1['lead_p10_s']:.1f}s, p90={ev_tst_f1['lead_p90_s']:.1f}s]",
          f"| events={ev_tst_f1['events_evaluated']}")
    print("Event metrics (TEST)  @ Q99(neg):",
          f"det_rate={ev_tst_q99['event_detection_rate']:.3f}",
          f"| lead_med={ev_tst_q99['lead_median_s']:.1f}s",
          f"[p10={ev_tst_q99['lead_p10_s']:.1f}s, p90={ev_tst_q99['lead_p90_s']:.1f}s]",
          f"| events={ev_tst_q99['events_evaluated']}")
    print("Event metrics (TEST)  @ FA-calibrated:",
          f"det_rate={ev_tst_fa['event_detection_rate']:.3f}",
          f"| lead_med={ev_tst_fa['lead_median_s']:.1f}s",
          f"[p10={ev_tst_fa['lead_p10_s']:.1f}s, p90={ev_tst_fa['lead_p90_s']:.1f}s]",
          f"| events={ev_tst_fa['events_evaluated']}")

    # False alerts on quiet days (using FA-calibrated thr)
    fa_val = false_alerts_stats_quiet(df_va, p_va, chosen_thr, MERGE_SEC, COOLDOWN_SEC) if 'p_va' in locals() else {"zero_day_count":0,"alerts_med":np.nan,"alerts_p90":np.nan}
    fa_tst = false_alerts_stats_quiet(df_te, p_te, chosen_thr, MERGE_SEC, COOLDOWN_SEC)
    print("\nFalse-alerts/day @ FA-calibrated thr:",
          f"VALID zero_days={fa_val['zero_day_count']}, med={fa_val['alerts_med']:.2f}, p90={fa_val['alerts_p90']:.2f}",
          f"| TEST zero_days={fa_tst['zero_day_count']}, med={fa_tst['alerts_med']:.2f}, p90={fa_tst['alerts_p90']:.2f}")

    # Sanity — shuffled
    def shuffled_ap(df_part: pd.DataFrame, probs: np.ndarray):
        if len(df_part) == 0:
            return np.nan
        rng = np.random.default_rng(123)
        y_shuf = df_part["y"].to_numpy().copy()
        rng.shuffle(y_shuf)
        return float(average_precision_score(y_shuf, probs)) if y_shuf.sum() > 0 else np.nan
    shuf_ap = shuffled_ap(df_te, p_te)
    print(f"Sanity — PR-AUC on shuffled TEST labels: {shuf_ap:.4f}")

    # ========== PLOTS & INTERPRETABILITY OUTPUTS ==========
    split_tag = f"{asset}"

    # --- PR & ROC curves ---
    if 'p_va' in locals():
        plot_pr_roc(y_va, p_va, f"{asset} VALID", f"{split_tag}_VALID")
    plot_pr_roc(y_te, p_te, f"{asset} TEST", f"{split_tag}_TEST")

    # --- Threshold scan (VALID) ---
    if 'fa_scan' in locals() and isinstance(fa_scan, dict) and "scan" in fa_scan:
        plot_threshold_scan(
            scan_records=fa_scan["scan"],
            target_fa=TARGET_FA_PER_DAY,
            chosen_thr=float(chosen_thr),
            out_png=PLOTS_DIR / f"{split_tag}_VALID_threshold_scan.png",
            split_name="VALID"
        )

    # --- XGBoost built-in 'gain' importance ---
    gain_map = xgb_gain_importance(final_bst, max_k=30)
    plot_barh_from_mapping(
        gain_map,
        title=f"{asset} — XGB feature importance (gain, top 30)",
        out_png=PLOTS_DIR / f"{split_tag}_feat_importance_gain.png",
        csv_out=PLOTS_DIR / f"{split_tag}_feat_importance_gain.csv"
    )

    # --- Permutation importance (AP drop) on VALID (fallback: TRAIN) ---
    perm_X, perm_y, perm_split = (X_va, y_va, "VALID") if 'X_va' in locals() and X_va is not None and len(X_va)>0 else (X_tr, y_tr, "TRAIN")
    perm_map = permutation_importance_ap(final_bst, perm_X, perm_y, n_repeats=5, random_state=123, max_features=60)
    plot_barh_from_mapping(
        dict(list(perm_map.items())[:30]),
        title=f"{asset} — Permutation importance (AP drop) on {perm_split} (top 30)",
        out_png=PLOTS_DIR / f"{split_tag}_feat_importance_permutation_{perm_split}.png",
        csv_out=PLOTS_DIR / f"{split_tag}_feat_importance_permutation_{perm_split}.csv"
    )

    # --- Optional: SHAP summary (guarded; samples for speed) ---
    if _HAVE_SHAP:
        try:
            # sample a manageable slab from VALID else TEST
            X_for_shap = (X_va if 'X_va' in locals() and X_va is not None and len(X_va)>0 else X_te)
            if X_for_shap is not None and len(X_for_shap)>0:
                n_samp = min(2000, len(X_for_shap))
                samp = X_for_shap.sample(n=n_samp, random_state=123)
                expl = shap.TreeExplainer(final_bst)
                sv = expl.shap_values(samp)
                plt.figure(figsize=(8,5))
                shap.summary_plot(sv, samp, show=False, max_display=25)
                _savefig(PLOTS_DIR / f"{split_tag}_SHAP_summary.png")
        except Exception as e:
            print("[INFO] SHAP skipped:", e)

    # Series aligned to df indices (for year plots/tables)
    if 'p_va' in locals():
        p_va_s = pd.Series(p_va, index=df_va.index)
    p_te_s = pd.Series(p_te, index=df_te.index)

    # ---- Alert-budget curve (VALID) ----
    if 'fa_scan' in locals() and isinstance(fa_scan, dict) and "scan" in fa_scan:
        plot_alert_budget_curve(
            scan_records=fa_scan["scan"],
            chosen_thr=float(chosen_thr),
            best_f1_thr=float(best_f1_thr) if 'best_f1_thr' in locals() else None,
            out_png=PLOTS_DIR / f"{asset}_VALID_alert_budget.png",
            split_name=f"{asset} VALID"
        )

    # ---- FA/day distributions (quiet days) ----
    if 'p_va' in locals():
        plot_fa_box(df_va, p_va, float(chosen_thr),
                    f"{asset} VALID", PLOTS_DIR / f"{asset}_VALID_FA_box.png",
                    MERGE_SEC, COOLDOWN_SEC)
    plot_fa_box(df_te, p_te, float(chosen_thr),
                f"{asset} TEST", PLOTS_DIR / f"{asset}_TEST_FA_box.png",
                MERGE_SEC, COOLDOWN_SEC)

    # ---- Score distributions & reliability ----
    if 'p_va' in locals():
        plot_score_hist(df_va, p_va, f"{asset} VALID", PLOTS_DIR / f"{asset}_VALID_score_hist.png")
        plot_reliability_simple(y_va, p_va, f"{asset} VALID", PLOTS_DIR / f"{asset}_VALID_reliability.png")
    plot_score_hist(df_te, p_te, f"{asset} TEST", PLOTS_DIR / f"{asset}_TEST_score_hist.png")
    plot_reliability_simple(y_te, p_te, f"{asset} TEST", PLOTS_DIR / f"{asset}_TEST_reliability.png")

    # ---- Year-by-year recall with 95% CIs (TEST @ conservative thr) ----
    plot_year_detection_ci(
        df_te, p_te_s, float(chosen_thr), Ev_tst,
        f"{asset} TEST", PLOTS_DIR / f"{asset}_TEST_year_recall.png"
    )

    # ---- Operating-point summary CSV (Conservative vs Best-F1) ----
    if 'p_va' in locals():
        save_operating_points_table(asset, "VALID", df_va, p_va_s, Ev_val,
                                    float(chosen_thr),
                                    float(best_f1_thr) if 'best_f1_thr' in locals() else None,
                                    PLOTS_DIR / f"{asset}_VALID_operating_points.csv")
    save_operating_points_table(asset, "TEST", df_te, p_te_s, Ev_tst,
                                float(chosen_thr),
                                float(best_f1_thr) if 'best_f1_thr' in locals() else None,
                                PLOTS_DIR / f"{asset}_TEST_operating_points.csv")



    # ===== Year-by-year TEST breakdown at FA-calibrated threshold =====
    years_test = sorted(df_te["date"].dt.year.dropna().unique())
    print("\n[TEST breakdown by year @ FA-calibrated]")
    for yy in years_test:
        sub = df_te[df_te["date"].dt.year == yy]
        if len(sub) == 0:
            continue
        p_sub = final_bst.predict(xgb.DMatrix(sub[feat_cols]))
        ev_sub = Ev_tst[Ev_tst["date"].dt.year == yy]
        ap_sub = average_precision_score(sub["y"], p_sub) if sub["y"].sum() > 0 else np.nan
        roc_sub= roc_auc_score(sub["y"], p_sub) if len(np.unique(sub["y"])) == 2 else np.nan
        em_sub = event_metrics_pretrough_strict_kofn(sub, p_sub, chosen_thr, ev_sub, H_PRE, EXCLUDE_LAST_S,
                                                     k_of_n=K_OF_N, merge_sec=MERGE_SEC, cooldown_sec=COOLDOWN_SEC)
        print(f"  {yy}: PR-AUC={ap_sub:.4f}, ROC-AUC={roc_sub:.4f}, "
              f"det={em_sub['event_detection_rate']:.3f}, lead_med={em_sub['lead_median_s']:.1f}s, events={em_sub['events_evaluated']}")

    # Save model and metadata
    mpath = MODEL_DIR / f"xgb_flashcrash_yearsplit_{asset}.json"
    final_bst.save_model(mpath)
    meta = dict(
        asset=asset,
        split_strategy="year",
        year_train=YEAR_TRAIN, year_valid=YEAR_VALID, year_test_from=YEAR_TEST_FROM,
        feature_cols=feat_cols,
        best_ntree_limit=int(get_best_iter(final_bst, 1)+1),
        scale_pos_weight=float(spw),
        valid_pr_auc=float(va_ap) if 'va_ap' in locals() and va_ap==va_ap else None,
        valid_roc_auc=float(va_roc) if 'va_roc' in locals() and va_roc==va_roc else None,
        test_pr_auc=float(te_ap) if te_ap==te_ap else None,
        test_roc_auc=float(te_roc) if te_roc==te_roc else None,
        thresholds={
            "fa_calibrated": float(chosen_thr),
            "best_f1": float(best_f1_thr) if 'best_f1_thr' in locals() else None,
            "q99_neg": float(q99_thr) if 'q99_thr' in locals() else None
        },
        detection_policy={
            "target_fa_per_day": TARGET_FA_PER_DAY,
            "merge_sec": MERGE_SEC,
            "cooldown_sec": COOLDOWN_SEC,
            "k_of_n": K_OF_N,
            "exclude_last_s": EXCLUDE_LAST_S,
            "H_PRE_sec": int(H_PRE.total_seconds())
        },
        event_metrics_valid_best_f1=locals().get("ev_val_f1", {}),
        event_metrics_valid_q99=locals().get("ev_val_q99", {}),
        event_metrics_valid_fa=locals().get("ev_val_fa", {}),
        event_metrics_test_best_f1=ev_tst_f1,
        event_metrics_test_q99=ev_tst_q99,
        event_metrics_test_fa=ev_tst_fa,
        false_alerts_valid=fa_val,
        false_alerts_test=fa_tst,
        sanity_shuffled_ap=shuf_ap,
        train_caps=caps if CAP_FEATURES_Q is not None else {}
    )
    (MODEL_DIR / f"metadata_yearsplit_{asset}.json").write_text(json.dumps(meta, indent=2, default=str))

print("\nDone.")


