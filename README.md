# Flash Crash Early Warning (BTC & ETH)

A supervised machine learning pipeline that predicts cryptocurrency **flash crashes** a few minutes **before** they happen using **public trade-only data** (Binance, 2021–2024, resampled to 200ms).

## Highlights
- **Goal:** Early warnings for BTC/ETH flash crashes using trade-only features (no order book feed).
- **Models:** XGBoost and Logistic Regression with strict temporal splits  
  - Train: **2021**, Validate: **2022**, Test: **2023–2024**
- **Operating point:** Conservative thresholds tuned to keep **median false alerts/day = 0** on quiet days.

## Results (Test 2023–2024)
- **ETH:** Detects **7/9** with ~**2:00** median lead time; median false alerts/day **0**  
- **BTC:** Detects **3/6** with ~**1:30** median lead time; median false alerts/day **0**

> Year-by-year behaviour suggests **regime sensitivity** (2024 > 2023), so thresholds may need periodic recalibration.

## What we learned
- Flash crashes have **detectable precursors** in trade-only data.
- **Distributed selling pressure (breadth)** is more predictive than single large trades.
- **Signal timing differs by asset:** BTC relies on **5–20s** windows, ETH on **80–160s** windows.
- Detection requires a **coverage vs. noise** trade-off; you can’t have both perfect detection and total silence.

## Flash crash definition (labeling)
A rapid price drop of **3.5–5.0%** over **5–30 minutes**, followed by **≥60% recovery within 60 minutes**.  
Detection is evaluated strictly **pre-trough**, with valid alerts in [t*−120s, t*−30s).

## Features (trades-only)
Four families computed over multiple windows:
- **Activity / Breadth & Volume**
- **Large-trade concentration**
- **Illiquidity (Amihud-style)**
- **Directional impact (Kyle-style slope)**

Windows used include **0–5s**, **5–20s**, **20–80s**, **80–160s**; example feature names: `breadth_5s_20s`, `amihud_rs_80s_160s`, `lambda_ols_20s_80s`.

## Practical use
- Works with **public trade data** (no paid LOB feed).
- Delivers **~1.5–2 minutes** of lead time—enough to escalate, adjust positions, or widen spreads.
- **Low nuisance rate** (median 0/day on quiet periods) makes it viable for monitoring desks.
- **Asset-specific calibration** recommended.

## Limitations
- **Small event counts** (BTC: 6, ETH: 9) ⇒ wide confidence intervals.
- **Temporal instability** (2024 ≫ 2023) ⇒ **adaptive thresholds** likely needed.
- **Trade-only** proxies (no bid–ask/depth) limit microstructure visibility.
- Predictive, not causal.

## Getting started

```bash
# create env
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# run pipeline (example)
python -m flashcrash.train --config configs/default.yaml
python -m flashcrash.evaluate --config configs/default.yaml

```
## Dataset Structure

Each file in `features_stream_dataset/asset=*/date=*/part-*.parquet` contains precomputed feature rows sampled every **200 ms** from Binance trade data.  
These features are derived entirely from public trade information; no order-book data is used.

| Column | Description | Type |
|---------|--------------|------|
| `t` | UTC timestamp of each observation | datetime64[ns, UTC] |
| `asset` | Asset symbol (e.g. BTCUSDT, ETHUSDT) | string |
| `date` | Daily partition date | datetime64[ns, UTC] |
| `y` | Binary label (1 = flash crash start, 0 = normal) | int |
| `breadth_*` | Activity breadth across time windows (5–160 s) | float |
| `volume_all_*` | Total traded volume across each window | float |
| `large_share_*` | Count / notional of large trades | float |
| `amihud_rs_*` | Amihud illiquidity ratio (return-to-volume) | float |
| `lambda_ols_*` | Directional impact (Kyle’s λ proxy) | float |
| `role`, `group_key` | Optional grouping metadata (for internal validation splits) | string |

## Example directory layout

```
features_stream_dataset/
├─ asset=BTCUSDT/
│  ├─ date=2021-01-02/part-000.parquet
│  ├─ date=2024-01-03/part-000.parquet
├─ asset=ETHUSDT/
│  ├─ date=2021-01-03/part-000.parquet
│  ├─ date=2024-01-03/part-000.parquet
```

> For privacy and file-size reasons, this repository includes **only a small demo subset**.  
> For the full, ready-to-use feature dataset (2021–2024), download from Google Drive: [https://drive.google.com/file/d/163oEYtCvWYIAqFXNYYvqS7z0oTKP4Z88/view?usp=drive_link.](https://drive.google.com/drive/folders/13ZIgfITOsCKLRhIe6G8Sk5_6aIZC2ffz?usp=drive_link)

## Layout (partitioned Parquet):

```
features_stream_dataset/
└─ asset=BTCUSDT/date=YYYY-MM-DD/part-*.parquet
└─ asset=ETHUSDT/date=YYYY-MM-DD/part-*.parquet
```
## How the Features Are Computed (method)

Features follow the dissertation’s methodology (trade-only microstructure, computed over rings of (0,5], (5,20], (20,80], (80,160] seconds):

- **Breadth / activity:** breadth_* (trade count per ring)

- **Volume:** volume_all_* (total traded size)

- **Large trade concentration:** large_share_count_*, large_share_notional_*

  - **“Large” is defined causally:** rolling 1-sec notional q=97.5% threshold, past-only as-of join

- **Illiquidity (Amihud):** amihud_rs_* = |returns| / dollar notional

- **Directional impact (Kyle’s λ proxy):** lambda_ols_* = cov(r, flow) / var(flow) in each ring

- **Label y:** 1 inside H_PRE = 120s before each trough time, else 0 (per day/asset partition)

A minimal schema table is already in this README under Dataset Structure.

## Reproduce or Recompute Features Yourself

If you want to generate the feature dataset locally from raw Binance aggTrades CSVs,
use the script below, it’s the exact code used to build the full Google Drive dataset.

##### Script: src/step3_stream_features_parallel.py

## Inputs
### Raw aggTrades CSVs
Downloaded from Binance data dumps or your own collector.
Expected folder layout:
```
data/
└─ BINANCE/
   └─ aggTrades/
      ├─ BTCUSDT/BTCUSDT-aggTrades-2021-01-02.csv
      ├─ BTCUSDT/BTCUSDT-aggTrades-2021-01-03.csv
      ├─ ETHUSDT/ETHUSDT-aggTrades-2021-01-02.csv
      └─ ETHUSDT/ETHUSDT-aggTrades-2021-01-03.csv
```

In the script, point the variable DATA_ROOT to your own copy of the folder:

#### Example for any operating system
DATA_ROOT    = Path("data/BINANCE")
EVENTS_CSV   = Path("./events_refined_with_trades.csv")   
MANIFEST     = Path("./agg_download_manifest.csv")  

Events file: events_refined_with_trades.csv
- Required columns: asset, date, t_trough_final (UTC)
Manifest: agg_download_manifest.csv
- Lists which (asset, date, role) to process. Example:
```
asset,date,role
BTCUSDT,2021-01-02,pos
BTCUSDT,2021-01-03,quiet
ETHUSDT,2021-01-02,near_miss
```

## What the script does

Loads daily Binance trade data with fast parquet caching (_cache_parquet/)

Applies Lee–Ready direction rule + tie-break with isBuyerMaker

Computes causal large-trade thresholds (rolling 1 s notional, past-only)

Bucketizes trades into 200 ms intervals

Builds ring-window features over (0, 5], (5, 20], (20, 80], (80, 160] seconds:

  - **breadth_*** – trade count per window
    
  - **volume_all_*** – total volume
    
  - **large_share_*** – large-trade concentration
    
  - **amihud_rs_*** – Amihud illiquidity ratio
    
  - **lambda_ols_*** – directional impact (Kyle’s λ proxy)

Labels positives in a 120 s pre-trough window (H_PRE = 120s)

Down-samples negatives per role (NEG_PER_DAY_*) to balance classes

Writes partitioned parquet files under
```
features_stream_dataset/asset=.../date=.../part-*.parquet
```

---

