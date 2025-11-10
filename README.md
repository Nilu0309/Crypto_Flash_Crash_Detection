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
