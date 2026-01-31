# VIX Options Skew Strategy (Snapshot + Signal + Backtest)

A Python pipeline to (1) collect daily VIX option-chain snapshots from `yfinance`, (2) compute implied-volatility “wing” signals (OTM call wing vs ATM), (3) generate defined-risk VIX call-spread trades, and (4) backtest performance (return, Sharpe, drawdown) using your saved snapshots.

> ⚠️ Important: `yfinance` does **not** provide historical option chain snapshots. This project solves that by saving the chain daily. Backtests use **your locally saved snapshots**.

---

## What this project does

### 1) Data collection (daily snapshots)
- Downloads the current **VIX option chain** for a few near-term expirations.
- Saves the chain to CSV files named like:
  - `asof=YYYY-MM-DD__exp=YYYY-MM-DD__calls.csv`
  - `asof=YYYY-MM-DD__exp=YYYY-MM-DD__puts.csv`

This turns “no historical chains” into a growing dataset you can backtest on.

### 2) Implied-vol surface construction
For each snapshot (as-of date + expiry):

Put–call parity (near ATM):

    C - P = df * (F - K)
    => F ≈ K + (C - P) / df


- Computes **Black-76 implied vol** for calls/puts using that estimated forward.

> Why Black-76? VIX options are typically priced off VIX futures (a forward-like underlying), so Black-76 is a reasonable practical approximation in a lightweight research pipeline.

### 3) Signal: OTM call wing steepness
The strategy uses a single scalar that measures how “expensive the wings” are:
  
    WingSteepness = IV(K_OTM) - IV(K_ATM)


- `K_ATM` = strike closest to `F`
- `K_OTM` = strike closest to `m_high * F` (default `m_high = 1.25`)

We compute a rolling z-score of this steepness using a lookback window:

    z = (S_t - mean) / std

### 4) Trades: defined-risk call spreads
Instead of selling/buying naked options, we use **call spreads** for capped risk:

- If wings are **rich** (z-score high):  
  **SELL call spread** (short convexity, but capped loss)
- If wings are **cheap** (z-score low):  
  **BUY call spread** (long convexity, defined premium)

Strike selection:
- `K1` ≈ ATM
- `K2` ≈ OTM wing (near `m_high * F`)

### 5) Backtest
Backtests run on your snapshot history and print:
- Total return
- CAGR
- Sharpe ratio
- Max drawdown
- Trade logs

---

## Why this can work (intuition)

This project is inspired by a common empirical feature of VIX options:

- Far OTM VIX calls often trade with significant “tail fear” premium.
- The **shape** of the VIX call wing reflects market-implied probability of volatility spikes / regime changes (crash-risk pricing).
- When the wing becomes *unusually steep* relative to recent history, it can indicate “overpaying for tail protection” (sell spreads).
- When the wing becomes *unusually flat*, it can indicate “underpaying for tail protection” (buy spreads).

This is not risk-free. The edge, if any, typically comes from **relative value + mean reversion in the skew**, expressed through **defined-risk structures**.

---

## Requirements

- Python 3.9+
- Packages:
  - `yfinance`
  - `pandas`
  - `numpy`
  - `scipy`

## Get Started

### Clone the Repository
```bash
git clone 
```

### Install Dependencies
```bash
python -m pip install yfinance pandas numpy scipy
```

### Today's signal
```bash
python vix_options_strategy.py --mode live
```

### Collect snapshots (run daily)
```bash
python vix_options_strategy.py --mode collect --snapshot_dir vix_snapshots
```

### Backtest (after you’ve collected multiple days)
```bash
python vix_options_strategy.py --mode backtest --snapshot_dir vix_snapshots --start 2018-01-01
```

### Optional Tuning
```bash
python vix_options_strategy.py --mode backtest --snapshot_dir vix_snapshots --lookback 5 --z_enter 0.75
```
