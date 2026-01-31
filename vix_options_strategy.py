import os
import json
import math
import glob
import datetime as dt
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, List

import numpy as np
import pandas as pd
import yfinance as yf
from scipy.optimize import brentq

# -------------------------
# Utilities / Metrics
# -------------------------

def annualized_sharpe(daily_returns: pd.Series, rf_daily: float = 0.0) -> float:
    r = daily_returns.dropna()
    if len(r) < 2:
        return np.nan
    excess = r - rf_daily
    vol = excess.std(ddof=1)
    if vol == 0:
        return np.nan
    return np.sqrt(252.0) * excess.mean() / vol

def max_drawdown(equity: pd.Series) -> float:
    if equity.empty:
        return np.nan
    peak = equity.cummax()
    dd = equity / peak - 1.0
    return dd.min()

def cagr(equity: pd.Series) -> float:
    if equity.empty or len(equity) < 2:
        return np.nan
    start = equity.index[0]
    end = equity.index[-1]
    years = (end - start).days / 365.25
    if years <= 0:
        return np.nan
    return (equity.iloc[-1] / equity.iloc[0]) ** (1.0 / years) - 1.0

# -------------------------
# Black-76 on VIX options (approx)
# Note: VIX options are typically priced on VIX futures;
# we approximate the forward using put-call parity from the chain.
# -------------------------

def norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

def black76_price(F: float, K: float, T: float, sigma: float, is_call: bool, r: float = 0.0) -> float:
    if T <= 0:
        intrinsic = max(F - K, 0.0) if is_call else max(K - F, 0.0)
        return math.exp(-r * T) * intrinsic
    if sigma <= 0:
        intrinsic = max(F - K, 0.0) if is_call else max(K - F, 0.0)
        return math.exp(-r * T) * intrinsic
    df = math.exp(-r * T)
    d1 = (math.log(F / K) + 0.5 * sigma * sigma * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    if is_call:
        return df * (F * norm_cdf(d1) - K * norm_cdf(d2))
    else:
        return df * (K * norm_cdf(-d2) - F * norm_cdf(-d1))

def implied_vol_black76(price: float, F: float, K: float, T: float, is_call: bool, r: float = 0.0) -> float:
    # robust bracket
    if price <= 0 or F <= 0 or K <= 0 or T <= 0:
        return np.nan

    # Lower bound (almost 0 vol) and upper bound (500% vol)
    def f(sig):
        return black76_price(F, K, T, sig, is_call, r) - price

    try:
        # ensure sign change
        lo, hi = 1e-6, 5.0
        flo, fhi = f(lo), f(hi)
        if flo * fhi > 0:
            return np.nan
        return brentq(f, lo, hi, maxiter=200)
    except Exception:
        return np.nan

def write_trade_ticket(out_dir: str, ticket: dict) -> str:
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    path = Path(out_dir) / f"trade_ticket_{ts}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(ticket, f, indent=2)
    return str(path)

def pick_front_expiry_snapshot_for_day(snaps_for_day: List[ChainSnapshot], asof: dt.date) -> Optional[ChainSnapshot]:
    future = [s for s in snaps_for_day if s.expiry > asof]
    if not future:
        return None
    return sorted(future, key=lambda s: s.expiry)[0]

def compute_signal_history_from_snapshots(snapshot_dir: str, cfg: StrategyConfig) -> pd.Series:
    snaps = load_snapshots(snapshot_dir)
    if not snaps:
        raise RuntimeError(f"No snapshots found in {snapshot_dir}. Collect snapshots first.")

    by_day: Dict[dt.date, List[ChainSnapshot]] = {}
    for s in snaps:
        by_day.setdefault(s.asof, []).append(s)

    all_days = sorted(by_day.keys())
    sig_series = pd.Series(dtype=float)

    for d in all_days:
        front = pick_front_expiry_snapshot_for_day(by_day[d], d)
        if front is None:
            continue
        try:
            surf = compute_iv_surface(front, r=cfg.r)
            calls, meta = surf["calls"], surf["meta"]
            F = float(meta["F"].iloc[0])
            sig = wing_steepness_signal(calls, F, m_high=cfg.m_high)
            sig_series.loc[pd.Timestamp(d)] = sig
        except Exception:
            # skip bad days
            continue

    return sig_series.sort_index()

# -------------------------
# Data acquisition
# -------------------------

@dataclass
class ChainSnapshot:
    asof: dt.date
    expiry: dt.date
    calls: pd.DataFrame
    puts: pd.DataFrame

def fetch_vix_history(start="2015-01-01") -> pd.DataFrame:
    vix = yf.download("^VIX", start=start, progress=False)
    if vix.empty:
        raise RuntimeError("Failed to download ^VIX history from yfinance.")
    vix = vix.rename(columns=str.lower)
    vix.index = pd.to_datetime(vix.index)
    return vix

def fetch_vix_option_chain(expiration: str) -> ChainSnapshot:
    """
    Pull current option chain for ^VIX at a given expiration.
    NOTE: On some setups, yfinance may not return options for ^VIX.
    """
    t = yf.Ticker("^VIX")
    ch = t.option_chain(expiration)
    asof = dt.date.today()
    expiry = dt.datetime.strptime(expiration, "%Y-%m-%d").date()
    calls = ch.calls.copy()
    puts = ch.puts.copy()
    return ChainSnapshot(asof=asof, expiry=expiry, calls=calls, puts=puts)

def list_vix_expirations() -> List[str]:
    t = yf.Ticker("^VIX")
    exps = getattr(t, "options", None)
    if not exps:
        # some installs might use t.options property; if missing/empty, fail clearly
        raise RuntimeError("No expirations available for ^VIX via yfinance on this system.")
    return list(exps)

# -------------------------
# Snapshot persistence (so you can backtest later)
# -------------------------

def save_snapshot(snapshot: ChainSnapshot, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    base = f"asof={snapshot.asof.isoformat()}__exp={snapshot.expiry.isoformat()}"
    calls_path = os.path.join(out_dir, base + "__calls.csv")
    puts_path = os.path.join(out_dir, base + "__puts.csv")
    snapshot.calls.to_csv(calls_path, index=False)
    snapshot.puts.to_csv(puts_path, index=False)

def load_snapshots(out_dir: str) -> List[ChainSnapshot]:
    calls_files = sorted(glob.glob(os.path.join(out_dir, "*__calls.csv")))
    snaps = []
    for cf in calls_files:
        pf = cf.replace("__calls.csv", "__puts.csv")
        if not os.path.exists(pf):
            continue
        # parse asof and exp from filename
        name = os.path.basename(cf)
        parts = name.split("__")
        asof = parts[0].split("asof=")[1]
        exp = parts[1].split("exp=")[1]
        asof_d = dt.datetime.strptime(asof, "%Y-%m-%d").date()
        exp_d = dt.datetime.strptime(exp, "%Y-%m-%d").date()
        calls = pd.read_csv(cf)
        puts = pd.read_csv(pf)
        snaps.append(ChainSnapshot(asof=asof_d, expiry=exp_d, calls=calls, puts=puts))
    return snaps

# -------------------------
# Calibration proxy inspired by the paper:
# "regime-switching probability mainly shows up in far OTM VIX calls"
#
# Practical implementation:
# - infer forward F from put-call parity near ATM
# - compute implied vols
# - compute wing steepness: IV(OTM call at moneyness m_high) - IV(ATM)
# - compare to rolling distribution => rich/cheap wings
# - trade defined-risk call spreads
# -------------------------

@dataclass
class Trade:
    entry_date: dt.date
    expiry: dt.date
    direction: str  # "BUY_CALL_SPREAD" or "SELL_CALL_SPREAD"
    K1: float       # lower strike
    K2: float       # higher strike
    entry_premium: float  # positive means paid; negative means received
    size: int       # number of spreads

def mid_price(df: pd.DataFrame) -> pd.Series:
    # yfinance has bid/ask; if missing, fallback to lastPrice
    bid = df.get("bid", pd.Series(np.nan, index=df.index))
    ask = df.get("ask", pd.Series(np.nan, index=df.index))
    last = df.get("lastPrice", pd.Series(np.nan, index=df.index))
    mid = (bid + ask) / 2.0
    mid = mid.fillna(last)
    return mid

def estimate_forward_from_parity(calls: pd.DataFrame, puts: pd.DataFrame, T: float, r: float = 0.0) -> Tuple[float, float]:
    """
    Use put-call parity around ATM:
      C - P = df*(F - K)  => F = K + (C - P)/df
    We'll compute across strikes where both call/put are liquid and take a median.
    """
    df = math.exp(-r * T)
    c = calls[["strike"]].copy()
    p = puts[["strike"]].copy()
    c["cmid"] = mid_price(calls).values
    p["pmid"] = mid_price(puts).values

    merged = pd.merge(c, p, on="strike", how="inner")
    merged = merged.dropna(subset=["cmid", "pmid"])
    merged = merged[(merged["cmid"] > 0) & (merged["pmid"] > 0)]

    if merged.empty:
        raise RuntimeError("Could not estimate forward: insufficient call/put overlap.")

    merged["F_est"] = merged["strike"] + (merged["cmid"] - merged["pmid"]) / df

    # focus near the "center" by using strikes with smallest |C-P|
    merged["abs_cp"] = (merged["cmid"] - merged["pmid"]).abs()
    merged = merged.sort_values("abs_cp").head(12)

    F = float(np.median(merged["F_est"]))
    quality = float(np.median(merged["abs_cp"]))  # smaller is better
    return F, quality

def compute_iv_surface(snapshot: ChainSnapshot, r: float = 0.0) -> Dict[str, pd.DataFrame]:
    asof = snapshot.asof
    expiry = snapshot.expiry
    T = max((expiry - asof).days / 365.25, 1e-6)

    calls = snapshot.calls.copy()
    puts = snapshot.puts.copy()
    calls["mid"] = mid_price(calls)
    puts["mid"] = mid_price(puts)

    F, quality = estimate_forward_from_parity(calls, puts, T, r=r)

    # IVs (calls + puts)
    calls["iv"] = calls.apply(lambda row: implied_vol_black76(row["mid"], F, row["strike"], T, True, r=r), axis=1)
    puts["iv"] = puts.apply(lambda row: implied_vol_black76(row["mid"], F, row["strike"], T, False, r=r), axis=1)

    calls = calls.dropna(subset=["iv"])
    puts = puts.dropna(subset=["iv"])
    calls["moneyness"] = calls["strike"] / F
    puts["moneyness"] = puts["strike"] / F

    meta = pd.DataFrame({"asof":[asof], "expiry":[expiry], "T":[T], "F":[F], "parity_quality":[quality]})
    return {"calls": calls, "puts": puts, "meta": meta}

def pick_strikes_for_spread(calls: pd.DataFrame, F: float, m_atm: float = 1.0, m_high: float = 1.25) -> Optional[Tuple[float, float, float, float]]:
    """
    Pick K1 ~ ATM (closest to F) and K2 ~ high strike around m_high*F.
    Return (K1, K2, C1_mid, C2_mid).
    """
    if calls.empty:
        return None
    tmp = calls.copy()
    tmp = tmp.dropna(subset=["strike", "mid"])
    tmp = tmp[(tmp["mid"] > 0)]
    if tmp.empty:
        return None

    tmp["dist_atm"] = (tmp["strike"] - (m_atm * F)).abs()
    K1_row = tmp.sort_values("dist_atm").iloc[0]

    tmp["dist_high"] = (tmp["strike"] - (m_high * F)).abs()
    K2_row = tmp.sort_values("dist_high").iloc[0]

    K1, C1 = float(K1_row["strike"]), float(K1_row["mid"])
    K2, C2 = float(K2_row["strike"]), float(K2_row["mid"])

    if K2 <= K1:
        # try next best high
        tmp2 = tmp[tmp["strike"] > K1].sort_values("dist_high")
        if tmp2.empty:
            return None
        K2_row = tmp2.iloc[0]
        K2, C2 = float(K2_row["strike"]), float(K2_row["mid"])

    return K1, K2, C1, C2

def wing_steepness_signal(calls: pd.DataFrame, F: float, m_high: float = 1.25) -> float:
    """
    Wing steepness proxy:
      IV(K_high) - IV(ATM)
    (a simple scalar capturing "OTM call wing richness")
    """
    tmp = calls.dropna(subset=["strike", "iv"]).copy()
    if tmp.empty:
        return np.nan
    tmp["dist_atm"] = (tmp["strike"] - F).abs()
    atm = tmp.sort_values("dist_atm").iloc[0]["iv"]

    tmp["dist_high"] = (tmp["strike"] - m_high * F).abs()
    wing = tmp.sort_values("dist_high").iloc[0]["iv"]

    return float(wing - atm)

def ticket_from_latest_snapshot(snapshot_dir: str, cfg: StrategyConfig, out_dir: str = "trade_tickets"):
    sig_series = compute_signal_history_from_snapshots(snapshot_dir, cfg)
    if sig_series.dropna().empty:
        raise RuntimeError("Signal history is empty. Check your snapshots / chain parsing.")

    # Need enough history for z-score
    recent = sig_series.dropna().tail(cfg.lookback)
    if len(recent) < max(10, cfg.lookback // 2):
        raise RuntimeError(
            f"Not enough snapshot days for z-score. Have {len(recent)}, need >{cfg.lookback}. "
            f"Collect more days or reduce --lookback."
        )

    today_ts = sig_series.dropna().index[-1]
    today = today_ts.date()
    S_t = float(sig_series.loc[today_ts])
    mu = float(recent.mean())
    sd = float(recent.std(ddof=1))
    z = (S_t - mu) / sd if sd > 0 else np.nan

    # Load latest day's front expiry snapshot to get strikes/premiums
    snaps = load_snapshots(snapshot_dir)
    by_day: Dict[dt.date, List[ChainSnapshot]] = {}
    for s in snaps:
        by_day.setdefault(s.asof, []).append(s)

    front = pick_front_expiry_snapshot_for_day(by_day[today], today)
    if front is None:
        raise RuntimeError("Could not find a front-expiry snapshot for the latest asof day.")

    surf = compute_iv_surface(front, r=cfg.r)
    calls, meta = surf["calls"], surf["meta"]
    F = float(meta["F"].iloc[0])

    picks = pick_strikes_for_spread(calls, F, m_atm=1.0, m_high=cfg.m_high)
    if not picks:
        raise RuntimeError("Could not pick strikes for spread from latest snapshot.")
    K1, K2, C1, C2 = picks
    spread_mid = float(C1 - C2)

    # Decide direction based on z-score threshold
    if z >= cfg.z_enter:
        direction = "SELL_CALL_SPREAD"
        # For selling a spread, your "limit" is typically a CREDIT you want to receive.
        # We'll represent limit as the spread mid credit.
        limit_price = spread_mid
    elif z <= -cfg.z_enter:
        direction = "BUY_CALL_SPREAD"
        limit_price = spread_mid
    else:
        direction = "NO_TRADE"
        limit_price = spread_mid

    ticket = {
        "asof": str(today),
        "expiry": str(front.expiry),
        "underlying": "^VIX",
        "strategy": "CALL_SPREAD",
        "signal": {
            "wing_steepness": S_t,
            "lookback": cfg.lookback,
            "z_enter": cfg.z_enter,
            "z_score": float(z),
            "m_high": cfg.m_high,
            "forward_estimate": float(F),
        },
        "quote": {
            "K1_call_mid": float(C1),
            "K2_call_mid": float(C2),
            "spread_mid": float(spread_mid),
            "note": "Mid prices from yfinance bid/ask midpoint (fallback to lastPrice)."
        },
        "recommendation": {
            "action": direction,
            "qty_spreads": cfg.max_spreads,
            "limit_price": float(limit_price),
            "legs": [
                {"right": "C", "strike": float(K1), "action": "BUY"},
                {"right": "C", "strike": float(K2), "action": "SELL"},
            ],
            "execution_note": "For SELL_CALL_SPREAD you typically enter as a credit; for BUY_CALL_SPREAD you enter as a debit."
        },
        "disclaimer": "Research tool only. Verify contract specs, settlement, and liquidity before trading."
    }

    if direction == "NO_TRADE":
        print("\n--- TICKET MODE ---")
        print(f"Latest asof: {today} | z={z:.2f} within [-{cfg.z_enter}, +{cfg.z_enter}] -> NO TRADE")
        print(f"Candidate spread: BUY C {K1:.1f} / SELL C {K2:.1f} @ mid {spread_mid:.4f}")
        return

    path = write_trade_ticket(out_dir, ticket)

    print("\n--- TRADE TICKET GENERATED ---")
    print(f"As of: {today} | Expiry: {front.expiry} | F~{F:.2f}")
    print(f"WingSteepness={S_t:.4f} | z={z:.2f} | threshold={cfg.z_enter}")
    print(f"Recommendation: {direction} {cfg.max_spreads}x CALL SPREAD")
    print(f"Legs: BUY C {K1:.1f} / SELL C {K2:.1f}")
    print(f"Spread mid: {spread_mid:.4f} (use as debit/credit limit reference)")
    print(f"Saved ticket: {path}")


# -------------------------
# Strategy rules
# -------------------------

@dataclass
class StrategyConfig:
    m_high: float = 1.25
    z_enter: float = 1.0
    lookback: int = 20
    max_spreads: int = 1
    r: float = 0.0
    hold_days: int = 5          # NEW: exit after N asof days
    notional_per_trade: float = 1000.0  # NEW: scale P&L in dollars

def decide_trade(asof: dt.date, expiry: dt.date, calls: pd.DataFrame, meta: pd.DataFrame,
                 hist_signals: pd.Series, cfg: StrategyConfig) -> Optional[Trade]:
    F = float(meta["F"].iloc[0])
    sig = wing_steepness_signal(calls, F, m_high=cfg.m_high)
    if np.isnan(sig):
        return None

    # rolling z-score
    recent = hist_signals.dropna().tail(cfg.lookback)
    if len(recent) < max(10, cfg.lookback // 2):
        return None

    mu = recent.mean()
    sd = recent.std(ddof=1)
    if sd == 0:
        return None
    z = (sig - mu) / sd

    picks = pick_strikes_for_spread(calls, F, m_atm=1.0, m_high=cfg.m_high)
    if not picks:
        return None
    K1, K2, C1, C2 = picks

    # Call spread premium (buy K1 sell K2)
    buy_spread_cost = C1 - C2  # >0 means you pay

    # If wings are "rich": sell call spread (collect premium)
    # If wings are "cheap": buy call spread
    if z >= cfg.z_enter:
        # Sell call spread: receive premium (negative entry_premium means cash in)
        return Trade(entry_date=asof, expiry=expiry, direction="SELL_CALL_SPREAD",
                     K1=K1, K2=K2, entry_premium=-(buy_spread_cost), size=cfg.max_spreads)
    elif z <= -cfg.z_enter:
        return Trade(entry_date=asof, expiry=expiry, direction="BUY_CALL_SPREAD",
                     K1=K1, K2=K2, entry_premium=(buy_spread_cost), size=cfg.max_spreads)
    return None

def spread_payoff_at_expiry(vix_settle: float, K1: float, K2: float, direction: str) -> float:
    """
    Call spread payoff (buy K1 sell K2):
      payoff = min(max(VIX-K1,0), K2-K1)
    For SELL_CALL_SPREAD, payoff is negative of that (you are short the spread).
    """
    long_payoff = min(max(vix_settle - K1, 0.0), K2 - K1)
    if direction == "BUY_CALL_SPREAD":
        return long_payoff
    elif direction == "SELL_CALL_SPREAD":
        return -long_payoff
    else:
        raise ValueError("Unknown direction")

# -------------------------
# Backtest requires saved chain snapshots
# -------------------------

def spread_mid_from_chain(snapshot: ChainSnapshot, K1: float, K2: float) -> Optional[float]:
    calls = snapshot.calls.copy()
    calls["mid"] = mid_price(calls)
    c1 = calls.loc[np.isclose(calls["strike"], K1), "mid"]
    c2 = calls.loc[np.isclose(calls["strike"], K2), "mid"]
    if len(c1) == 0 or len(c2) == 0:
        return None
    return float(c1.iloc[0] - c2.iloc[0])

def backtest_from_snapshots(snapshot_dir: str, vix_hist: pd.DataFrame, cfg: StrategyConfig):
    snaps = load_snapshots(snapshot_dir)
    if not snaps:
        raise RuntimeError(f"No snapshots found in {snapshot_dir}.")

    snaps = sorted(snaps, key=lambda s: (s.asof, s.expiry))
    by_day: Dict[dt.date, List[ChainSnapshot]] = {}
    for s in snaps:
        by_day.setdefault(s.asof, []).append(s)

    all_days = sorted(by_day.keys())
    hist_signal_series = pd.Series(dtype=float)

    open_trade: Optional[Trade] = None
    open_trade_entry_idx: Optional[int] = None

    equity = 1.0
    equity_curve = []
    trade_log = []

    for i, d in enumerate(all_days):
        equity_curve.append((pd.Timestamp(d), equity))

        # choose front expiry available that day
        day_snaps = [s for s in by_day[d] if s.expiry > d]
        if not day_snaps:
            continue
        front = sorted(day_snaps, key=lambda s: s.expiry)[0]

        surf = compute_iv_surface(front, r=cfg.r)
        calls, meta = surf["calls"], surf["meta"]
        F = float(meta["F"].iloc[0])

        sig = wing_steepness_signal(calls, F, m_high=cfg.m_high)
        hist_signal_series.loc[pd.Timestamp(d)] = sig

        # EXIT logic: after hold_days
        if open_trade is not None and open_trade_entry_idx is not None:
            if i - open_trade_entry_idx >= cfg.hold_days:
                # try to exit using today's chain (same expiry)
                # find snapshot for the same expiry on this asof date
                same_exp = [s for s in by_day[d] if s.expiry == open_trade.expiry]
                if same_exp:
                    exit_snap = same_exp[0]
                    exit_mid = spread_mid_from_chain(exit_snap, open_trade.K1, open_trade.K2)
                    if exit_mid is not None:
                        # P&L for long spread: (exit - entry); for short spread: -(exit - entry)
                        entry = open_trade.entry_premium
                        pnl_per_spread = (exit_mid - entry)
                        if open_trade.direction == "SELL_CALL_SPREAD":
                            pnl_per_spread = -pnl_per_spread

                        pnl_dollars = pnl_per_spread * 100.0 * open_trade.size  # option multiplier
                        # scale to equity with notional
                        equity *= (1.0 + pnl_dollars / cfg.notional_per_trade)

                        trade_log.append({
                            "date": d, "event": "EXIT", "expiry": open_trade.expiry,
                            "direction": open_trade.direction, "K1": open_trade.K1, "K2": open_trade.K2,
                            "entry_mid": entry, "exit_mid": exit_mid,
                            "pnl_per_spread": pnl_per_spread, "pnl_$": pnl_dollars,
                            "signal": sig
                        })

                        open_trade = None
                        open_trade_entry_idx = None

        # ENTRY logic
        if open_trade is None:
            # need enough history for z-score
            trade = decide_trade(d, front.expiry, calls, meta, hist_signal_series, cfg)
            if trade is not None:
                # overwrite entry premium to be actual spread mid today
                picks = (trade.K1, trade.K2)
                entry_mid = spread_mid_from_chain(front, picks[0], picks[1])
                if entry_mid is None:
                    continue

                # For buy spread, you pay entry_mid; for sell spread, you receive entry_mid.
                trade.entry_premium = entry_mid
                open_trade = trade
                open_trade_entry_idx = i

                trade_log.append({
                    "date": d, "event": "ENTRY", "expiry": trade.expiry,
                    "direction": trade.direction, "K1": trade.K1, "K2": trade.K2,
                    "entry_mid": entry_mid, "signal": sig
                })

    eq = pd.Series([v for _, v in equity_curve], index=[t for t, _ in equity_curve], name="equity").sort_index()
    daily_ret = eq.pct_change().fillna(0.0)

    summary = pd.DataFrame([{
        "CAGR": cagr(eq),
        "Sharpe": annualized_sharpe(daily_ret),
        "MaxDrawdown": max_drawdown(eq),
        "TotalReturn": eq.iloc[-1] / eq.iloc[0] - 1.0,
        "Days": len(eq),
        "Trades": sum(1 for x in trade_log if x["event"] == "EXIT")
    }])

    trades_df = pd.DataFrame(trade_log)
    return trades_df, eq, daily_ret, summary

# -------------------------
# Live / forward mode: compute today's signal and suggested trade
# -------------------------

def live_signal_and_trade(cfg: StrategyConfig):
    exps = list_vix_expirations()
    # pick nearest expiry at least 7 days out
    today = dt.date.today()
    future_exps = []
    for e in exps:
        ed = dt.datetime.strptime(e, "%Y-%m-%d").date()
        if (ed - today).days >= 7:
            future_exps.append(e)
    if not future_exps:
        raise RuntimeError("No suitable expirations >= 7 days found.")
    exp = future_exps[0]

    snap = fetch_vix_option_chain(exp)
    surf = compute_iv_surface(snap, r=cfg.r)
    calls, meta = surf["calls"], surf["meta"]
    F = float(meta["F"].iloc[0])
    sig = wing_steepness_signal(calls, F, m_high=cfg.m_high)

    picks = pick_strikes_for_spread(calls, F, m_atm=1.0, m_high=cfg.m_high)
    if not picks:
        print("Could not pick strikes for spread.")
        return
    K1, K2, C1, C2 = picks
    buy_cost = C1 - C2

    ticket = {
    "asof": str(snap.asof),
    "expiry": str(snap.expiry),
    "underlying": "^VIX",
    "strategy": "CALL_SPREAD",
    "legs": [
        {"right": "C", "action": "BUY",  "strike": float(K1)},
        {"right": "C", "action": "SELL", "strike": float(K2)},
    ],
    "qty_spreads": 1,
    "limit_price": float(buy_cost),  # user can adjust
    "model": {
        "forward_estimate": float(F),
        "wing_steepness": float(sig),
        "m_high": float(cfg.m_high),
        "note": "This is a candidate ticket. For BUY/SELL decision use backtest z-score signal."
        }
    }
    write_trade_ticket("trade_tickets", ticket)

def collect_snapshots(out_dir: str, max_exps: int = 3):
    exps = list_vix_expirations()
    today = dt.date.today()
    # choose first few expirations that are in the future
    future_exps = []
    for e in exps:
        ed = dt.datetime.strptime(e, "%Y-%m-%d").date()
        if ed > today:
            future_exps.append(e)
    future_exps = future_exps[:max_exps]

    for e in future_exps:
        snap = fetch_vix_option_chain(e)
        save_snapshot(snap, out_dir)

    print(f"Saved {len(future_exps)} expirations for asof={today} into {out_dir}")

# -------------------------
# Main
# -------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--hold_days", type=int, default=5)
    parser.add_argument("--mode", choices=["live", "collect", "backtest", "ticket"], required=True)
    parser.add_argument("--snapshot_dir", default="vix_snapshots")
    parser.add_argument("--start", default="2015-01-01")
    parser.add_argument("--m_high", type=float, default=1.25)
    parser.add_argument("--z_enter", type=float, default=1.0)
    parser.add_argument("--lookback", type=int, default=20)
    args = parser.parse_args()

    cfg = StrategyConfig(m_high=args.m_high, z_enter=args.z_enter, lookback=args.lookback, hold_days=args.hold_days)


    if args.mode == "live":
        live_signal_and_trade(cfg)

    elif args.mode == "collect":
        collect_snapshots(args.snapshot_dir, max_exps=3)

    elif args.mode == "backtest":
        vix_hist = fetch_vix_history(start=args.start)
        trades_df, equity, daily_ret, summary = backtest_from_snapshots(args.snapshot_dir, vix_hist, cfg)

        print("\n=== BACKTEST SUMMARY ===")
        print(summary.to_string(index=False))

        print("\n=== LAST 20 EQUITY POINTS ===")
        print(equity.tail(20).to_string())

        print("\n=== TRADES (last 30 rows) ===")
        if trades_df.empty:
            print("(no trades)")
        else:
            print(trades_df.tail(30).to_string(index=False))
    
    elif args.mode == "ticket":
        ticket_from_latest_snapshot(args.snapshot_dir, cfg)

