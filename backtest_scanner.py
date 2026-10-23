"""
Walk-forward backtest for fno_pullback_scanner.py
-------------------------------------------------
Put this file in the SAME folder as fno_pullback_scanner.py and run:

    python backtest_scanner.py

How it works (no look-ahead):
  * For every stock and every day, the scanner sees ONLY data up to that day's close.
  * If it signals, the trade is entered at the NEXT day's OPEN.
  * Exit = stop loss, target (prior swing high), or time stop - whichever comes first.
  * If SL and target are both hit in the same candle, SL is assumed first (conservative).
  * One open trade per stock at a time.
  * Round-trip cost/slippage is deducted from every trade.

Outputs: backtest_trades.csv + summary on screen.
"""

import numpy as np
import pandas as pd
import yfinance as yf

from fno_pullback_scanner import analyze, load_universe

# ----------------------------- CONFIG ---------------------------------
HISTORY      = "5y"     # data downloaded
MAX_HOLD     = 20       # time stop (trading days)
COST_PCT     = 0.10     # round-trip cost + slippage, in % of entry
LOOKBACK     = 300      # bars handed to the scanner each day
WARMUP       = 220      # first bar index eligible for signals
BASELINE_FWD = 10       # days, for the "random uptrend day" baseline
# -----------------------------------------------------------------------


def simulate(df, i, sl, target):
    """Enter next-day open. Returns (exit_idx, exit_price, reason) or None."""
    if i + 1 >= len(df):
        return None
    entry = df["Open"].iloc[i + 1]
    if entry <= sl or entry >= target:        # gapped through levels -> skip
        return None

    last = min(i + MAX_HOLD, len(df) - 1)
    for j in range(i + 1, last + 1):
        o, h, l = df["Open"].iloc[j], df["High"].iloc[j], df["Low"].iloc[j]
        if o <= sl:
            return j, o, "SL (gap)", entry
        if l <= sl:
            return j, sl, "SL", entry
        if o >= target:
            return j, o, "Target (gap)", entry
        if h >= target:
            return j, target, "Target", entry
    return last, df["Close"].iloc[last], "Time exit", entry


def profit_factor(r):
    gains, losses = r[r > 0].sum(), -r[r < 0].sum()
    return gains / losses if losses > 0 else np.inf


def summarize(t, label="ALL TRADES"):
    if t.empty:
        print(f"{label}: no trades")
        return
    r = t["R"]
    win = (r > 0).mean() * 100
    print(f"\n=== {label} ===")
    print(f"Trades            : {len(t)}")
    print(f"Win rate          : {win:.1f}%")
    print(f"Avg R / trade     : {r.mean():.2f}   (expectancy)")
    print(f"Avg win / loss (R): {r[r > 0].mean():.2f} / {r[r < 0].mean():.2f}")
    print(f"Profit factor     : {profit_factor(r):.2f}")
    print(f"Avg net return    : {t['Net %'].mean():.2f}%   median {t['Net %'].median():.2f}%")
    print(f"Avg holding days  : {t['Days'].mean():.1f}")
    eq = t.sort_values("Exit Date")["R"].cumsum()
    print(f"Total R           : {eq.iloc[-1]:.1f}   max drawdown (R): {(eq - eq.cummax()).min():.1f}")


def main():
    syms = load_universe()
    tickers = [s + ".NS" for s in syms]
    print(f"Downloading {len(tickers)} stocks ({HISTORY})...")
    data = yf.download(tickers, period=HISTORY, group_by="ticker",
                       auto_adjust=True, progress=False, threads=True)

    trades, baseline = [], []

    for n, (s, tk) in enumerate(zip(syms, tickers), 1):
        try:
            df = data[tk].dropna()
        except Exception:
            continue
        if len(df) < WARMUP + MAX_HOLD + 10:
            continue

        c = df["Close"]
        sma50, sma200 = c.rolling(50).mean(), c.rolling(200).mean()
        uptrend = (c > sma200) & (sma50 > sma200)

        # baseline: forward return from ANY uptrend day (what luck alone gives)
        fwd = c.shift(-BASELINE_FWD) / c - 1
        baseline.append(fwd[uptrend].dropna())

        busy_until = -1
        for i in range(WARMUP, len(df) - 1):
            if i <= busy_until or not uptrend.iloc[i]:
                continue
            res = analyze(s, df.iloc[max(0, i - LOOKBACK): i + 1])
            if not res:
                continue

            sim = simulate(df, i, res["SL"], res["Target"])
            if sim is None:
                continue
            j, exit_px, reason, entry = sim
            risk = entry - res["SL"]
            cost = entry * COST_PCT / 100
            r_mult = (exit_px - entry - cost) / risk

            trades.append({
                "Symbol": s,
                "Signal Date": df.index[i].date(),
                "Entry Date": df.index[i + 1].date(),
                "Exit Date": df.index[j].date(),
                "Setup": res["Setup"],
                "Support": res["Support"].split(" ")[0] if "Swing" not in res["Support"] else "Swing low",
                "Score": res["Score"],
                "Entry": round(entry, 2),
                "SL": res["SL"],
                "Target": res["Target"],
                "Exit": round(exit_px, 2),
                "Reason": reason,
                "Days": j - i,
                "Net %": round((exit_px - entry - cost) / entry * 100, 2),
                "R": round(r_mult, 2),
            })
            busy_until = j

        if n % 20 == 0:
            print(f"  processed {n}/{len(syms)}  trades so far: {len(trades)}")

    t = pd.DataFrame(trades)
    if t.empty:
        print("No trades generated. Loosen the scanner CONFIG and retry.")
        return
    t.to_csv("backtest_trades.csv", index=False)

    summarize(t)

    # Baseline comparison
    base = pd.concat(baseline)
    print(f"\nBaseline: avg {BASELINE_FWD}-day return from ANY uptrend day = "
          f"{base.mean() * 100:.2f}% (win rate {(base > 0).mean() * 100:.1f}%)")

    print("\n--- Exit reasons ---")
    print(t["Reason"].value_counts().to_string())

    print("\n--- By setup type ---")
    for k, g in t.groupby("Setup"):
        print(f"{k:28s} n={len(g):4d}  win={100*(g['R']>0).mean():5.1f}%  "
              f"avgR={g['R'].mean():5.2f}  PF={profit_factor(g['R']):.2f}")

    print("\n--- By score bucket ---")
    t["Bucket"] = pd.cut(t["Score"], [0, 60, 70, 80, 101], labels=["<60", "60-70", "70-80", "80+"])
    for k, g in t.groupby("Bucket", observed=True):
        print(f"{str(k):8s} n={len(g):4d}  win={100*(g['R']>0).mean():5.1f}%  avgR={g['R'].mean():5.2f}")

    print("\n--- By year (consistency check) ---")
    t["Year"] = pd.to_datetime(t["Exit Date"]).dt.year
    for k, g in t.groupby("Year"):
        print(f"{k}  n={len(g):4d}  win={100*(g['R']>0).mean():5.1f}%  "
              f"avgR={g['R'].mean():5.2f}  totalR={g['R'].sum():6.1f}")

    print("\nSaved: backtest_trades.csv")
    print("Caveats: current F&O list => survivorship bias; Yahoo data can have gaps; "
          "no position sizing / margin modeled.")


if __name__ == "__main__":
    main()
