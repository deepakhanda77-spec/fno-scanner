"""
F&O Pullback-to-Support Bounce Scanner
--------------------------------------
Finds NSE F&O stocks that are:
  1. In a MEDIUM-TERM UPTREND      (above rising 50 DMA, 50 DMA > 200 DMA, close > 200 DMA)
  2. In a SHORT-TERM CORRECTION    (4-18% off recent high, RSI dipped, closed below 20 EMA)
  3. At a SUPPORT                  (swing low / 50-100-200 DMA / 61.8% fib) - touched or briefly undercut
  4. BOUNCING / RECLAIMING support (bullish close above prior day's high, RSI turning up)

Install :  pip install yfinance pandas numpy
Run     :  python fno_pullback_scanner.py
Output  :  scan_results.csv + table on screen

Optional: put your own F&O symbols (one per line, no .NS) in fo_list.txt
          to override the built-in list. NSE changes the F&O list periodically.

NOT financial advice. Use for idea generation, confirm on your charts.
"""

import os
import numpy as np
import pandas as pd
import yfinance as yf

# ----------------------------- CONFIG ---------------------------------
PERIOD            = "2y"
MIN_PULLBACK_PCT  = 4.0     # min fall from 40-day high
MAX_PULLBACK_PCT  = 18.0    # max fall (deeper = trend likely damaged)
SUPPORT_TOL_UP    = 0.015   # recent low may be up to 1.5% ABOVE support (touch)
SUPPORT_TOL_DOWN  = 0.030   # ...or up to 3% BELOW support (false breakdown / spring)
LOOKBACK_SWING    = 120     # bars used to find swing-low supports
RECENT_BARS       = 5       # bars in which support must have been tested
MIN_RR            = 1.5     # minimum reward:risk to report
# -----------------------------------------------------------------------

DEFAULT_FNO = """
RELIANCE HDFCBANK ICICIBANK INFY TCS SBIN AXISBANK KOTAKBANK LT ITC BHARTIARTL
HINDUNILVR BAJFINANCE BAJAJFINSV MARUTI M&M TATAMOTORS TATASTEEL JSWSTEEL HINDALCO
SUNPHARMA DRREDDY CIPLA DIVISLAB APOLLOHOSP TITAN ASIANPAINT ULTRACEMCO GRASIM
ADANIENT ADANIPORTS NTPC POWERGRID ONGC COALINDIA BPCL IOC GAIL VEDL
WIPRO HCLTECH TECHM LTIM EICHERMOT HEROMOTOCO BAJAJ-AUTO TVSMOTOR ASHOKLEY
NESTLEIND BRITANNIA DABUR GODREJCP COLPAL TATACONSUM PIDILITIND BERGEPAINT
INDUSINDBK BANKBARODA PNB CANBK FEDERALBNK IDFCFIRSTB AUBANK
SBILIFE HDFCLIFE ICICIGI ICICIPRULI CHOLAFIN MUTHOOTFIN SHRIRAMFIN
DLF GODREJPROP OBEROIRLTY LODHA
TATAPOWER JINDALSTEL SAIL NMDC HAL BEL BHEL SIEMENS ABB CUMMINS
DIXON POLYCAB HAVELLS VOLTAS CROMPTON
ZOMATO TRENT IRCTC INDIGO PERSISTENT COFORGE MPHASIS
LUPIN AUROPHARMA TORNTPHARM ALKEM BIOCON
PFC RECLTD LICHSGFIN SBICARD
""".split()


def load_universe():
    if os.path.exists("fo_list.txt"):
        with open("fo_list.txt") as f:
            syms = [s.strip().upper() for s in f if s.strip()]
    else:
        syms = DEFAULT_FNO
    return sorted(set(syms))


# ----------------------------- INDICATORS -----------------------------
def rsi(close, n=14):
    d = close.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / dn.replace(0, np.nan))


def atr(df, n=14):
    pc = df["Close"].shift()
    tr = pd.concat([df["High"] - df["Low"],
                    (df["High"] - pc).abs(),
                    (df["Low"] - pc).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False).mean()


def swing_lows(low, window=5):
    """Pivot lows: lowest point within +/- window bars."""
    roll = low.rolling(2 * window + 1, center=True).min()
    return low[(low == roll)].dropna()


# ----------------------------- CORE LOGIC -----------------------------
def analyze(sym, df):
    df = df.dropna()
    if len(df) < 220:
        return None

    c, h, l, o, v = (df[x] for x in ["Close", "High", "Low", "Open", "Volume"])
    sma50, sma100, sma200 = (c.rolling(n).mean() for n in (50, 100, 200))
    ema5, ema20 = c.ewm(span=5).mean(), c.ewm(span=20).mean()
    r = rsi(c)
    a = atr(df)

    # ---- 1. Medium-term uptrend ----
    trend = (
        c.iloc[-1] > sma200.iloc[-1]
        and sma50.iloc[-1] > sma200.iloc[-1]
        and sma50.iloc[-1] > sma50.iloc[-21]          # 50 DMA rising over last month
        and c.iloc[-1] > c.iloc[-126] * 1.05          # up >5% in ~6 months
    )
    if not trend:
        return None

    # ---- 2. Short-term correction ----
    swing_high = h.iloc[-40:].max()
    recent_low = l.iloc[-RECENT_BARS:].min()
    pullback = (swing_high - recent_low) / swing_high * 100
    corrected = (
        MIN_PULLBACK_PCT <= pullback <= MAX_PULLBACK_PCT
        and r.iloc[-10:].min() < 45
        and (c.iloc[-10:] < ema20.iloc[-10:]).any()
    )
    if not corrected:
        return None

    # ---- 3. Support test ----
    pivots = swing_lows(l.iloc[-LOOKBACK_SWING:-RECENT_BARS])
    swing_lo_120 = l.iloc[-LOOKBACK_SWING:].min()
    fib618 = swing_high - 0.618 * (swing_high - swing_lo_120)

    candidates = {f"Swing low {p:.1f}": p for p in pivots.values}
    candidates["50 DMA"] = sma50.iloc[-1]
    candidates["100 DMA"] = sma100.iloc[-1]
    candidates["200 DMA"] = sma200.iloc[-1]
    candidates["61.8% Fib"] = fib618

    hits = []
    for name, s in candidates.items():
        if np.isnan(s):
            continue
        if s * (1 - SUPPORT_TOL_DOWN) <= recent_low <= s * (1 + SUPPORT_TOL_UP) \
                and c.iloc[-1] > s:
            hits.append((name, s))
    if not hits:
        return None

    # pick the support closest to the recent low
    sup_name, sup = min(hits, key=lambda x: abs(x[1] - recent_low))
    undercut = recent_low < sup
    setup = "False breakdown + reclaim" if undercut else "Support bounce"

    # ---- 4. Bounce / reclaim trigger ----
    bullish_candle = c.iloc[-1] > o.iloc[-1]
    trigger = (
        c.iloc[-1] > h.iloc[-2]                       # close above yesterday's high
        and bullish_candle
        and c.iloc[-1] > ema5.iloc[-1]
        and r.iloc[-1] > r.iloc[-2]
    )
    if not trigger:
        return None

    # ---- Trade levels ----
    entry = c.iloc[-1]
    sl = recent_low - 0.5 * a.iloc[-1]
    target = swing_high
    risk = entry - sl
    rr = (target - entry) / risk if risk > 0 else np.nan
    if np.isnan(rr) or rr < MIN_RR:
        return None

    vol_ratio = v.iloc[-1] / v.iloc[-21:-1].mean()

    # ---- Score (0-100) ----
    score = 0
    score += 25 if c.iloc[-1] > sma50.iloc[-1] else 15            # trend strength
    score += 20 if undercut else 15                               # spring = stronger
    score += min(20, vol_ratio * 10)                              # volume confirmation
    score += min(20, rr * 6)                                      # reward:risk
    score += 15 if 30 <= r.iloc[-1] <= 55 else 5                  # RSI room to run

    return {
        "Symbol": sym,
        "Setup": setup,
        "Support": sup_name,
        "Support Lvl": round(sup, 1),
        "Recent Low": round(recent_low, 1),
        "CMP": round(entry, 1),
        "Pullback %": round(pullback, 1),
        "RSI": round(r.iloc[-1], 1),
        "Vol x Avg": round(vol_ratio, 2),
        "SL": round(sl, 1),
        "Target": round(target, 1),
        "R:R": round(rr, 2),
        "Score": round(score, 1),
    }


def main():
    syms = load_universe()
    tickers = [s + ".NS" for s in syms]
    print(f"Downloading {len(tickers)} F&O stocks...")
    data = yf.download(tickers, period=PERIOD, group_by="ticker",
                       auto_adjust=True, progress=False, threads=True)

    rows = []
    for s, t in zip(syms, tickers):
        try:
            res = analyze(s, data[t])
            if res:
                rows.append(res)
        except Exception as e:
            print(f"  skip {s}: {e}")

    if not rows:
        print("\nNo stock matches all conditions today. Try loosening "
              "MIN_PULLBACK_PCT / SUPPORT_TOL_* in CONFIG.")
        return

    out = pd.DataFrame(rows).sort_values("Score", ascending=False)
    out.to_csv("scan_results.csv", index=False)
    pd.set_option("display.width", 220, "display.max_columns", None)
    print(f"\n{len(out)} candidate(s):\n")
    print(out.to_string(index=False))
    print("\nSaved to scan_results.csv")


if __name__ == "__main__":
    main()
