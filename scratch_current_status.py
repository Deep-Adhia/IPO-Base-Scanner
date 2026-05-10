"""
scratch_current_status.py
Shows current price position relative to base for KWIL, MEESHO, TRUALT.
Answers: "Are they still in the base right now, or have they broken out?"
"""
import pandas as pd
import yfinance as yf
from datetime import datetime, date

SYMBOLS = {
    "KWIL":   "2026-02-16",
    "MEESHO": "2025-12-10",
    "TRUALT": "2025-10-03",
}

CONSOL_WINDOWS = [10, 20, 30]
MAX_PRNG = 25.0

print(f"\nLIVE STATUS CHECK — {datetime.now().strftime('%Y-%m-%d %H:%M IST')}")
print("="*70)

for sym, listing in SYMBOLS.items():
    ticker = f"{sym}.NS"
    df = yf.download(ticker, period="6mo", interval="1d", progress=False, auto_adjust=True)
    if df.empty:
        print(f"[SKIP] {sym} — no data")
        continue
    df = df.reset_index()
    df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
    df = df.rename(columns={"Date":"DATE","Open":"OPEN","High":"HIGH","Low":"LOW","Close":"CLOSE","Volume":"VOLUME"})
    df["DATE"] = pd.to_datetime(df["DATE"]).dt.date
    df = df.sort_values("DATE").reset_index(drop=True)

    lhigh       = float(df["HIGH"].iloc[0])
    curr_close  = float(df["CLOSE"].iloc[-1])
    curr_date   = df["DATE"].iloc[-1]
    pct_from_lh = (curr_close - lhigh) / lhigh * 100
    week_ago    = float(df["CLOSE"].iloc[-6]) if len(df) >= 6 else curr_close
    week_chg    = (curr_close - week_ago) / week_ago * 100

    # Find the current (most recent) base — look back up to 30 days
    # A "base" is defined as: price range in last W days is tight (PRNG < 35%)
    # and current price is within the base (not yet broken out)
    base_status = "WATCHING"
    base_info   = {}

    for w in [10, 20, 30]:
        if len(df) < w + 1:
            continue
        base = df.iloc[-w-1:-1]   # exclude today's candle
        base_low  = float(base["LOW"].min())
        base_high = float(base["HIGH"].max())
        prng = (base_high - base_low) / base_low * 100
        avgv = float(base["VOLUME"].mean())
        vol_today = float(df["VOLUME"].iloc[-1])
        vol_ratio = vol_today / avgv if avgv > 0 else 0

        # Current price relative to base
        if curr_close > base_high:
            pos = "ABOVE BASE (Breakout?)"
            base_status = "BREAKOUT"
        elif curr_close > base_high * 0.97:
            pos = "AT RESISTANCE"
            base_status = "AT PIVOT"
        else:
            pos = "INSIDE BASE"

        base_info[w] = {
            "low": base_low, "high": base_high,
            "prng": prng, "vol_ratio": vol_ratio,
            "pos": pos, "pct_to_breakout": round((base_high - curr_close) / curr_close * 100, 1)
        }
        break  # use smallest window (tightest base)

    print(f"\n  {sym}  |  Listed: {listing}  |  Listing High: Rs{lhigh:.2f}")
    print(f"  Latest Close: Rs{curr_close:.2f} on {curr_date}")
    print(f"  vs Listing High: {pct_from_lh:+.1f}%   |   1-Week Change: {week_chg:+.1f}%")

    if base_info:
        w  = list(base_info.keys())[0]
        bi = base_info[w]
        print(f"  [{w}d Base]  Rs{bi['low']:.2f} — Rs{bi['high']:.2f}  "
              f"(PRNG={bi['prng']:.1f}%)  Vol={bi['vol_ratio']:.1f}x  ->  {bi['pos']}")
        if base_status != "BREAKOUT":
            print(f"  Distance to breakout pivot: {bi['pct_to_breakout']:.1f}% above current price")
        else:
            print(f"  ** Price is ABOVE the {w}d base. Watch for follow-through!")
    print(f"  STATUS: [{base_status}]")

print(f"\n{'='*70}")
print("  Reminder: Scanner will alert on BREAKOUT + PRNG<=25% + VOL>=1.2x")
print("="*70)
