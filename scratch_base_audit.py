"""
scratch_base_audit.py
Forensic IPO Base Auditor — checks KWIL, TRUALT, MEESHO
against current scanner parameters and shows what it sees.
"""

import sys
import os
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta

OUTPUT_DIR = "base_audit_charts"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- Mirror scanner constants (from streamlined_ipo_scanner.py) ---
CONSOL_WINDOWS = [10, 20, 30, 60, 90, 120]
MAX_PRNG       = 25.0
VOL_MULT       = 1.2
ABS_VOL_MIN    = 3_000_000

SYMBOLS = {
    "KWIL":   {"listing_date": "2026-02-16"},
    "MEESHO": {"listing_date": "2025-12-10"},
    "TRUALT": {"listing_date": "2025-10-03"},
}

def fetch(symbol):
    ticker = f"{symbol}.NS"
    try:
        df = yf.download(ticker, period="2y", interval="1d", progress=False, auto_adjust=True)
        if df.empty:
            print(f"[WARN] No data for {ticker}")
            return None
        df = df.reset_index()
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
        df = df.rename(columns={"Date":"DATE","Open":"OPEN","High":"HIGH","Low":"LOW","Close":"CLOSE","Volume":"VOLUME"})
        df["DATE"] = pd.to_datetime(df["DATE"]).dt.date
        df = df.sort_values("DATE").reset_index(drop=True)
        print(f"[OK] {symbol}: {len(df)} candles  ({df['DATE'].iloc[0]} to {df['DATE'].iloc[-1]})")
        return df
    except Exception as e:
        print(f"[ERROR] {symbol}: {e}")
        return None

def scan_bases(df, w):
    """Scan for bases using scanner logic. Returns list of candidate (j, base_high, base_low, prng, vol_ratio)."""
    candidates = []
    if len(df) < w:
        return candidates
    lhigh = df["HIGH"].iloc[0]   # listing day high

    for j in range(w, len(df)):
        base = df.iloc[j-w:j]
        low   = base["LOW"].min()
        high2 = base["HIGH"].max()

        # Context rule: base must be 8-35% below listing high
        perf = (low - lhigh) / lhigh
        if not (0.08 <= -perf <= 0.50):   # wider than scanner for research
            continue

        prng = (high2 - low) / low * 100
        avgv = base["VOLUME"].mean()
        if avgv <= 0:
            continue

        vol_ratio = df["VOLUME"].iat[j] / avgv
        breakout_close = df["CLOSE"].iat[j]

        # Breakout condition: close above base high
        is_breakout = breakout_close > high2

        candidates.append({
            "j": j,
            "date": df["DATE"].iat[j],
            "base_high": round(high2, 2),
            "base_low":  round(low, 2),
            "prng":      round(prng, 2),
            "vol_ratio": round(vol_ratio, 2),
            "is_breakout": is_breakout,
            "window": w,
            "lhigh": round(lhigh, 2),
            "pct_below_lhigh": round(-perf * 100, 1),
            "vol_ok": vol_ratio >= VOL_MULT or (df["VOLUME"].iat[j] * breakout_close) >= ABS_VOL_MIN,
            "prng_ok": prng <= MAX_PRNG,
        })
    return candidates

def plot_symbol(symbol, listing_date_str, df):
    listing_date = pd.to_datetime(listing_date_str).date()
    df_plot = df.copy()

    lhigh = df_plot["HIGH"].iloc[0]

    # Find best candidate across all windows
    all_candidates = []
    for w in CONSOL_WINDOWS:
        cands = scan_bases(df, w)
        # keep only breakout candles
        breakouts = [c for c in cands if c["is_breakout"]]
        all_candidates.extend(breakouts)

    # Sort by tightness (PRNG)
    all_candidates.sort(key=lambda x: x["prng"])

    print(f"\n{'='*60}")
    print(f"  {symbol}  |  Listed: {listing_date_str}  |  Listing High: {lhigh:.2f}")
    print(f"{'='*60}")
    if not all_candidates:
        print("  [!] NO breakout bases detected by scanner logic")
    else:
        for c in all_candidates[:5]:
            caught = c["prng_ok"] and c["vol_ok"]
            status = "[CATCH]" if caught else "[MISS ]"
            miss_reason = ""
            if not c["prng_ok"]: miss_reason += f" PRNG={c['prng']}>{MAX_PRNG}"
            if not c["vol_ok"]:  miss_reason += f" VOL={c['vol_ratio']:.1f}x<{VOL_MULT}x"
            print(f"  {status} Date={c['date']}  W={c['window']:3d}d  "
                  f"Base={c['base_low']:.2f}-{c['base_high']:.2f}  "
                  f"PRNG={c['prng']:.1f}%  VOL={c['vol_ratio']:.1f}x  "
                  f"PctBelowHigh={c['pct_below_lhigh']:.0f}%{miss_reason}")

    # ---- Build forensic chart ----
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.72, 0.28], vertical_spacing=0.04)

    dates = df_plot["DATE"].astype(str)

    fig.add_trace(go.Candlestick(
        x=dates, open=df_plot["OPEN"], high=df_plot["HIGH"],
        low=df_plot["LOW"], close=df_plot["CLOSE"],
        increasing_line_color="#26a69a", decreasing_line_color="#ef5350",
        name="Price", showlegend=False), row=1, col=1)

    vol_colors = ["#26a69a" if r["CLOSE"] >= r["OPEN"] else "#ef5350"
                  for _, r in df_plot.iterrows()]
    fig.add_trace(go.Bar(x=dates, y=df_plot["VOLUME"],
                         marker_color=vol_colors, opacity=0.6,
                         name="Volume", showlegend=False), row=2, col=1)

    # 20d avg vol
    df_plot["AVG20"] = df_plot["VOLUME"].rolling(20, min_periods=1).mean()
    fig.add_trace(go.Scatter(x=dates, y=df_plot["AVG20"],
                             line=dict(color="gold", width=1.5, dash="dot"),
                             name="20D Avg Vol"), row=2, col=1)

    # Listing high line
    fig.add_hline(y=lhigh, line_dash="dash", line_color="mediumpurple",
                  line_width=1.5, row=1, col=1)
    fig.add_annotation(x=dates.iloc[-1], y=lhigh, text=f"Listing High {lhigh:.2f}",
                       font=dict(color="mediumpurple", size=10),
                       showarrow=False, xanchor="right", yanchor="bottom")

    # Overlay top-5 bases
    colors = ["#FFD700","#00BFFF","#FF6B6B","#90EE90","#FF69B4"]
    for i, c in enumerate(all_candidates[:5]):
        col = colors[i % len(colors)]
        j = c["j"]
        bw = c["window"]
        base_start = dates.iat[max(0, j-bw)]
        base_end   = dates.iat[j-1]
        caught = c["prng_ok"] and c["vol_ok"]
        label = "CAUGHT" if caught else "MISSED"
        border = "#00FF00" if caught else "#FF4444"

        fig.add_shape(type="rect",
                      x0=base_start, y0=c["base_low"],
                      x1=base_end,   y1=c["base_high"],
                      line=dict(color=border, width=1.5, dash="dot"),
                      fillcolor=f"rgba(255,255,255,0.05)", row=1, col=1)

        fig.add_annotation(x=base_start, y=c["base_high"],
                           text=f"W={bw}d PRNG={c['prng']}% {label}",
                           font=dict(color=border, size=9),
                           showarrow=False, xanchor="left", yanchor="bottom",
                           bgcolor="rgba(0,0,0,0.7)", borderpad=3)

        # Breakout candle marker
        fig.add_annotation(x=dates.iat[j], y=df_plot["HIGH"].iat[j],
                           text="BO", font=dict(color=col, size=10),
                           showarrow=True, arrowhead=2, arrowcolor=col, ay=-30,
                           bgcolor="rgba(0,0,0,0.7)")

    fig.update_layout(
        title=f"IPO BASE FORENSIC AUDIT — {symbol} (Listed {listing_date_str})",
        paper_bgcolor="#0d1117", plot_bgcolor="#0d1117",
        font=dict(color="#cdd9e5"),
        xaxis_rangeslider_visible=False,
        height=700,
        legend=dict(bgcolor="rgba(0,0,0,0.5)")
    )
    fig.update_xaxes(gridcolor="#21262d", showgrid=True)
    fig.update_yaxes(gridcolor="#21262d", showgrid=True)

    path = os.path.join(OUTPUT_DIR, f"{symbol}_base_audit.html")
    fig.write_html(path)
    print(f"  [Chart] Saved -> {path}")
    return all_candidates

# ---- Main ----
if __name__ == "__main__":
    summary = {}
    for sym, meta in SYMBOLS.items():
        df = fetch(sym)
        if df is None or df.empty:
            print(f"[SKIP] {sym} — no data")
            continue
        cands = plot_symbol(sym, meta["listing_date"], df)
        caught  = [c for c in cands if c["prng_ok"] and c["vol_ok"]]
        missed  = [c for c in cands if not (c["prng_ok"] and c["vol_ok"])]
        summary[sym] = {"total_breakouts": len(cands), "caught": len(caught), "missed": len(missed)}

    print("\n" + "="*60)
    print("  SUMMARY")
    print("="*60)
    for sym, s in summary.items():
        print(f"  {sym:10s}  Breakouts={s['total_breakouts']:3d}  "
              f"Caught={s['caught']:3d}  Missed={s['missed']:3d}")
    print("\n  Open base_audit_charts/*.html to inspect visually.")
