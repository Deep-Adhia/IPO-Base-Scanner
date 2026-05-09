import os
import sys
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yfinance as yf
from datetime import datetime, timedelta
import requests
from dotenv import load_dotenv

load_dotenv()

# MongoDB Setup
from db import signals_col, positions_col, logs_col

UPSTOX_ACCESS_TOKEN = os.getenv("UPSTOX_ACCESS_TOKEN")
OUTPUT_DIR = "validation_charts"

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

def get_upstox_mapping():
    """Load symbol to instrument_key mapping"""
    try:
        from db import get_instrument_key_mapping
        mapping = get_instrument_key_mapping()
        if mapping:
            return mapping
    except Exception as e:
        print(f"Could not load Upstox mapping from DB: {e}")

    # Fallback to CSV
    if os.path.exists("ipo_upstox_mapping.csv"):
        df = pd.read_csv("ipo_upstox_mapping.csv")
        return dict(zip(df['symbol'], df['instrument_key']))
    return {}

UPSTOX_MAPPING = get_upstox_mapping()

def fetch_data_upstox(symbol, start_date, end_date):
    """Fetch accurate historical data from Upstox"""
    if not UPSTOX_ACCESS_TOKEN:
        return None

    instrument_key = UPSTOX_MAPPING.get(symbol)
    if not instrument_key:
        return None

    for d in [start_date, end_date]:
        if isinstance(d, datetime): d = d.strftime("%Y-%m-%d")
        elif isinstance(d, pd.Timestamp): d = d.strftime("%Y-%m-%d")
        elif hasattr(d, 'date'): d = d.strftime("%Y-%m-%d")

    if isinstance(start_date, datetime): start_date = start_date.strftime("%Y-%m-%d")
    elif isinstance(start_date, pd.Timestamp): start_date = start_date.strftime("%Y-%m-%d")
    elif hasattr(start_date, 'date'): start_date = start_date.strftime("%Y-%m-%d")

    if isinstance(end_date, datetime): end_date = end_date.strftime("%Y-%m-%d")
    elif isinstance(end_date, pd.Timestamp): end_date = end_date.strftime("%Y-%m-%d")
    elif hasattr(end_date, 'date'): end_date = end_date.strftime("%Y-%m-%d")

    url = f"https://api.upstox.com/v2/historical-candle/{instrument_key}/day/{end_date}/{start_date}"
    headers = {
        'Accept': 'application/json',
        'Authorization': f'Bearer {UPSTOX_ACCESS_TOKEN}'
    }

    try:
        res = requests.get(url, headers=headers, timeout=10)
        if res.status_code == 200:
            data = res.json().get('data', {}).get('candles', [])
            if not data:
                return None
            records = []
            for row in data:
                records.append({
                    "DATE": pd.to_datetime(row[0]).date(),
                    "OPEN": float(row[1]),
                    "HIGH": float(row[2]),
                    "LOW": float(row[3]),
                    "CLOSE": float(row[4]),
                    "VOLUME": int(row[5])
                })
            df = pd.DataFrame(records)
            df = df.sort_values("DATE").reset_index(drop=True)
            return df
    except Exception as e:
        print(f"Error fetching Upstox for {symbol}: {e}")

    return None

def fetch_data_yfinance(symbol, start_date, end_date):
    """Fallback fetch from YFinance"""
    try:
        yf_sym = f"{symbol}.NS"
        end_dt = pd.to_datetime(end_date) + pd.Timedelta(days=1)
        df = yf.download(yf_sym, start=start_date, end=end_dt, progress=False)
        if df.empty:
            return None
        df = df.reset_index()
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
        col_map = {"Date": "DATE", "Open": "OPEN", "High": "HIGH",
                   "Low": "LOW", "Close": "CLOSE", "Volume": "VOLUME"}
        df = df.rename(columns=col_map)
        df["DATE"] = pd.to_datetime(df["DATE"]).dt.date
        return df[["DATE", "OPEN", "HIGH", "LOW", "CLOSE", "VOLUME"]]
    except Exception as e:
        print(f"Error fetching YFinance for {symbol}: {e}")
        return None

def fetch_historical_data(symbol, start_date, end_date):
    """Try Upstox, fallback to YFinance"""
    df = fetch_data_upstox(symbol, start_date, end_date)
    if df is not None and not df.empty:
        return df, "Upstox"
    df = fetch_data_yfinance(symbol, start_date, end_date)
    if df is not None and not df.empty:
        return df, "YFinance"
    return None, "Failed"

def get_listing_date(symbol):
    """Fetch listing date from db or csv"""
    try:
        from db import db
        meta = db["metadata"].find_one({"_id": "listing_dates"})
        if meta and symbol in meta.get("dates", {}):
            return pd.to_datetime(meta["dates"][symbol]).date()
    except:
        pass
    if os.path.exists("ipo_listing_data.csv"):
        df = pd.read_csv("ipo_listing_data.csv")
        row = df[df["symbol"] == symbol]
        if not row.empty:
            return pd.to_datetime(row.iloc[0]["listing_date"]).date()
    return None


def plot_signal(signal_doc, is_rejected=False):
    """
    Forensic Evidence Panel for IPO Base Scanner signals.

    Each chart is a self-contained audit document showing:
    - Color-coded frame (GREEN = ACCEPTED, RED = REJECTED)
    - Base zone shaded rectangle with explicit High/Low labels
    - Breakout level line (the actual O(N) trigger)
    - Entry, Stop, Target overlays with price labels
    - Signal candle triangle marker
    - O(N^2) flaw watermark if trigger is below base high
    - Forensic metrics panel: wick%, vol z-score, base tightness,
      close/high%, breakout extension%, R:R ratio
    - 20-day average volume line on volume subplot
    """

    symbol = signal_doc.get("symbol")
    status_text = "REJECTED" if is_rejected else "ACCEPTED"

    # ── Timestamps ────────────────────────────────────────────────────────────
    ts_field = "timestamp" if is_rejected else "created_at"
    signal_timestamp = pd.to_datetime(signal_doc.get(ts_field))
    signal_date = signal_timestamp.date()

    # ── Extract geometric properties ──────────────────────────────────────────
    if is_rejected:
        metrics_raw = signal_doc.get("details", {}).get("metrics", {})
        if "metrics" in signal_doc and isinstance(signal_doc["metrics"], dict):
            metrics_raw = signal_doc["metrics"]
        consolidation_window = int(metrics_raw.get("w", 40))
        entry_price  = metrics_raw.get("breakout_close") or metrics_raw.get("entry_price")
        stop_loss    = None
        target_price = None
        exit_price   = None
        rejection_reason = (signal_doc.get("rejection_reason")
                            or signal_doc.get("details", {}).get("rejection_reason", "N/A"))
        grade   = metrics_raw.get("grade", "N/A")
        pnl_pct = None
    else:
        sid = signal_doc.get("signal_id", "")
        parts = sid.split("_")
        consolidation_window = int(parts[-1]) if parts and parts[-1].isdigit() else 40
        entry_price  = signal_doc.get("breakout_close") or signal_doc.get("entry_price")
        stop_loss    = signal_doc.get("stop_loss")
        target_price = signal_doc.get("target_price")
        rejection_reason = None
        grade   = signal_doc.get("grade", "N/A")
        pnl_pct = signal_doc.get("pnl_pct")
        exit_price = signal_doc.get("exit_price")
    
    # --- Extract cohorts ---
    cohorts = signal_doc.get("valid_cohorts", [])
    cohort_text = ", ".join([c[0] for c in cohorts]) if cohorts else "N/A"

    # ── Phase 4: Research Metadata Extraction ─────────────────────────────────
    pattern_type  = signal_doc.get("pattern_type", "UNKNOWN")
    market_regime = signal_doc.get("market_regime", "UNKNOWN")
    data_quality  = signal_doc.get("data_quality", "UNKNOWN")
    ghost_entry   = signal_doc.get("potential_entry") # For rejections

    # ── Data fetch ────────────────────────────────────────────────────────────
    listing_date = get_listing_date(symbol)
    if not listing_date:
        listing_date = signal_date - pd.Timedelta(days=365)

    df, source = fetch_historical_data(
        symbol, listing_date, datetime.today().date()
    )
    if df is None or df.empty:
        print(f"  [SKIP] No data for {symbol}")
        return

    df_prior = df[df["DATE"] <= signal_date].copy()
    if df_prior.empty:
        print(f"  [SKIP] No prior data for {symbol} on {signal_date}")
        return

    lhigh = df_prior["HIGH"].iloc[0]

    # ── Reconstruct base window ───────────────────────────────────────────────
    base_end_idx   = len(df_prior) - 2
    base_start_idx = max(0, base_end_idx - consolidation_window + 1)

    if base_end_idx >= 0:
        base_window     = df_prior.iloc[base_start_idx : base_end_idx + 1]
        base_high       = base_window["HIGH"].max()
        base_low        = base_window["LOW"].min()
        base_start_date = base_window["DATE"].iloc[0]
        base_end_date   = base_window["DATE"].iloc[-1]
        avg_vol         = base_window["VOLUME"].mean()
        std_vol         = base_window["VOLUME"].std() if len(base_window) > 1 else 1.0
    else:
        base_high = lhigh * 0.8; base_low = lhigh * 0.5
        base_start_date = df_prior["DATE"].iloc[0]; base_end_date = signal_date
        avg_vol = df_prior["VOLUME"].mean(); std_vol = df_prior["VOLUME"].std() or 1.0

    base_tightness = round((base_high - base_low) / base_low * 100, 2) if base_low > 0 else 0

    # ── Signal candle fingerprint ─────────────────────────────────────────────
    sc = df[df["DATE"] == signal_date]
    sig_open = sig_high = sig_low = sig_close = sig_vol = None
    wick_pct = close_vs_high = vol_zscore = breakout_ext_pct = None
    is_flawed = False

    if not sc.empty:
        sig_open  = sc["OPEN"].iloc[0]
        sig_high  = sc["HIGH"].iloc[0]
        sig_low   = sc["LOW"].iloc[0]
        sig_close = sc["CLOSE"].iloc[0]
        sig_vol   = sc["VOLUME"].iloc[0]
        crange    = sig_high - sig_low
        wick_pct        = round((sig_high - sig_close) / crange * 100, 1) if crange > 0 else 0
        close_vs_high   = round(sig_close / sig_high * 100, 1)
        vol_zscore      = round((sig_vol - avg_vol) / std_vol, 2) if std_vol > 0 else 0
        breakout_ext_pct = round((sig_close - base_high) / base_high * 100, 2) if base_high > 0 else 0
        is_flawed       = (sig_close <= base_high)

    # ── Color scheme ──────────────────────────────────────────────────────────
    if is_rejected:
        frame_color = "#FF4444"
        frame_fill  = "rgba(255,68,68,0.07)"
        verdict_label = "REJECTED"
    else:
        frame_color = "#00CC66"
        frame_fill  = "rgba(0,204,102,0.07)"
        verdict_label = "ACCEPTED"

    # ── Title ─────────────────────────────────────────────────────────────────
    if is_rejected:
        title_str = f"{verdict_label}  |  {symbol}  |  Reason: {rejection_reason}  |  Grade: {grade}"
    else:
        title_str = f"{verdict_label}  |  {symbol}  |  Grade: {grade}"
        if pnl_pct is not None:
            title_str += f"  |  PnL: {pnl_pct:+.2f}%"

    # ── Build figure ──────────────────────────────────────────────────────────
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        vertical_spacing=0.04,
        row_heights=[0.72, 0.28],
    )

    # Candlestick
    fig.add_trace(go.Candlestick(
        x=df["DATE"], open=df["OPEN"], high=df["HIGH"],
        low=df["LOW"], close=df["CLOSE"],
        increasing_line_color="#26a69a", decreasing_line_color="#ef5350",
        name="Price", showlegend=False
    ), row=1, col=1)

    # Volume bars
    vol_colors = [
        "#26a69a" if row["CLOSE"] >= row["OPEN"] else "#ef5350"
        for _, row in df.iterrows()
    ]
    fig.add_trace(go.Bar(
        x=df["DATE"], y=df["VOLUME"],
        marker_color=vol_colors, opacity=0.65,
        name="Volume", showlegend=False
    ), row=2, col=1)

    # 20-day avg volume line
    df_avol = df.copy()
    df_avol["AVG_VOL"] = df_avol["VOLUME"].rolling(20, min_periods=1).mean()
    fig.add_trace(go.Scatter(
        x=df_avol["DATE"], y=df_avol["AVG_VOL"],
        line=dict(color="rgba(255,215,0,0.75)", width=1.5, dash="dot"),
        name="20D Avg Vol", showlegend=True
    ), row=2, col=1)

    # ── Listing High (context only, not trigger) ──────────────────────────────
    fig.add_hline(y=lhigh, line_dash="dash", line_color="mediumpurple",
                  line_width=1, row=1, col=1)
    fig.add_annotation(
        x=str(df["DATE"].iloc[-1]), y=lhigh,
        text=f"Listing High  Rs{lhigh:.2f}",
        font=dict(color="mediumpurple", size=11),
        showarrow=False, xanchor="right", yanchor="bottom"
    )

    # ── BASE ZONE shaded rectangle ────────────────────────────────────────────
    fig.add_shape(
        type="rect",
        x0=str(base_start_date), y0=base_low,
        x1=str(base_end_date),   y1=base_high,
        line=dict(color=frame_color, width=1.5, dash="dot"),
        fillcolor=frame_fill,
        row=1, col=1
    )
    fig.add_annotation(
        x=str(base_start_date), y=base_high,
        text=(f"BASE ZONE ({consolidation_window}d)<br>"
              f"H: Rs{base_high:.2f}  L: Rs{base_low:.2f}  Range: {base_tightness:.1f}%"),
        font=dict(color=frame_color, size=10),
        showarrow=False, xanchor="left", yanchor="bottom",
        bgcolor="rgba(0,0,0,0.65)", bordercolor=frame_color, borderwidth=1, borderpad=5
    )

    # ── BREAKOUT LEVEL — the actual O(N) trigger ──────────────────────────────
    fig.add_hline(y=base_high, line_color="#FFD700", line_width=2, row=1, col=1)
    fig.add_annotation(
        x=str(df["DATE"].iloc[0]), y=base_high,
        text=f"BREAKOUT LEVEL  Rs{base_high:.2f}",
        font=dict(color="#FFD700", size=11, family="monospace"),
        showarrow=False, xanchor="left", yanchor="top",
        bgcolor="rgba(0,0,0,0.65)", borderpad=4
    )

    # ── Entry / Stop / Target levels ──────────────────────────────────────────
    if entry_price:
        fig.add_hline(y=entry_price, line_color="#00CC66",
                      line_width=1.5, row=1, col=1)
        fig.add_annotation(
            x=str(df["DATE"].iloc[-1]), y=entry_price,
            text=f"ENTRY  Rs{entry_price:.2f}",
            font=dict(color="#00CC66", size=10),
            showarrow=False, xanchor="right", yanchor="top",
            bgcolor="rgba(0,0,0,0.6)", borderpad=3
        )
    if stop_loss:
        fig.add_hline(y=stop_loss, line_color="#FF4444",
                      line_width=1.5, line_dash="dash", row=1, col=1)
        fig.add_annotation(
            x=str(df["DATE"].iloc[-1]), y=stop_loss,
            text=f"STOP  Rs{stop_loss:.2f}",
            font=dict(color="#FF4444", size=10),
            showarrow=False, xanchor="right", yanchor="top",
            bgcolor="rgba(0,0,0,0.6)", borderpad=3
        )
    if target_price:
        fig.add_hline(y=target_price, line_color="#4488FF",
                      line_width=1.5, line_dash="dash", row=1, col=1)
        fig.add_annotation(
            x=str(df["DATE"].iloc[-1]), y=target_price,
            text=f"TARGET  Rs{target_price:.2f}",
            font=dict(color="#4488FF", size=10),
            showarrow=False, xanchor="right", yanchor="bottom",
            bgcolor="rgba(0,0,0,0.6)", borderpad=3
        )

    if exit_price:
        fig.add_hline(y=exit_price, line_color="#FFD700",
                      line_width=1.5, line_dash="dot", row=1, col=1)
        fig.add_annotation(
            x=str(df["DATE"].iloc[-1]), y=exit_price,
            text=f"EXIT  Rs{exit_price:.2f}",
            font=dict(color="#FFD700", size=10),
            showarrow=False, xanchor="right", yanchor="bottom",
            bgcolor="rgba(0,0,0,0.6)", borderpad=3
        )

    # ── Signal candle marker ──────────────────────────────────────────────────
    if sig_high is not None:
        fig.add_annotation(
            x=str(signal_date), y=sig_high,
            text="SIGNAL",
            font=dict(color=frame_color, size=12, family="monospace"),
            showarrow=True, arrowhead=2, arrowcolor=frame_color,
            arrowsize=1.5, ay=-36,
            bgcolor="rgba(0,0,0,0.72)", bordercolor=frame_color, borderwidth=1, borderpad=4
        )

        # O(N^2) flaw watermark
        if is_flawed:
            fig.add_annotation(
                x=0.5, y=0.52, xref="paper", yref="paper",
                text=("WARNING: O(N2) FLAW DETECTED<br>"
                      "Trigger is BELOW base high<br>"
                      "Pullback Recovery -- NOT a Breakout"),
                font=dict(size=17, color="rgba(255,80,80,0.92)", family="monospace"),
                showarrow=False, align="center",
                bgcolor="rgba(10,0,0,0.85)",
                bordercolor="#FF4444", borderwidth=2, borderpad=14
            )

    # ── Signal date divider ───────────────────────────────────────────────────
    fig.add_vline(
        x=str(signal_date),
        line_dash="dash", line_color="rgba(200,200,200,0.4)", line_width=1
    )
    fig.add_annotation(
        x=str(signal_date), y=1.02, yref="paper",
        text="Signal Date", showarrow=False,
        font=dict(color="rgba(200,200,200,0.6)", size=9), xanchor="left"
    )

    # ── Forensic Metrics Panel ────────────────────────────────────────────────
    if wick_pct is not None:
        try:
            rr = round((target_price - entry_price) / (entry_price - stop_loss), 2) \
                if (target_price and entry_price and stop_loss and (entry_price - stop_loss) > 0) \
                else "N/A"
        except:
            rr = "N/A"

        panel = [
            "FORENSIC METRICS",
            "-----------------------------",
        ]
        if is_rejected:
            panel.append(f"Reject:   {rejection_reason or 'N/A'}")
            if ghost_entry:
                panel.append(f"Ghost Ent:₹{ghost_entry:.2f}")
        panel += [
            f"Pattern:  {pattern_type}",
            f"Regime:   {market_regime}",
            f"Cohorts:  {cohort_text}",
            f"Quality:  {data_quality}",
            f"Ext %:    {breakout_ext_pct:+.2f}%",
            f"Wick %:   {wick_pct:.1f}%",
            f"Cls/High: {close_vs_high:.1f}%",
            f"Vol Z:    {vol_zscore:.2f}x",
            f"Base Rng: {base_tightness:.2f}%",
            f"R:R:      {rr}",
            f"Window:   {consolidation_window}d",
            f"Source:   {source}",
        ]
        if pnl_pct is not None:
            panel.append(f"PnL:      {pnl_pct:+.2f}%")
        if is_flawed:
            panel.append("*** O(N2) FLAW ***")

        fig.add_annotation(
            x=0.01, y=0.97, xref="paper", yref="paper",
            text="<br>".join(panel),
            font=dict(size=11, color="#CCCCCC", family="monospace"),
            showarrow=False, align="left", xanchor="left", yanchor="top",
            bgcolor="rgba(12,12,28,0.90)",
            bordercolor=frame_color, borderwidth=1.5, borderpad=10
        )

    # ── Layout ────────────────────────────────────────────────────────────────
    fig.update_layout(
        title=dict(
            text=title_str,
            font=dict(size=15, color=frame_color, family="monospace"),
            x=0.0, xanchor="left"
        ),
        xaxis_rangeslider_visible=False,
        height=880,
        template="plotly_dark",
        paper_bgcolor="#080814",
        plot_bgcolor="#080814",
        legend=dict(
            orientation="h", yanchor="bottom", y=1.01,
            xanchor="right", x=1,
            font=dict(size=10)
        ),
        margin=dict(l=60, r=60, t=80, b=40),
    )

    # Colored outer border frame
    fig.add_shape(
        type="rect", xref="paper", yref="paper",
        x0=0, y0=0, x1=1, y1=1,
        line=dict(color=frame_color, width=3),
        fillcolor="rgba(0,0,0,0)", layer="above"
    )

    fig.update_xaxes(gridcolor="rgba(255,255,255,0.05)")
    fig.update_yaxes(gridcolor="rgba(255,255,255,0.05)")

    filename = f"{OUTPUT_DIR}/{signal_date.strftime('%Y-%m-%d')}_{symbol}_{status_text}.html"
    fig.write_html(filename)
    print(f"  Saved: {filename}")
    return filename


def generate_index(accepted_data, rejected_data):
    """Generates a summary index page for easy forensic auditing."""
    print("Generating Forensic Audit Index...")
    
    # CSS & Header
    html_start = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>IPO Base Scanner | Forensic Audit Index</title>
    <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600&display=swap" rel="stylesheet">
    <style>
        :root {{
            --bg: #080814;
            --card-bg: #121228;
            --text: #e0e0e0;
            --primary: #4DA8DA;
            --accepted: #00CC66;
            --rejected: #FF4444;
            --accent: #FFD700;
        }}
        body {{ 
            font-family: 'Outfit', sans-serif; 
            background-color: var(--bg); 
            color: var(--text); 
            margin: 0; 
            padding: 40px;
            line-height: 1.6;
        }}
        .header {{
            text-align: center;
            margin-bottom: 60px;
        }}
        h1 {{ 
            font-size: 2.8rem; 
            color: var(--primary); 
            margin-bottom: 10px;
            letter-spacing: -1px;
            font-weight: 600;
        }}
        p.subtitle {{
            font-size: 1.2rem;
            opacity: 0.6;
            font-weight: 300;
        }}
        .container {{ 
            display: grid; 
            grid-template-columns: 1fr 1fr;
            gap: 40px; 
            max-width: 1400px;
            margin: 0 auto;
        }}
        .column {{ 
            background: var(--card-bg);
            padding: 35px;
            border-radius: 20px;
            box-shadow: 0 15px 40px rgba(0,0,0,0.4);
            border: 1px solid rgba(255,255,255,0.05);
        }}
        h2 {{ 
            font-size: 1.6rem;
            margin-top: 0;
            padding-bottom: 20px;
            border-bottom: 2px solid rgba(255,255,255,0.08);
            display: flex;
            align-items: center;
            gap: 12px;
            font-weight: 600;
        }}
        h2.accepted {{ color: var(--accepted); }}
        h2.rejected {{ color: var(--rejected); }}
        
        ul {{ list-style: none; padding: 0; }}
        li {{ 
            margin: 14px 0; 
        }}
        a {{ 
            color: var(--text); 
            text-decoration: none; 
            font-size: 1.1rem; 
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 12px 20px;
            background: rgba(255,255,255,0.02);
            border-radius: 12px;
            border: 1px solid rgba(255,255,255,0.03);
            transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1);
        }}
        a:hover {{ 
            background: rgba(255,255,255,0.07);
            border-color: rgba(255,255,255,0.15);
            color: #fff;
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(0,0,0,0.2);
        }}
        .symbol {{
            font-weight: 600;
            letter-spacing: 0.5px;
            min-width: 100px;
        }}
        .date {{
            font-size: 0.9rem;
            opacity: 0.4;
            font-family: 'JetBrains Mono', monospace;
        }}
        .badge {{
            font-size: 0.75rem;
            padding: 4px 10px;
            border-radius: 6px;
            text-transform: uppercase;
            font-weight: 600;
            letter-spacing: 0.5px;
        }}
        .badge-accepted {{ background: rgba(0,204,102,0.1); color: var(--accepted); border: 1px solid rgba(0,204,102,0.3); }}
        .badge-rejected {{ background: rgba(255,68,68,0.1); color: var(--rejected); border: 1px solid rgba(255,68,68,0.3); }}
        
        .cohorts {{
            display: flex;
            gap: 4px;
        }}
        .cohort-badge {{
            font-size: 0.65rem;
            padding: 2px 6px;
            border-radius: 4px;
            background: rgba(255,255,255,0.05);
            border: 1px solid rgba(255,255,255,0.1);
            color: #AAA;
            font-weight: 600;
        }}
        .cohort-P {{ border-color: rgba(0,204,255,0.4); color: #00ccff; }}
        .cohort-S {{ border-color: rgba(255,204,0,0.4); color: #ffcc00; }}
        .cohort-U {{ border-color: rgba(255,102,204,0.4); color: #ff66cc; }}
        
        @media (max-width: 1000px) {{
            .container {{ grid-template-columns: 1fr; }}
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Forensic Audit Index</h1>
        <p class="subtitle">Evidence Panels for IPO Base Geometric Verification</p>
    </div>
    
    <div class="container">
        <div class="column">
            <h2 class="accepted">🟢 ACCEPTED Signals</h2>
            <ul>
"""

    # Build Accepted List
    acc_html = ""
    if not accepted_data:
        acc_html = "<li><p style='opacity:0.4; font-style:italic;'>No accepted signals found.</p></li>"
    else:
        for s in accepted_data:
            cohort_badges = ""
            for c in s.get("cohorts", []):
                char = c[0].upper()
                cohort_badges += f'<span class="cohort-badge cohort-{char}">{char}</span>'
            
            acc_html += f"""
                <li>
                    <a href='{s['file']}' target='_blank'>
                        <div class='header-row'>
                            <span class='symbol'>{s['symbol']}</span>
                            <span class='date'>{s['date']}</span>
                        </div>
                        <div class='meta-row'>
                            <span class='pattern'>{s['pattern']}</span>
                            <span class='regime'>{s['regime']}</span>
                        </div>
                        <span class='badge badge-accepted'>ACTIVE / CLOSED</span>
                        <div class='cohorts'>{cohort_badges}</div>
                    </a>
                </li>"""

    mid_html = """
            </ul>
        </div>
        <div class="column">
            <h2 class="rejected">🔴 REJECTED Signals</h2>
            <ul>
"""

    # Build Rejected List
    rej_html = ""
    if not rejected_data:
        rej_html = "<li><p style='opacity:0.4; font-style:italic;'>No rejected signals found.</p></li>"
    else:
        for s in rejected_data:
            rej_html += f"""
                <li>
                    <a href='{s['file']}' target='_blank'>
                        <div class='header-row'>
                            <span class='symbol'>{s['symbol']}</span>
                            <span class='date'>{s['date']}</span>
                        </div>
                        <div class='meta-row'>
                            <span class='pattern'>{s['pattern']}</span>
                            <span class='reason'>{s['reason']}</span>
                        </div>
                        <span class='badge badge-rejected'>FILTERED</span>
                    </a>
                </li>"""

    end_html = """
            </ul>
        </div>
    </div>
</body>
</html>
"""

    full_html = html_start + acc_html + mid_html + rej_html + end_html
    with open(os.path.join(OUTPUT_DIR, "index.html"), "w", encoding="utf-8") as f:
        f.write(full_html)
    print(f"  [DONE] Index generated at {os.path.join(OUTPUT_DIR, 'index.html')}")


def run():

    print("Generating Forensic Evidence Panels...")

    # Clear all old charts (preserve index.html)
    old_charts = [
        f for f in os.listdir(OUTPUT_DIR)
        if f.endswith(".html") and f != "index.html"
    ]
    if old_charts:
        print(f"  Clearing {len(old_charts)} old chart(s)...")
        for fname in old_charts:
            os.remove(os.path.join(OUTPUT_DIR, fname))
        print("  Cleared.")

    print("Fetching ACCEPTED signals...")
    accepted_cursor = signals_col.find(
        {"status": {"$in": ["ACTIVE", "CLOSED"]}}
    ).sort("created_at", -1).limit(300)
    
    accepted_data = []
    for doc in accepted_cursor:
        fname = plot_signal(doc, is_rejected=False)
        if fname:
            accepted_data.append({
                "file": fname.split("/")[-1],
                "symbol": doc.get("symbol"),
                "date": pd.to_datetime(doc.get("created_at")).strftime("%Y-%m-%d"),
                "cohorts": doc.get("valid_cohorts", []),
                "pattern": doc.get("pattern_type", "UNKNOWN"),
                "regime": doc.get("market_regime", "UNKNOWN")
            })

    print("Fetching REJECTED signals...")
    rejected_cursor = logs_col.find(
        {"action": "REJECTED_BREAKOUT"}
    ).sort("_id", -1).limit(300)

    rejected_data = []
    processed = set()
    for doc in rejected_cursor:
        sym = doc.get("symbol")
        if sym in processed:
            continue
        processed.add(sym)
        fname = plot_signal(doc, is_rejected=True)
        if fname:
            rejected_data.append({
                "file": fname.split("/")[-1],
                "symbol": doc.get("symbol"),
                "date": pd.to_datetime(doc.get("timestamp")).strftime("%Y-%m-%d"),
                "pattern": doc.get("pattern_type", "UNKNOWN"),
                "regime": doc.get("market_regime", "UNKNOWN"),
                "reason": doc.get("rejection_reason", "N/A")
            })

    print(f"\n[DONE] All forensic charts generated in ./{OUTPUT_DIR}/")
    generate_index(accepted_data, rejected_data)



if __name__ == "__main__":
    run()
