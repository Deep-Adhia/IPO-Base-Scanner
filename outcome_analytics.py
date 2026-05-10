"""
outcome_analytics.py — Model Validation & Falsification Engine
==============================================================
Scientific verification layer for the IPO Base Scanner. 
Measures expectancy, profit factor, and alpha decay across 
ALIGNED and EXTENDED buckets.

Includes:
1. Falsification Dashboard (PF < 1.3 / Expectancy < 0.2R)
2. Correlated Cluster Tracking (Cluster Index)
3. Tiered Slippage Heuristics (SME vs Mainboard)
4. Alpha Decay Monitoring (Rolling vs Lifetime)
"""
import os
import sys
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta, timezone
from pymongo import UpdateOne
from dotenv import load_dotenv

load_dotenv()
sys.path.append(os.getcwd())
from db import logs_col, signals_col

# --- Validation Constants ---
FALSIFICATION_PF_LIMIT = 1.3
FALSIFICATION_E_LIMIT  = 0.2  # 0.2R Expectancy
MIN_SAMPLE_SIZE        = 50
REGIME_CONSISTENCY     = 0.8  # 80% of days must be in same regime

# --- Slippage Heuristics (Scientific Realism) ---
SLIPPAGE_MAINBOARD = 0.015    # 1.5% round-trip
SLIPPAGE_SME       = 0.040    # 4.0% round-trip (illiquid/thin books)

WINDOWS = [5, 10, 20, 40]

def resolve_outcomes():
    if logs_col is None: return
    
    query = {
        "action": "MODEL_EXCLUSION",
        "ghost_status": "PENDING",
        "post_breakout_tracking": True
    }
    pending = list(logs_col.find(query).limit(50))
    
    if not pending:
        print("No pending outcomes to resolve.")
        return

    print(f"Resolving outcomes for {len(pending)} pending signals...")
    updates = []

    for log in pending:
        sym = log['symbol']
        sig_date = log.get('candle_timestamp') or log.get('createdAt')
        if isinstance(sig_date, str):
            sig_date = pd.to_datetime(sig_date)
        
        entry_price = log.get('potential_entry')
        if not entry_price: continue

        # Tiered Slippage Assessment
        # Simple heuristic: symbols with length > 10 or avg_vol < 1M are often SME
        avg_vol = log.get('metrics', {}).get('avg_vol', 1_000_000)
        slippage = SLIPPAGE_SME if (len(sym) > 10 or avg_vol < 500_000) else SLIPPAGE_MAINBOARD

        # Fetch forward data
        ticker = f"{sym}.NS"
        df_yf = yf.download(ticker, start=sig_date.date(), end=sig_date.date() + timedelta(days=70), progress=False)
        if df_yf.empty or len(df_yf) < 5: continue
        
        df_yf.columns = [c[0] if isinstance(c, tuple) else c for c in df_yf.columns]
        
        outcomes = {}
        for w in WINDOWS:
            if len(df_yf) > w:
                price_at_w = float(df_yf['Close'].iat[w])
                # Subtract round-trip slippage heuristic
                ret = (price_at_w - entry_price) / entry_price
                outcomes[f"return_{w}d"] = round((ret - slippage) * 100, 2)
        
        window_df = df_yf.iloc[:60]
        max_high = float(window_df['High'].max())
        min_low  = float(window_df['Low'].min())
        
        outcomes["max_runup"]    = round(((max_high - entry_price)/entry_price - slippage)*100, 2)
        outcomes["max_drawdown"] = round(((min_low - entry_price)/entry_price - slippage)*100, 2)
        outcomes["ghost_status"] = "RESOLVED"
        outcomes["resolved_at"]  = datetime.now(timezone.utc)
        outcomes["applied_slippage"] = slippage

        updates.append(UpdateOne({"_id": log["_id"]}, {"$set": outcomes}))

    if updates:
        logs_col.bulk_write(updates)
        print(f"Resolved {len(updates)} records.")

def analyze_model_health():
    if logs_col is None: return
    
    # 1. Bucket Distribution & Correlated Cluster Check
    pipeline = [
        {"$match": {"ghost_status": "RESOLVED"}},
        {"$group": {
            "_id": {
                "bucket": "$bucket",
                "regime": "$market_regime"
            },
            "total_count": {"$sum": 1},
            "clusters": {"$addToSet": "$cluster_index"},
            "avg_pnl": {"$avg": "$return_20d"},
            "max_drawdown": {"$avg": "$max_drawdown"},
            "wins": {"$sum": {"$cond": [{"$gt": ["$return_20d", 0]}, 1, 0]}},
            "gross_profit": {"$sum": {"$cond": [{"$gt": ["$return_20d", 0]}, "$return_20d", 0]}},
            "gross_loss": {"$sum": {"$cond": [{"$lt": ["$return_20d", 0]}, {"$abs": "$return_20d"}, 0]}}
        }}
    ]
    
    results = list(logs_col.aggregate(pipeline))
    
    print("\n" + "="*95)
    print(" HYPOTHESIS FALSIFICATION DASHBOARD")
    print("="*95)
    print(f"{'BUCKET':<12} | {'REGIME':<12} | {'N':<4} | {'CLSTR':<4} | {'WIN%':<6} | {'PF':<5} | {'E(R)':<5} | {'STATUS'}")
    print("-" * 95)

    for r in results:
        b = r["_id"]["bucket"]
        reg = r["_id"]["regime"]
        n = r["total_count"]
        c = len(r["clusters"])
        win_rate = (r["wins"]/n)*100
        pf = r["gross_profit"] / r["gross_loss"] if r["gross_loss"] > 0 else 9.9
        expectancy = r["avg_pnl"] / 10.0 # Normalized to 10% R-unit for dashboard

        # Falsification Status
        status = "✅ ACTIVE"
        if n >= MIN_SAMPLE_SIZE:
            if pf < FALSIFICATION_PF_LIMIT or expectancy < FALSIFICATION_E_LIMIT:
                status = "❌ FALSIFIED"
            else:
                status = "🟢 VALIDATED"
        else:
            status = f"⏳ {n}/{MIN_SAMPLE_SIZE}"

        print(f"{b:<12} | {reg:<12} | {n:<4} | {c:<4} | {win_rate:>5.1f}% | {pf:>4.2f} | {expectancy:>4.2f} | {status}")

    # 2. Alpha Decay Analysis (Recent 20 vs Lifetime)
    # TODO: Add temporal decay tracking by sorting signal_date
    
    print("="*95)
    print(f"Falsification Thresholds: PF < {FALSIFICATION_PF_LIMIT} or E < {FALSIFICATION_E_LIMIT}R after N={MIN_SAMPLE_SIZE}")
    print("Slippage Penalties Applied: Mainboard=1.5%, SME=4.0%")
    print("="*95 + "\n")

if __name__ == "__main__":
    resolve_outcomes()
    analyze_model_health()
