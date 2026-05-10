"""
outcome_analytics.py — Model Validation & Expectancy Tracking
=============================================================
Calculates the "Truth" for both ALIGNED and EXTENDED signals.
Computes multi-window returns (5d, 10d, 20d, 40d) to determine
if our filters (PRNG/VOL) are alpha-generators or noise.
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

# Centralized imports from project db handler
sys.path.append(os.getcwd())
from db import logs_col, signals_col

WINDOWS = [5, 10, 20, 40]

def fetch_forward_data(symbol, start_date, days_needed=60):
    """Fetch price action for the window after the signal."""
    ticker = f"{symbol}.NS"
    end_date = start_date + timedelta(days=days_needed + 10) 
    try:
        df = yf.download(ticker, start=start_date, end=end_date, progress=False, auto_adjust=True)
        if df.empty: return None
        df = df.reset_index()
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
        df = df.rename(columns={"Date":"DATE","High":"HIGH","Low":"LOW","Close":"CLOSE","Volume":"VOLUME"})
        return df
    except:
        return None

def resolve_outcomes():
    if logs_col is None:
        print("Database not connected.")
        return
        
    # 1. Find PENDING exclusions (ALIGNED or EXTENDED)
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
        if not entry_price:
            continue

        df = fetch_forward_data(sym, sig_date.date())
        if df is None or len(df) < 5:
            continue

        outcomes = {}
        for w in WINDOWS:
            if len(df) > w:
                price_at_w = float(df['CLOSE'].iat[w])
                outcomes[f"return_{w}d"] = round((price_at_w - entry_price) / entry_price * 100, 2)
        
        window_df = df.iloc[:60]
        max_high = float(window_df['HIGH'].max())
        min_low  = float(window_df['LOW'].min())
        
        outcomes["max_runup"]    = round((max_high - entry_price) / entry_price * 100, 2)
        outcomes["max_drawdown"] = round((min_low - entry_price) / entry_price * 100, 2)
        outcomes["ghost_status"] = "RESOLVED"
        outcomes["resolved_at"]  = datetime.now(timezone.utc)

        updates.append(UpdateOne({"_id": log["_id"]}, {"$set": outcomes}))

    if updates:
        res = logs_col.bulk_write(updates)
        print(f"Successfully resolved {res.modified_count} outcomes.")

def analyze_bucket_expectancy():
    if logs_col is None: return
    
    pipeline = [
        {"$match": {"ghost_status": "RESOLVED"}},
        {"$group": {
            "_id": "$bucket",
            "count": {"$sum": 1},
            "avg_5d": {"$avg": "$return_5d"},
            "avg_20d": {"$avg": "$return_20d"},
            "avg_max_runup": {"$avg": "$max_runup"},
            "avg_max_drawdown": {"$avg": "$max_drawdown"}
        }}
    ]
    
    results = list(logs_col.aggregate(pipeline))
    print("\n" + "="*80)
    print(f"{'BUCKET':<12} | {'N':<4} | {'Avg 5d':<8} | {'Avg 20d':<8} | {'Avg MaxUp':<10} | {'Avg MaxDn'}")
    print("-" * 80)
    for r in results:
        print(f"{r['_id']:<12} | {r['count']:<4} | {r['avg_5d']:>7.1f}% | {r['avg_20d']:>7.1f}% | {r['avg_max_runup']:>8.1f}% | {r['avg_max_drawdown']:>8.1f}%")
    print("="*80 + "\n")

if __name__ == "__main__":
    resolve_outcomes()
    analyze_bucket_expectancy()
