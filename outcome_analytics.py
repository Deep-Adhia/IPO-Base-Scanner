"""
outcome_analytics.py — Edge Research & Expectancy Tracking (Phase 6)
====================================================================
PURPOSE: The final validation layer. Analyzes both ACCEPTED and REJECTED
         signals to determine the true statistical edge of different
         Pattern Archetypes and Market Regimes.

KEY METRICS:
  - Avg R-Multiple & Median R
  - Win Rate (Secondary to R-Multiple)
  - Time-to-Target (Efficiency)
  - Ghost PnL (Are we rejecting true winners?)
  - False Positive Rate (Failure within 5 candles)

This script will run periodically (e.g., weekly) to update research models.
"""

import os
import sys
import pandas as pd
from datetime import datetime, timedelta

sys.path.append(os.getcwd())
from db import signals_col, logs_col

# ── 1. Ghost PnL Resolution ──────────────────────────────────────────────────

def resolve_pending_ghost_pnl():
    """
    Finds REJECTED logs with ghost_status == "PENDING".
    If the observation window has passed, it calculates the MFE/MAE and final PnL
    against the actual historical price action and marks it "RESOLVED".
    """
    print("Resolving pending Ghost PnL records... (Skeleton)")
    # TODO: Implement historical data fetch for the window after rejection
    # Compute:
    # 1. Did it hit target before stop?
    # 2. Maximum Favorable Excursion (MFE)
    # 3. Maximum Adverse Excursion (MAE)
    pass


# ── 2. Expectancy & Edge Analytics ───────────────────────────────────────────

def analyze_archetype_expectancy():
    """
    Groups signals by pattern_type and market_regime to compute real edge.
    """
    print("\n[ Analyzing Archetype Expectancy ]")
    
    pipeline = [
        {"$match": {"status": "CLOSED"}},
        {"$group": {
            "_id": {
                "pattern": "$pattern_type",
                "regime": "$market_regime"
            },
            "count": {"$sum": 1},
            "avg_pnl": {"$avg": "$pnl_pct"},
            "wins": {"$sum": {"$cond": [{"$gt": ["$pnl_pct", 0]}, 1, 0]}}
        }},
        {"$sort": {"count": -1}}
    ]
    
    results = list(signals_col.aggregate(pipeline))
    
    if not results:
        print("  Not enough closed data to run analytics.")
        return
        
    for r in results:
        pat = r["_id"].get("pattern", "UNKNOWN")
        reg = r["_id"].get("regime", "UNKNOWN")
        cnt = r["count"]
        pnl = r.get("avg_pnl") or 0.0
        wins = r.get("wins", 0)
        win_rate = (wins / cnt) * 100 if cnt > 0 else 0
        
        print(f"  {pat[:30]:<30} | {reg:<12} | N={cnt:<4} | WinRate: {win_rate:>5.1f}% | AvgPnL: {pnl:>5.1f}%")

def analyze_rejection_efficiency():
    """
    Checks if our rejection filters are saving us money or costing us alpha.
    """
    print("\n[ Analyzing Rejection Efficiency (Ghost PnL) ]")
    # TODO: Aggregate RESOLVED ghost PnL to see if certain rejection_reasons
    # have positive expectancy (i.e., we shouldn't be rejecting them).
    print("  (Pending Ghost PnL resolution data...)")

if __name__ == "__main__":
    resolve_pending_ghost_pnl()
    analyze_archetype_expectancy()
    analyze_rejection_efficiency()
    
    print("\n[ Phase 6 Analytics Groundwork Ready ]")
