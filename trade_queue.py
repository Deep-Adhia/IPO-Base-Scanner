"""
trade_queue.py — Active Trade Prioritization Queue (Phase 5)
=============================================================
PURPOSE: Rank ACTIVE signals by setup quality to aid capital allocation.
         This is a prioritization tool ONLY. It does NOT filter or suppress signals.
         Grade score influences execution order, never signal validity.

WEIGHTING (finalized in implementation plan):
  - Volume Expansion  (40%): Higher breakout vol relative to base avg = higher conviction
  - Base Tightness   (35%): Inverse PRNG — tighter base = cleaner pivot
  - Base Duration    (15%): Longer base = more distribution absorbed
  - Follow-through    (10%): Placeholder for future follow-through quality scoring

GUARDRAIL: Do NOT use this score to reject trades or add conditions.
"""

import sys
import os
from datetime import datetime
sys.path.append(os.getcwd())

from db import signals_col


# -- Grade Score Calculator ---------------------------------------------------

def compute_grade_score(signal: dict) -> float:
    """
    Returns a 0–100 grade score for prioritization.
    Lower is better for risk (tight base), higher is better for volume.
    """
    metrics = signal.get("metrics", {})
    vol_ratio = float(metrics.get("vol_ratio", 1.0))
    prng      = float(metrics.get("prng", 50.0))
    w         = int(metrics.get("w", 10))

    # Component scores (each 0–100)
    # Volume: log-normalised so a 5x ratio ≈ 80 and a 10x ratio ≈ 100
    import math
    vol_score = min(100, (math.log1p(max(vol_ratio - 1, 0)) / math.log1p(9)) * 100)

    # Tightness: 0% PRNG = 100, MAX_PRNG (45%) = 0, linear
    MAX_PRNG = 45.0
    tightness_score = max(0, (1 - prng / MAX_PRNG) * 100)

    # Duration: 30 days ≈ 100, <5 days ≈ 0, capped
    duration_score = min(100, (w / 30) * 100)

    # Follow-through: placeholder (no data yet, defaults to 50)
    followthrough_score = 50.0

    grade_score = (
        vol_ratio       * 0.40 * vol_score         / max(vol_ratio, 1) +
        tightness_score * 0.35 +
        duration_score  * 0.15 +
        followthrough_score * 0.10
    )

    # Simpler, more stable version:
    grade_score = (
        (vol_score * 0.40) +
        (tightness_score * 0.35) +
        (duration_score * 0.15) +
        (followthrough_score * 0.10)
    )

    return round(grade_score, 2)


# -- Main Queue Builder -------------------------------------------------------

def build_trade_queue(top_n: int = 5, cohort_filter: str = None):
    """
    Fetches all ACTIVE signals, computes grade_score, and prints the top N setups.
    Optionally filter by cohort (PERMISSIVE | STRICT | ULTRA_STRICT).
    """
    query = {"status": "ACTIVE"}
    if cohort_filter:
        query["valid_cohorts"] = cohort_filter

    active_signals = list(signals_col.find(query))
    if not active_signals:
        print("No ACTIVE signals found in the database.")
        return []

    # Compute and attach grade_score
    for sig in active_signals:
        sig["_grade_score"] = compute_grade_score(sig)

    # Sort descending by grade_score
    ranked = sorted(active_signals, key=lambda s: s["_grade_score"], reverse=True)

    # Persist grade_score back to DB for future reference
    for sig in ranked:
        signals_col.update_one(
            {"signal_id": sig["signal_id"]},
            {"$set": {"grade_score": sig["_grade_score"]}}
        )

    # -- Print Output ----------------------------------------------------------
    print(f"\n{'='*65}")
    print(f"  IPO FORENSIC SCANNER — ACTIVE TRADE QUEUE  ({datetime.today().date()})")
    print(f"{'='*65}")
    print(f"  Total active signals: {len(active_signals)}")
    print(f"  Showing top {min(top_n, len(ranked))} by grade score")
    print(f"{'='*65}\n")

    for rank, sig in enumerate(ranked[:top_n], start=1):
        pattern   = sig.get("pattern_type", "UNKNOWN")
        regime    = sig.get("market_regime", "UNKNOWN")
        cohorts   = ", ".join(sig.get("valid_cohorts", []))
        entry     = sig.get("entry_price", 0)
        stop      = sig.get("stop_loss", 0)
        target    = sig.get("target_price", 0)
        risk_pct  = round((entry - stop) / entry * 100, 1) if entry > 0 else 0
        rr        = round((target - entry) / (entry - stop), 1) if (entry - stop) > 0 else 0
        score     = sig["_grade_score"]
        created   = sig.get("created_at", "")

        print(f"  #{rank}  {sig['symbol']:<12}  Score: {score:>6.1f}  |  {sig.get('grade','')}")
        print(f"      Pattern : {pattern}")
        print(f"      Regime  : {regime}")
        print(f"      Cohorts : {cohorts}")
        print(f"      Entry   : Rs.{entry:.2f}   Stop: Rs.{stop:.2f}   Target: Rs.{target:.2f}")
        print(f"      Risk    : {risk_pct}%   R/R: 1:{rr}   Signal Date: {str(created)[:10]}")
        print()

    # Also print a summary of ALL signals by pattern type
    by_pattern = {}
    for sig in active_signals:
        pt = sig.get("pattern_type", "UNKNOWN")
        by_pattern.setdefault(pt, []).append(sig)

    print(f"{'-'*65}")
    print("  FULL ACTIVE BREAKDOWN BY PATTERN TYPE")
    print(f"{'-'*65}")
    for pt, sigs in sorted(by_pattern.items()):
        print(f"  {pt:<35}  {len(sigs):>3} signals")
    print(f"{'='*65}\n")

    return ranked


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="IPO Trade Prioritization Queue")
    parser.add_argument("--top",    type=int, default=5,    help="Number of top setups to show")
    parser.add_argument("--cohort", type=str, default=None, help="Filter by cohort (STRICT, ULTRA_STRICT, PERMISSIVE)")
    args = parser.parse_args()

    build_trade_queue(top_n=args.top, cohort_filter=args.cohort)
