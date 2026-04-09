#!/usr/bin/env python3
"""
Generate monthly_system_report.json from scanner logs + positions.

Primary sources:
- logs/YYYY-MM-DD/listing_day.jsonl
- ipo_positions.csv
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
from collections import Counter
from datetime import datetime
from statistics import mean


def parse_ts(ts: str) -> datetime | None:
    if not ts:
        return None
    cleaned = ts.replace(" IST", "").strip()
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
        try:
            return datetime.strptime(cleaned, fmt)
        except ValueError:
            continue
    return None


def to_float(value) -> float | None:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def load_listing_rows(logs_root: str) -> list[dict]:
    rows: list[dict] = []
    pattern = os.path.join(logs_root, "*", "listing_day.jsonl")
    for path in sorted(glob.glob(pattern)):
        try:
            with open(path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if row.get("scanner") == "listing_day":
                        rows.append(row)
        except OSError:
            continue
    return rows


def in_month(row: dict, month: str) -> bool:
    ts = parse_ts(row.get("timestamp", ""))
    return bool(ts and ts.strftime("%Y-%m") == month)


def load_positions(path: str, month: str) -> list[dict]:
    if not os.path.isfile(path):
        return []
    out: list[dict] = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            grade = (row.get("grade") or "").strip().upper()
            if grade != "LISTING_BREAKOUT":
                continue
            entry_date = (row.get("entry_date") or "").strip()
            if not entry_date.startswith(month):
                continue
            out.append(row)
    return out


def category_from_pnl(pnl: float | None, status: str) -> str:
    st = (status or "").upper()
    if st == "ACTIVE":
        return "OPEN"
    if pnl is None:
        return "UNKNOWN"
    return "WINNER" if pnl > 0 else "LOSER"


def correlation_block(trades: list[dict], key: str) -> dict:
    values = [t for t in trades if t.get(key) is not None and t.get("pnl_pct") is not None]
    if not values:
        return {"sample_size": 0}
    pairs = sorted(values, key=lambda x: x[key])
    buckets = {"low": [], "mid": [], "high": []}
    n = len(pairs)
    for i, t in enumerate(pairs):
        if i < n / 3:
            buckets["low"].append(t["pnl_pct"])
        elif i < 2 * n / 3:
            buckets["mid"].append(t["pnl_pct"])
        else:
            buckets["high"].append(t["pnl_pct"])
    return {
        "sample_size": n,
        "avg_pnl_low_bucket": round(mean(buckets["low"]), 2) if buckets["low"] else None,
        "avg_pnl_mid_bucket": round(mean(buckets["mid"]), 2) if buckets["mid"] else None,
        "avg_pnl_high_bucket": round(mean(buckets["high"]), 2) if buckets["high"] else None,
    }


def infer_why_outcome(trade: dict) -> list[str]:
    reasons: list[str] = []
    category = trade.get("category")
    pnl = trade.get("pnl_pct")
    days = trade.get("days_held")
    cfm = trade.get("confirmation_time_min")
    vol = trade.get("volume_multiple")
    score = trade.get("leader_score")
    exit_reason = (trade.get("exit_reason") or "").upper()

    if category == "WINNER":
        if pnl is not None and pnl >= 15:
            reasons.append("strong_momentum_follow_through")
        if days is not None and days >= 10:
            reasons.append("trend_persistence")
        if cfm is not None and cfm <= 45:
            reasons.append("fast_confirmation")
        if score is not None and score >= 7:
            reasons.append("high_leader_score")
        if vol is not None and vol >= 1.8:
            reasons.append("strong_volume_participation")
        if "TRAIL" in exit_reason:
            reasons.append("protected_profits_with_trailing_exit")
    elif category == "LOSER":
        if pnl is not None and pnl <= -7:
            reasons.append("deep_adverse_move")
        if days is not None and days <= 3:
            reasons.append("failed_soon_after_entry")
        if cfm is not None and cfm >= 75:
            reasons.append("slow_confirmation_weaker_demand")
        if score is not None and score <= 4:
            reasons.append("low_leader_score")
        if vol is not None and vol < 1.2:
            reasons.append("weak_volume_confirmation")
        if "SL" in exit_reason or "STOP" in exit_reason:
            reasons.append("stopped_out")
    else:
        reasons.append("open_or_insufficient_outcome_data")

    if not reasons:
        reasons.append("insufficient_feature_history")
    return reasons


def aggregate_why(trades: list[dict], category: str) -> dict:
    counter: Counter[str] = Counter()
    sample = 0
    for t in trades:
        if t.get("category") != category:
            continue
        sample += 1
        for reason in t.get("why_outcome", []):
            counter[reason] += 1
    top = [{"reason": r, "count": c} for r, c in counter.most_common(8)]
    return {"sample_size": sample, "top_reasons": top}


def classify_primary_cause(trade: dict) -> tuple[str, float]:
    category = trade.get("category")
    pnl = trade.get("pnl_pct")
    days = trade.get("days_held")
    cfm = trade.get("confirmation_time_min")
    vol = trade.get("volume_multiple")
    score = trade.get("leader_score")
    exit_reason = (trade.get("exit_reason") or "").upper()

    weighted: dict[str, float] = {}

    def add(label: str, weight: float) -> None:
        weighted[label] = weighted.get(label, 0.0) + weight

    if category == "WINNER":
        if pnl is not None:
            if pnl >= 20:
                add("strong_momentum_follow_through", 4.0)
            elif pnl >= 10:
                add("steady_positive_follow_through", 2.5)
        if days is not None and days >= 10:
            add("trend_persistence", 2.5)
        if cfm is not None:
            if cfm <= 30:
                add("fast_confirmation", 2.5)
            elif cfm <= 60:
                add("normal_confirmation_quality", 1.0)
        if score is not None:
            if score >= 7:
                add("high_leader_score", 2.0)
            elif score >= 5:
                add("acceptable_leader_score", 1.0)
        if vol is not None and vol >= 1.8:
            add("strong_volume_participation", 1.5)
        if "TRAIL" in exit_reason:
            add("protected_profits_with_trailing_exit", 2.0)

    elif category == "LOSER":
        if pnl is not None:
            if pnl <= -9:
                add("deep_adverse_move", 4.0)
            elif pnl <= -6:
                add("moderate_adverse_move", 2.5)
        if days is not None and days <= 3:
            add("failed_soon_after_entry", 2.5)
        if cfm is not None and cfm >= 75:
            add("slow_confirmation_weaker_demand", 2.0)
        if score is not None and score <= 4:
            add("low_leader_score", 2.0)
        if vol is not None and vol < 1.2:
            add("weak_volume_confirmation", 1.5)
        if "SL" in exit_reason or "STOP" in exit_reason:
            add("stopped_out", 2.0)

    else:
        add("open_or_insufficient_outcome_data", 1.0)

    if not weighted:
        return "insufficient_feature_history", 0.0

    ordered = sorted(weighted.items(), key=lambda x: x[1], reverse=True)
    best_label, best_score = ordered[0]
    total = sum(weighted.values())
    confidence = round(best_score / total, 3) if total > 0 else 0.0
    return best_label, confidence


def aggregate_primary_causes(trades: list[dict], category: str) -> dict:
    counter: Counter[str] = Counter()
    conf_sum = 0.0
    sample = 0
    for t in trades:
        if t.get("category") != category:
            continue
        sample += 1
        label = t.get("primary_outcome_cause") or "unknown"
        counter[label] += 1
        conf_sum += float(t.get("cause_confidence") or 0.0)
    distribution = [{"cause": k, "count": v} for k, v in counter.most_common(8)]
    return {
        "sample_size": sample,
        "avg_confidence": round(conf_sum / sample, 3) if sample else 0.0,
        "distribution": distribution,
    }


def build_report(month: str, logs_root: str, positions_path: str) -> dict:
    rows = [r for r in load_listing_rows(logs_root) if in_month(r, month)]
    by_action = Counter(r.get("action") for r in rows if r.get("action"))
    reject_reasons = Counter(
        (r.get("details") or {}).get("reason", "unknown")
        for r in rows
        if r.get("action") == "PENDING_REJECTED"
    )

    confirmed_symbols = {
        (r.get("symbol") or "").upper()
        for r in rows
        if r.get("action") == "PENDING_CONFIRMED"
    }
    fallback_to_positions = not confirmed_symbols
    positions = load_positions(positions_path, month)

    first_breakout_by_symbol: dict[str, dict] = {}
    confirm_details_by_symbol: dict[str, dict] = {}
    for r in rows:
        sym = (r.get("symbol") or "").upper()
        if not sym:
            continue
        if r.get("action") == "BREAKOUT_SIGNAL" and sym not in first_breakout_by_symbol:
            first_breakout_by_symbol[sym] = r
        if r.get("action") == "PENDING_CONFIRMED" and sym not in confirm_details_by_symbol:
            confirm_details_by_symbol[sym] = r

    trades: list[dict] = []
    for pos in positions:
        symbol = (pos.get("symbol") or "").upper()
        if not fallback_to_positions and symbol not in confirmed_symbols:
            continue

        breakout = first_breakout_by_symbol.get(symbol, {})
        breakout_details = breakout.get("details") or {}
        confirm = confirm_details_by_symbol.get(symbol, {})
        confirm_details = confirm.get("details") or {}

        entry_price = to_float(pos.get("entry_price"))
        exit_price = to_float(pos.get("exit_price"))
        pnl_pct = to_float(pos.get("pnl_pct"))
        current_price = to_float(pos.get("current_price"))
        status = pos.get("status") or ""
        resolved_exit_price = exit_price if exit_price is not None else current_price
        max_gain_pct = (
            round(((resolved_exit_price - entry_price) / entry_price) * 100, 2)
            if entry_price and resolved_exit_price is not None
            else None
        )

        trade = {
            "symbol": symbol,
            "entry_date": pos.get("entry_date"),
            "entry_price": entry_price,
            "exit_date": pos.get("exit_date") or None,
            "exit_price": exit_price,
            "pnl_pct": pnl_pct,
            "days_held": int(float(pos.get("days_held") or 0)),
            "leader_score": confirm_details.get("leader_score"),
            "volume_multiple": breakout_details.get("volume_multiple"),
            "confirmation_time_min": confirm_details.get("elapsed_minutes"),
            "max_gain_pct": max_gain_pct,
            "max_drawdown_pct": None,
            "exit_reason": pos.get("exit_reason") or "UNKNOWN",
            "category": category_from_pnl(pnl_pct, status),
        }
        trade["why_outcome"] = infer_why_outcome(trade)
        primary_cause, cause_confidence = classify_primary_cause(trade)
        trade["primary_outcome_cause"] = primary_cause
        trade["cause_confidence"] = cause_confidence
        trades.append(trade)

    closed = [t for t in trades if t["category"] in ("WINNER", "LOSER")]
    winners = [t for t in closed if t["category"] == "WINNER"]
    losers = [t for t in closed if t["category"] == "LOSER"]
    avg_win = round(mean([t["pnl_pct"] for t in winners]), 2) if winners else 0.0
    avg_loss = round(mean([t["pnl_pct"] for t in losers]), 2) if losers else 0.0
    win_rate = round((len(winners) / len(closed) * 100), 2) if closed else 0.0
    expectancy = round((win_rate / 100.0) * avg_win + (1 - win_rate / 100.0) * avg_loss, 2)

    summary = {
        "total_trades": len(closed),
        "winners": len(winners),
        "losers": len(losers),
        "win_rate": win_rate,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "largest_win": max((t["pnl_pct"] for t in winners), default=0.0),
        "largest_loss": min((t["pnl_pct"] for t in losers), default=0.0),
        "expectancy": expectancy,
    }

    pending_created = by_action.get("PENDING_STARTED", 0)
    confirmed = by_action.get("PENDING_CONFIRMED", 0)
    rejected = by_action.get("PENDING_REJECTED", 0)
    breakouts_detected = by_action.get("BREAKOUT_SIGNAL", 0)
    confirmation_rate = round(100.0 * confirmed / pending_created, 2) if pending_created else 0.0

    funnel = {
        "total_breakouts_detected": breakouts_detected,
        "pending_created": pending_created,
        "confirmed": confirmed,
        "rejected": rejected,
        "confirmation_rate": confirmation_rate,
    }

    report = {
        "month": month,
        "summary": summary,
        "funnel_metrics": funnel,
        "rejection_reasons": dict(reject_reasons),
        "trades": trades,
        "correlations": {
            "leader_score_vs_outcome": correlation_block(trades, "leader_score"),
            "confirmation_time_vs_outcome": correlation_block(trades, "confirmation_time_min"),
            "volume_multiple_vs_outcome": correlation_block(trades, "volume_multiple"),
        },
        "outcome_drivers": {
            "why_winners": aggregate_why(trades, "WINNER"),
            "why_losers": aggregate_why(trades, "LOSER"),
            "primary_causes_winners": aggregate_primary_causes(trades, "WINNER"),
            "primary_causes_losers": aggregate_primary_causes(trades, "LOSER"),
        },
        "metadata": {
            "fallback_mode_positions_used_without_confirm_logs": fallback_to_positions,
            "notes": [
                "max_drawdown_pct requires intraday/eod path history and is null when unavailable",
                "volume_multiple is null for legacy logs that did not store it",
                "confirmation_time_min is null for legacy logs without PENDING_CONFIRMED details",
                "why_outcome is heuristic and becomes stronger as richer fields are logged",
                "primary_outcome_cause and cause_confidence use weighted heuristic scoring",
            ],
        },
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate monthly system report JSON.")
    parser.add_argument("--month", required=True, help="Target month as YYYY-MM")
    parser.add_argument("--logs-root", default="logs", help="Logs root folder")
    parser.add_argument("--positions", default="ipo_positions.csv", help="Positions CSV path")
    parser.add_argument(
        "--output",
        default="monthly_system_report.json",
        help="Output JSON file path",
    )
    args = parser.parse_args()

    report = build_report(args.month, args.logs_root, args.positions)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)

    print(f"Wrote {args.output}")
    print(f"Month: {args.month}")
    print(f"Closed trades: {report['summary']['total_trades']}")
    print(f"Win rate: {report['summary']['win_rate']}%")
    print(f"Expectancy: {report['summary']['expectancy']}")


if __name__ == "__main__":
    main()
