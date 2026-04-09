#!/usr/bin/env python3
"""
Summarize listing-day pending confirmation funnel from logs/listing_day.jsonl.

Reads actions: PENDING_STARTED, PENDING_REJECTED, PENDING_CONFIRMED, BREAKOUT_SIGNAL.
Optionally joins confirmed symbols to ipo_positions.csv for outcome snapshot.
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
from collections import Counter, defaultdict
from datetime import datetime


CONFIRM_ACTIONS = frozenset(
    ("PENDING_STARTED", "PENDING_REJECTED", "PENDING_CONFIRMED", "BREAKOUT_SIGNAL")
)


def _parse_ts(ts: str) -> datetime | None:
    if not ts:
        return None
    s = ts.replace(" IST", "").strip()
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            continue
    return None


def load_listing_day_rows(logs_root: str) -> list[dict]:
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
                    if row.get("scanner") != "listing_day":
                        continue
                    rows.append(row)
        except OSError:
            continue
    return rows


def filter_by_date(
    rows: list[dict], from_date: str | None, to_date: str | None
) -> list[dict]:
    if not from_date and not to_date:
        return rows
    out: list[dict] = []
    for r in rows:
        ts = _parse_ts(r.get("timestamp", ""))
        if ts is None:
            continue
        d = ts.strftime("%Y-%m-%d")
        if from_date and d < from_date:
            continue
        if to_date and d > to_date:
            continue
        out.append(r)
    return out


def load_positions(path: str) -> dict[str, dict]:
    if not os.path.isfile(path):
        return {}
    by_sym: dict[str, dict] = {}
    with open(path, newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            sym = (row.get("symbol") or "").strip().upper()
            if sym:
                by_sym[sym] = row
    return by_sym


def load_pending_file(path: str) -> tuple[int, list[str]]:
    if not os.path.isfile(path):
        return 0, []
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return 0, []
    if not isinstance(data, dict):
        return 0, []
    syms = [s for s in data.keys() if isinstance(s, str)]
    return len(syms), sorted(syms)


def main() -> None:
    p = argparse.ArgumentParser(description="Analyze listing confirmation quality from daily JSONL logs.")
    p.add_argument("--logs-root", default="logs", help="Root folder with YYYY-MM-DD/listing_day.jsonl")
    p.add_argument("--from-date", default=None, help="Inclusive YYYY-MM-DD filter on log timestamp")
    p.add_argument("--to-date", default=None, help="Inclusive YYYY-MM-DD filter on log timestamp")
    p.add_argument("--positions", default="ipo_positions.csv", help="Optional positions CSV for outcomes")
    p.add_argument("--pending-file", default="listing_pending_breakouts.json", help="Current pending state file")
    p.add_argument("--json", dest="json_out", default=None, help="Write full report as JSON to this path")
    args = p.parse_args()

    rows = load_listing_day_rows(args.logs_root)
    rows = filter_by_date(rows, args.from_date, args.to_date)

    relevant = [r for r in rows if r.get("action") in CONFIRM_ACTIONS]
    by_action = Counter(r["action"] for r in relevant)

    rejected = [r for r in relevant if r.get("action") == "PENDING_REJECTED"]
    reject_reasons = Counter(
        (r.get("details") or {}).get("reason", "unknown") for r in rejected
    )

    confirmed_syms = {r.get("symbol", "").upper() for r in relevant if r.get("action") == "PENDING_CONFIRMED"}
    signal_syms = {r.get("symbol", "").upper() for r in relevant if r.get("action") == "BREAKOUT_SIGNAL"}

    started_events = [r for r in relevant if r.get("action") == "PENDING_STARTED"]
    started_by_sym: dict[str, list[dict]] = defaultdict(list)
    for r in started_events:
        started_by_sym[(r.get("symbol") or "").upper()].append(r)

    positions = load_positions(args.positions)
    outcome_rows: list[dict] = []
    for sym in sorted(confirmed_syms):
        pos = positions.get(sym)
        if not pos:
            continue
        try:
            pnl = float(pos.get("pnl_pct") or 0)
        except (TypeError, ValueError):
            pnl = None
        outcome_rows.append(
            {
                "symbol": sym,
                "status": pos.get("status"),
                "pnl_pct": pnl,
                "grade": pos.get("grade"),
                "had_breakout_log": sym in signal_syms,
            }
        )

    pending_count, pending_symbols = load_pending_file(args.pending_file)

    funnel_confirm_rate = None
    if by_action["PENDING_STARTED"]:
        funnel_confirm_rate = round(
            100.0 * by_action["PENDING_CONFIRMED"] / by_action["PENDING_STARTED"], 2
        )

    report = {
        "logs_root": os.path.abspath(args.logs_root),
        "date_filter": {"from": args.from_date, "to": args.to_date},
        "counts": dict(by_action),
        "reject_reasons": dict(reject_reasons),
        "unique_symbols": {
            "pending_started": len(started_by_sym),
            "pending_confirmed": len(confirmed_syms),
            "breakout_signal_logged": len(signal_syms),
            "confirmed_also_signal": len(confirmed_syms & signal_syms),
        },
        "funnel_confirm_pct_of_starts": funnel_confirm_rate,
        "current_pending_file": {
            "path": args.pending_file,
            "count": pending_count,
            "symbols": pending_symbols,
        },
        "positions_join": {
            "positions_file": args.positions,
            "confirmed_symbols_with_position_row": len(outcome_rows),
            "outcomes": outcome_rows,
        },
    }

    if outcome_rows:
        closed = [o for o in outcome_rows if (o.get("status") or "").upper() == "CLOSED"]
        if closed:
            wins = sum(1 for o in closed if (o.get("pnl_pct") or 0) > 0)
            report["positions_join"]["closed_after_confirm"] = len(closed)
            report["positions_join"]["closed_win_rate_pct"] = round(
                100.0 * wins / len(closed), 2
            )

    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, default=str)
        print(f"Wrote {args.json_out}")

    print("Listing confirmation quality (listing_day.jsonl)")
    print(f"  Logs: {report['logs_root']}")
    if args.from_date or args.to_date:
        print(f"  Date filter: {args.from_date or '…'} .. {args.to_date or '…'}")
    print("  Event counts:")
    for a in sorted(CONFIRM_ACTIONS):
        print(f"    {a}: {by_action.get(a, 0)}")
    print("  Reject reasons (PENDING_REJECTED):")
    for reason, n in reject_reasons.most_common():
        print(f"    {reason}: {n}")
    print("  Unique symbols:")
    for k, v in report["unique_symbols"].items():
        print(f"    {k}: {v}")
    if funnel_confirm_rate is not None:
        print(f"  Confirm / start (event ratio): {funnel_confirm_rate}%")
    print(f"  Current pending (file): {pending_count} {pending_symbols[:12]}{'…' if len(pending_symbols) > 12 else ''}")
    pj = report["positions_join"]
    print(f"  Positions join: {pj['confirmed_symbols_with_position_row']} rows for confirmed symbols")
    if "closed_win_rate_pct" in pj:
        print(f"    Closed win rate (among joined): {pj['closed_win_rate_pct']}% ({pj['closed_after_confirm']} closed)")


if __name__ == "__main__":
    main()
