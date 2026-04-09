#!/usr/bin/env python3
"""
Combine scanner logs into one analysis-ready file.

Usage examples:
  python combine_logs_for_ai.py
  python combine_logs_for_ai.py --scanner listing_day
  python combine_logs_for_ai.py --start-date 2026-03-01 --end-date 2026-03-31
  python combine_logs_for_ai.py --output combined_logs_march.jsonl --format jsonl
  python combine_logs_for_ai.py --format txt
"""

import argparse
import json
import os
from datetime import datetime
from typing import Dict, List, Optional


def parse_date(s: str) -> datetime.date:
    return datetime.strptime(s, "%Y-%m-%d").date()


def list_log_files(logs_root: str) -> List[str]:
    files: List[str] = []
    if not os.path.isdir(logs_root):
        return files

    for day_dir in sorted(os.listdir(logs_root)):
        day_path = os.path.join(logs_root, day_dir)
        if not os.path.isdir(day_path):
            continue
        for name in sorted(os.listdir(day_path)):
            if name.endswith(".jsonl"):
                files.append(os.path.join(day_path, name))
    return files


def parse_jsonl_file(path: str) -> List[Dict]:
    rows: List[Dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            raw = line.strip()
            if not raw:
                continue
            try:
                obj = json.loads(raw)
                obj["_source_file"] = path
                obj["_source_line"] = line_no
                rows.append(obj)
            except json.JSONDecodeError:
                # Keep malformed line as a structured event instead of dropping it.
                rows.append(
                    {
                        "timestamp": "",
                        "version": "",
                        "scanner": "unknown",
                        "symbol": "",
                        "action": "MALFORMED_LOG_LINE",
                        "details": {"raw": raw},
                        "_source_file": path,
                        "_source_line": line_no,
                    }
                )
    return rows


def file_date_from_path(path: str) -> Optional[datetime.date]:
    # Expected path: logs/YYYY-MM-DD/<file>.jsonl
    parts = path.replace("\\", "/").split("/")
    if len(parts) < 2:
        return None
    maybe_date = parts[-2]
    try:
        return parse_date(maybe_date)
    except ValueError:
        return None


def filter_rows(
    rows: List[Dict],
    scanner: Optional[str],
    start_date: Optional[datetime.date],
    end_date: Optional[datetime.date],
) -> List[Dict]:
    out: List[Dict] = []
    for r in rows:
        if scanner and r.get("scanner") != scanner:
            continue

        row_date = file_date_from_path(r.get("_source_file", ""))
        if start_date and row_date and row_date < start_date:
            continue
        if end_date and row_date and row_date > end_date:
            continue
        out.append(r)
    return out


def format_txt_row(r: Dict) -> str:
    ts = r.get("timestamp", "")
    scanner = r.get("scanner", "")
    symbol = r.get("symbol", "")
    action = r.get("action", "")
    details = r.get("details", {})
    details_str = json.dumps(details, ensure_ascii=False)
    return f"[{ts}] scanner={scanner} symbol={symbol} action={action} details={details_str}"


def summarize(rows: List[Dict]) -> Dict:
    scanners: Dict[str, int] = {}
    actions: Dict[str, int] = {}
    symbols: Dict[str, int] = {}
    for r in rows:
        scanners[r.get("scanner", "unknown")] = scanners.get(r.get("scanner", "unknown"), 0) + 1
        actions[r.get("action", "UNKNOWN")] = actions.get(r.get("action", "UNKNOWN"), 0) + 1
        sym = r.get("symbol", "")
        if sym:
            symbols[sym] = symbols.get(sym, 0) + 1
    return {
        "total_rows": len(rows),
        "scanners": dict(sorted(scanners.items(), key=lambda x: (-x[1], x[0]))),
        "actions": dict(sorted(actions.items(), key=lambda x: (-x[1], x[0]))),
        "top_symbols": dict(sorted(symbols.items(), key=lambda x: (-x[1], x[0]))[:25]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Combine all scanner logs into one file for AI analysis.")
    parser.add_argument("--logs-root", default="logs", help="Root logs directory (default: logs)")
    parser.add_argument(
        "--output",
        default="combined_logs_for_ai.jsonl",
        help="Output file path (default: combined_logs_for_ai.jsonl)",
    )
    parser.add_argument(
        "--format",
        choices=["jsonl", "txt"],
        default="jsonl",
        help="Output format: jsonl (best for AI) or txt (default: jsonl)",
    )
    parser.add_argument(
        "--scanner",
        choices=["listing_day", "consolidation"],
        default=None,
        help="Only include one scanner type",
    )
    parser.add_argument("--start-date", default=None, help="Include logs from date YYYY-MM-DD")
    parser.add_argument("--end-date", default=None, help="Include logs through date YYYY-MM-DD")
    parser.add_argument(
        "--with-summary",
        action="store_true",
        help="Also write summary JSON file next to output",
    )
    args = parser.parse_args()

    start_date = parse_date(args.start_date) if args.start_date else None
    end_date = parse_date(args.end_date) if args.end_date else None

    files = list_log_files(args.logs_root)
    all_rows: List[Dict] = []
    for p in files:
        all_rows.extend(parse_jsonl_file(p))

    filtered = filter_rows(all_rows, args.scanner, start_date, end_date)

    with open(args.output, "w", encoding="utf-8") as out:
        if args.format == "jsonl":
            for row in filtered:
                out.write(json.dumps(row, ensure_ascii=False) + "\n")
        else:
            for row in filtered:
                out.write(format_txt_row(row) + "\n")

    print(f"Combined rows: {len(filtered)}")
    print(f"Output file: {args.output}")

    if args.with_summary:
        summary_path = args.output + ".summary.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summarize(filtered), f, ensure_ascii=False, indent=2)
        print(f"Summary file: {summary_path}")


if __name__ == "__main__":
    main()

