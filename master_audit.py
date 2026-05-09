#!/usr/bin/env python3
"""
master_audit.py -- IPO-Base-Scanner System Integrity Audit
===========================================================
Run daily or weekly to verify system health across three layers:

  Section 1: Database Integrity
  Section 2: Telemetry / Log Quality
  Section 3: Strategy Consistency

Usage:
  python master_audit.py             # Full audit, human-readable output
  python master_audit.py --json      # Full audit, JSON output (for CI)
  python master_audit.py --section 1|2|3  # Run a single section

Exit codes:
  0 = PASS  (no issues found)
  1 = WARN  (review recommended)
  2 = FAIL  (action required)
"""

import os
import sys
import json
import argparse
import re
from datetime import datetime, timezone, timedelta
from collections import Counter

from dotenv import load_dotenv
from pymongo import MongoClient

load_dotenv()

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
# Keep this in sync with streamlined_ipo_scanner.py SCANNER_VERSION.
# Section 3 will flag any drift automatically.
EXPECTED_VERSION = "2.4.1"

NSE_HOLIDAYS_2025_2026 = {
    "2025-01-26", "2025-03-14", "2025-04-14", "2025-04-18",
    "2025-04-21", "2025-05-01", "2025-08-15", "2025-10-02",
    "2025-10-24", "2025-11-05", "2025-11-20", "2025-12-25",
    "2026-01-26", "2026-02-26", "2026-03-20", "2026-04-02",
    "2026-04-03", "2026-04-06", "2026-04-14", "2026-04-30",
    "2026-05-01", "2026-06-19", "2026-08-15", "2026-09-29",
    "2026-10-02", "2026-10-22", "2026-11-04", "2026-11-24",
    "2026-12-25",
}

MAX_REALISTIC_PNL_PCT    = 150.0
MIN_REALISTIC_PNL_PCT    = -60.0
MAX_RUNUP_REALISTIC      = 200.0
MAX_ENTRY_ABOVE_BKT_PCT  = 8.0


# ---------------------------------------------------------------------------
# DB helper
# ---------------------------------------------------------------------------
def get_db():
    uri = os.getenv("MONGO_URI")
    if not uri:
        raise RuntimeError("MONGO_URI not set in environment.")
    return MongoClient(uri)["ipo_scanner_v2"]


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------
class AuditResult:
    def __init__(self):
        self.errors   = []
        self.warnings = []
        self.info     = []

    def error(self, msg): self.errors.append(str(msg))
    def warn(self,  msg): self.warnings.append(str(msg))
    def ok(self,    msg): self.info.append(str(msg))

    @property
    def exit_code(self):
        if self.errors:   return 2
        if self.warnings: return 1
        return 0

    def status_label(self):
        return "PASS" if self.exit_code == 0 else ("WARN" if self.exit_code == 1 else "FAIL")

    def print_report(self, section_name):
        print("\n" + "=" * 70)
        print("  " + section_name)
        print("-" * 70)
        for msg in self.info:
            print("  [OK]   " + msg)
        for msg in self.warnings:
            print("  [WARN] " + msg)
        for msg in self.errors:
            print("  [ERR]  " + msg)
        print("\n  [%s]  %d errors, %d warnings\n" % (
            self.status_label(), len(self.errors), len(self.warnings)))

    def to_dict(self, section_name):
        return {
            "section":  section_name,
            "status":   self.status_label(),
            "errors":   self.errors,
            "warnings": self.warnings,
            "info":     self.info,
        }


# ===========================================================================
# SECTION 1: DATABASE INTEGRITY
# ===========================================================================
def audit_database_integrity(db):
    r = AuditResult()

    signals   = list(db.signals.find({},   {"_id": 0}))
    positions = list(db.positions.find({}, {"_id": 0}))

    # NOTE: signals can have status 'ACTIVE', 'CLOSED', or 'WATCH'.
    # Only 'ACTIVE' signals are expected to have matching positions.
    active_sig_syms = {s["symbol"] for s in signals if s.get("status") == "ACTIVE"}
    active_position_syms = {p["symbol"] for p in positions if p.get("status") == "ACTIVE"}
    all_position_syms   = {p["symbol"]: p.get("status") for p in positions}

    # 1a. Orphan ACTIVE signals -- ACTIVE signal but NO position record at all
    orphans_no_pos = active_sig_syms - set(all_position_syms.keys())
    if orphans_no_pos:
        r.error("ACTIVE signals with NO position record (crash mid-write?): %s" % sorted(orphans_no_pos))
    else:
        r.ok("All ACTIVE signals have at least one position record.")

    # 1b. ACTIVE signals where the position is CLOSED (signal status not synced after exit)
    sig_active_pos_closed = {
        sym for sym in active_sig_syms
        if all_position_syms.get(sym) == "CLOSED"
    }
    if sig_active_pos_closed:
        r.warn("Signal=ACTIVE but Position=CLOSED (exit not reflected in signals): %s\n"
               "       Run a sync script or close_signal_in_db() for these." % sorted(sig_active_pos_closed))
    else:
        r.ok("All ACTIVE signals have a matching ACTIVE position.")

    # 1b2. Position=ACTIVE but signal is CLOSED (backfill created position, signal later closed)
    pos_active_sig_closed = {
        sym for sym in active_position_syms
        if sym not in active_sig_syms and any(
            s.get("status") == "CLOSED" for s in signals if s.get("symbol") == sym
        )
    }
    if pos_active_sig_closed:
        r.warn("Position=ACTIVE but signal is CLOSED (position status not synced): %s" % sorted(pos_active_sig_closed))
    else:
        r.ok("No active positions with a closed signal (sync OK).")

    # 1c. Inverted stop-loss (exclude WATCH -- they have entry=stop=0 by design)
    bad = [s["symbol"] for s in signals
           if s.get("stop_loss", 0) >= s.get("entry_price", 1)
           and s.get("status") != "WATCH"]
    if bad:
        r.error("Signals with stop_loss >= entry_price (inverted): %s" % bad)
    else:
        r.ok("No inverted stop-loss values detected.")

    # 1d. Inverted target (exclude WATCH -- they have entry=target=0 by design)
    bad = [s["symbol"] for s in signals
           if s.get("target_price", 0) <= s.get("entry_price", 1)
           and s.get("status") != "WATCH"]
    if bad:
        r.error("Signals with target_price <= entry_price (inverted): %s" % bad)
    else:
        r.ok("No inverted target prices detected.")

    # 1e. Zero / negative entry price (exclude WATCH -- they intentionally have entry=0)
    bad = [s["symbol"] for s in signals
           if s.get("entry_price", 0) <= 0 and s.get("status") != "WATCH"]
    if bad:
        r.error("Signals with entry_price <= 0: %s" % bad)
    else:
        r.ok("All signals have a positive entry price.")

    # 1f. Duplicate signal_ids (WATCH signals accumulate one per day -- signal_ids should still be unique)
    ids     = [s.get("signal_id") for s in signals if s.get("signal_id")]
    dup_ids = [sid for sid, cnt in Counter(ids).items() if cnt > 1]
    if dup_ids:
        r.error("Duplicate signal_ids detected: %s" % dup_ids)
    else:
        watch_cnt  = sum(1 for s in signals if s.get("status") == "WATCH")
        trade_cnt  = sum(1 for s in signals if s.get("status") in ("ACTIVE", "CLOSED"))
        r.ok("All %d signal_ids are unique (%d trade signals, %d watchlist entries)." % (
            len(ids), trade_cnt, watch_cnt))

    # 1g. Unrealistic PnL on closed positions
    for p in positions:
        pnl = p.get("pnl_pct", 0)
        sym = p.get("symbol", "?")
        if p.get("status") == "CLOSED" and (
                pnl > MAX_REALISTIC_PNL_PCT or pnl < MIN_REALISTIC_PNL_PCT):
            r.warn("Position %s: pnl_pct=%.1f%% is outside realistic range "
                   "[%.0f%%, %.0f%%]. Check manually." % (
                       sym, pnl, MIN_REALISTIC_PNL_PCT, MAX_REALISTIC_PNL_PCT))

    # 1h. Entries on NSE holidays
    for p in positions:
        sym = p.get("symbol", "?")
        ed  = p.get("entry_date", "")
        if isinstance(ed, datetime):
            ed = ed.strftime("%Y-%m-%d")
        elif isinstance(ed, str):
            ed = ed[:10]
        if ed in NSE_HOLIDAYS_2025_2026:
            r.warn("Position %s: entry_date=%s is an NSE holiday. Verify." % (sym, ed))

    r.ok("Scanned %d signals and %d positions." % (len(signals), len(positions)))
    return r


# ===========================================================================
# SECTION 2: TELEMETRY / LOG QUALITY
# ===========================================================================
def audit_log_quality(db):
    r = AuditResult()
    today       = datetime.now(timezone.utc).date()
    today_start = datetime(today.year, today.month, today.day, tzinfo=timezone.utc)
    two_days_ago = today_start - timedelta(days=3)

    # 2a. SCAN_COMPLETED events today
    for scanner in ("consolidation", "listing_day", "positions", "watchlist"):
        count = db.logs.count_documents({
            "scanner": scanner,
            "action":  "SCAN_COMPLETED",
            "timestamp": {"$gte": today_start},
        })
        if scanner in ("consolidation", "listing_day") and count == 0:
            r.warn("No SCAN_COMPLETED log today for scanner='%s'. Did it run?" % scanner)
        elif count > 0:
            r.ok("'%s' SCAN_COMPLETED: %d event(s) today." % (scanner, count))

    # 2b. Rejection ratio today
    for scanner in ("consolidation", "listing_day"):
        total    = db.logs.count_documents({"scanner": scanner, "timestamp": {"$gte": today_start}})
        rejected = db.logs.count_documents({"scanner": scanner, "log_type": "REJECTED",
                                            "timestamp": {"$gte": today_start}})
        if total == 0:
            r.warn("'%s': No log entries at all today -- scanner may not have run." % scanner)
            continue
        ratio = rejected / total * 100
        if ratio > 97:
            r.warn("'%s' rejection ratio: %.1f%% (>97%%). "
                   "Filters may be too aggressive or data feed issue." % (scanner, ratio))
        elif ratio < 30:
            r.warn("'%s' rejection ratio: %.1f%% (<30%%). "
                   "Unusually few rejections -- verify scanner ran correctly." % (scanner, ratio))
        else:
            r.ok("'%s' rejection ratio: %.1f%% (%d/%d). Normal range." % (
                scanner, ratio, rejected, total))

    # 2c. Missing required fields in logs
    bad_docs = list(db.logs.find({
        "$or": [
            {"symbol":    {"$exists": False}},
            {"action":    {"$exists": False}},
            {"timestamp": {"$exists": False}},
        ]
    }, {"_id": 0, "log_id": 1, "scanner": 1}).limit(10))
    if bad_docs:
        r.error("Log documents missing required fields: %s" %
                [d.get("log_id", "?") for d in bad_docs])
    else:
        r.ok("All sampled log documents contain required fields.")

    # 2d. Version drift in today's logs
    wrong_ver = db.logs.count_documents({
        "timestamp": {"$gte": today_start},
        "version":   {"$ne": EXPECTED_VERSION},
    })
    if wrong_ver > 0:
        r.warn("%d log(s) today written with version != '%s'. "
               "Possible stale worker or partial deployment." % (wrong_ver, EXPECTED_VERSION))
    else:
        r.ok("All today's logs carry version='%s'." % EXPECTED_VERSION)

    # 2e. DAILY_SNAPSHOT coverage for active positions
    active_syms = [p["symbol"] for p in
                   db.positions.find({"status": "ACTIVE"}, {"symbol": 1, "_id": 0})]
    if active_syms:
        snapshotted = {
            doc["symbol"]
            for doc in db.logs.find({
                "action":    "DAILY_SNAPSHOT",
                "timestamp": {"$gte": today_start},
                "symbol":    {"$in": active_syms},
            }, {"symbol": 1, "_id": 0})
        }
        missing = set(active_syms) - snapshotted
        if missing:
            r.warn("Active positions missing DAILY_SNAPSHOT today: %s. "
                   "MTM tracker may not have run." % sorted(missing))
        else:
            r.ok("All %d active positions have a DAILY_SNAPSHOT today." % len(active_syms))

        for sym in active_syms:
            last = db.logs.find_one(
                {"action": "DAILY_SNAPSHOT", "symbol": sym},
                sort=[("timestamp", -1)]
            )
            if last:
                ts = last["timestamp"]
                if ts.tzinfo is None:
                    ts = ts.replace(tzinfo=timezone.utc)
                if ts < two_days_ago:
                    r.warn("Position %s: last DAILY_SNAPSHOT was %s (>2 business days ago)." % (
                        sym, last["timestamp"].strftime("%Y-%m-%d")))
    else:
        r.ok("No active positions -- DAILY_SNAPSHOT check skipped.")

    return r


# ===========================================================================
# SECTION 3: STRATEGY CONSISTENCY
# ===========================================================================
def audit_strategy_consistency(db):
    r = AuditResult()
    base_dir = os.path.dirname(os.path.abspath(__file__))

    def extract_ver(filepath, pattern):
        try:
            with open(filepath, encoding="utf-8") as f:
                content = f.read()
            m = re.search(pattern, content)
            return m.group(1) if m else None
        except FileNotFoundError:
            return None

    # 3a. Version consistency across files
    versions = {
        "streamlined_ipo_scanner.py": extract_ver(
            os.path.join(base_dir, "streamlined_ipo_scanner.py"),
            r'SCANNER_VERSION\s*=\s*["\']([^"\']+)["\']'),
        "db.py": extract_ver(
            os.path.join(base_dir, "db.py"),
            r'SCANNER_VERSION\s*=\s*["\']([^"\']+)["\']'),
        "README badge": extract_ver(
            os.path.join(base_dir, "README.md"),
            r'badge/version-([0-9.]+)-orange'),
        "README footer": extract_ver(
            os.path.join(base_dir, "README.md"),
            r'systematic IPO momentum trading \| v([0-9]+\.[0-9]+\.[0-9]+) \|'),
    }

    drift = [(name, ver) for name, ver in versions.items()
             if ver is not None and ver != EXPECTED_VERSION]
    if drift:
        for name, ver in drift:
            r.error("Version drift: %s has '%s', expected '%s'." % (
                name, ver, EXPECTED_VERSION))
    else:
        r.ok("All version strings match '%s' across scanner, db.py, and README." % EXPECTED_VERSION)

    # 3b. V2 signals with missing sector
    missing_sector = db.signals_v2.count_documents({
        "$or": [
            {"sector": {"$in": ["Unknown", None, ""]}},
            {"sector": {"$exists": False}},
        ]
    })
    if missing_sector:
        r.warn("%d V2 signal(s) have sector='Unknown' or missing. "
               "Re-run backfill_v2_from_v1.py to enrich these." % missing_sector)
    else:
        r.ok("All V2 signals have sector populated.")

    # 3c. V2 signals with null nifty_trend_slope
    missing_slope = db.signals_v2.count_documents({
        "$or": [
            {"market_context.nifty_trend_slope": {"$exists": False}},
            {"market_context.nifty_trend_slope": None},
        ]
    })
    if missing_slope:
        r.warn("%d V2 signal(s) have null nifty_trend_slope. "
               "Point-in-time enrichment may be incomplete." % missing_slope)
    else:
        r.ok("All V2 signals have nifty_trend_slope populated.")

    # 3d. Unrealistic max_runup in V2 outcomes
    unrealistic = list(db.signals_v2.find(
        {"outcome.max_runup_pct": {"$gt": MAX_RUNUP_REALISTIC}},
        {"symbol": 1, "outcome.max_runup_pct": 1, "_id": 0}
    ))
    if unrealistic:
        r.warn("%d V2 signal(s) with max_runup_pct >%.0f%% (possible data error): %s" % (
            len(unrealistic), MAX_RUNUP_REALISTIC,
            [(d["symbol"], d["outcome"]["max_runup_pct"]) for d in unrealistic]))
    else:
        r.ok("No V2 signals with unrealistic runup (>%.0f%%)." % MAX_RUNUP_REALISTIC)

    # 3e. Entry price vs breakout level -- validates the MAX_ENTRY_ABOVE_BREAKOUT_PCT guard
    v1_sigs = list(db.signals.find(
        {"breakout_level": {"$gt": 0}, "entry_price": {"$gt": 0}},
        {"symbol": 1, "entry_price": 1, "breakout_level": 1, "_id": 0}
    ))
    too_extended = [
        s for s in v1_sigs
        if (s["entry_price"] / s["breakout_level"] - 1) * 100 > MAX_ENTRY_ABOVE_BKT_PCT
    ]
    if too_extended:
        r.warn("%d signal(s) have entry >%.0f%% above breakout level "
               "(guard may not have fired): %s" % (
                   len(too_extended), MAX_ENTRY_ABOVE_BKT_PCT,
                   [s["symbol"] for s in too_extended[:5]]))
    else:
        r.ok("All signals have entry within %.0f%% of breakout level." % MAX_ENTRY_ABOVE_BKT_PCT)

    # 3f. Legacy ACTIVE signals missing entry_note (pre-v2.4.1 -- expected)
    missing_note = db.signals.count_documents({
        "status": "ACTIVE", "entry_note": {"$exists": False}
    })
    if missing_note:
        r.ok("Note: %d ACTIVE signal(s) lack 'entry_note' (pre-v2.4.1 records -- expected)." % missing_note)

    return r


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="IPO-Base-Scanner master audit")
    parser.add_argument("--json",    action="store_true", help="Output as JSON")
    parser.add_argument("--section", type=int, choices=[1, 2, 3],
                        help="Run only one section")
    args = parser.parse_args()

    print("\n" + "=" * 70)
    print("  IPO-Base-Scanner -- Master System Audit")
    print("  Run at: %s IST" % datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print("  Expected version: %s" % EXPECTED_VERSION)
    print("=" * 70)

    try:
        db = get_db()
    except RuntimeError as e:
        print("\n[ERR] Cannot connect to MongoDB: %s" % e)
        sys.exit(2)

    sections = {
        1: ("Section 1: Database Integrity",      lambda: audit_database_integrity(db)),
        2: ("Section 2: Telemetry / Log Quality", lambda: audit_log_quality(db)),
        3: ("Section 3: Strategy Consistency",    lambda: audit_strategy_consistency(db)),
    }

    run_nums = [args.section] if args.section else [1, 2, 3]

    results   = {}
    worst_ext = 0
    for num in run_nums:
        name, fn = sections[num]
        result   = fn()
        results[num] = (name, result)
        if not args.json:
            result.print_report(name)
        if result.exit_code > worst_ext:
            worst_ext = result.exit_code

    if args.json:
        output = {
            "audit_time":       datetime.now().isoformat(),
            "expected_version": EXPECTED_VERSION,
            "sections":         [res.to_dict(name) for name, res in results.values()],
            "overall_status":   "PASS" if worst_ext == 0 else ("WARN" if worst_ext == 1 else "FAIL"),
        }
        print(json.dumps(output, indent=2))
    else:
        status = "PASS" if worst_ext == 0 else ("WARN" if worst_ext == 1 else "FAIL")
        print("\n" + "=" * 70)
        print("  Overall: [%s]" % status)
        print("=" * 70 + "\n")

    sys.exit(worst_ext)


if __name__ == "__main__":
    main()
