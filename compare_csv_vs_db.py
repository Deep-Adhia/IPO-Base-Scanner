"""
compare_csv_vs_db.py — Phase 2 Dual-Write Validator

Run after each live scan to verify CSV and MongoDB are in sync.
Cutover condition: 3 consecutive trading days with zero mismatches and zero failures.

Usage:
    python compare_csv_vs_db.py
    python compare_csv_vs_db.py --today-logs-only
"""

import os
import sys
import json
import argparse
import pandas as pd
from datetime import datetime, timezone, timedelta
from pymongo import MongoClient
from dotenv import load_dotenv

load_dotenv()

MONGO_URI = os.getenv("MONGO_URI", "")
SIGNALS_CSV = "ipo_signals.csv"
POSITIONS_CSV = "ipo_positions.csv"
MAPPING_CSV = "ipo_upstox_mapping.csv"
LISTING_DATA_CSV = "ipo_listing_data.csv"
RECENT_IPO_CSV = "recent_ipo_symbols.csv"

IST = timezone(timedelta(hours=5, minutes=30))
today_ist = datetime.now(IST).strftime("%Y-%m-%d")

FIELDS_TO_SPOT_CHECK = ["entry_price", "stop_loss", "grade", "status", "signal_type"]

def normalize_signal_id(sid: str) -> str:
    """Apply the same normalization logic the backfill script used."""
    if not isinstance(sid, str):
        return sid
    parts = sid.split("_")
    # Legacy CONSOL: CONSOL_{SYM}_{DATE}_{...}_LIVE  -> CONSOL_{SYM}_{DATE}
    if sid.startswith("CONSOL_") and "_LIVE" in sid and len(parts) >= 3:
        return f"CONSOL_{parts[1]}_{parts[2]}"
    # Legacy LISTING with time suffix: LISTING_{SYM}_{DATE}_{HHMM} -> LISTING_{SYM}_{DATE}
    if sid.startswith("LISTING_") and len(parts) == 4 and len(parts[3]) == 4 and parts[3].isdigit():
        return f"LISTING_{parts[1]}_{parts[2]}"
    return sid

# ─── Connection ──────────────────────────────────────────────────────────────

client = MongoClient(MONGO_URI) if MONGO_URI else None
db = client["ipo_scanner_v2"] if client else None

if db is None:
    print("❌ MONGO_URI not set. Cannot compare.")
    sys.exit(1)

signals_col = db["signals"]
positions_col = db["positions"]
logs_col = db["logs"]
instrument_keys_col = db["instrument_keys"]
listing_data_col = db["listing_data"]
ipos_col = db["ipos"]

# ─── Helpers ─────────────────────────────────────────────────────────────────

def load_csv(path):
    if not os.path.exists(path):
        return pd.DataFrame()
    return pd.read_csv(path, encoding="utf-8")

def pass_fail(ok):
    return "✅ MATCH" if ok else "❌ MISMATCH"

# ─── Signal Validation ───────────────────────────────────────────────────────

def validate_signals():
    print("\n" + "-" * 40)
    print("  SIGNALS")
    print("-" * 40)
    
    csv_df = load_csv(SIGNALS_CSV)
    if csv_df.empty:
        print("  ⚠️  No CSV data found.")
        return 0

    csv_total = len(csv_df)
    csv_active = len(csv_df[csv_df.get("status", pd.Series()) == "ACTIVE"]) if "status" in csv_df.columns else "N/A"
    csv_ids = set(normalize_signal_id(s) for s in csv_df["signal_id"].dropna().tolist()) if "signal_id" in csv_df.columns else set()

    db_total = signals_col.count_documents({})
    db_active = signals_col.count_documents({"status": "ACTIVE"})
    db_ids = set(signals_col.distinct("signal_id"))

    total_ok = csv_total == db_total
    active_ok = csv_active == db_active
    ids_in_csv_not_db = csv_ids - db_ids
    ids_in_db_not_csv = db_ids - csv_ids
    ids_ok = len(ids_in_csv_not_db) == 0 and len(ids_in_db_not_csv) == 0

    print(f"  Total      : CSV={csv_total}  DB={db_total}  {pass_fail(total_ok)}")
    print(f"  Active     : CSV={csv_active}  DB={db_active}  {pass_fail(active_ok)}")
    print(f"  signal_ids : {pass_fail(ids_ok)}")
    
    mismatches = 0
    if not total_ok: mismatches += 1
    if not ids_ok: mismatches += 1

    # Field-level spot check
    raw_sample = list(csv_df["signal_id"].dropna())[:5]
    sample_ids = [normalize_signal_id(s) for s in raw_sample]
    for sid in sample_ids:
        rows = csv_df[csv_df["signal_id"].apply(normalize_signal_id) == sid]
        if rows.empty: continue
        csv_row = rows.iloc[0].to_dict()
        db_doc = signals_col.find_one({"signal_id": sid}, {"_id": 0})
        if db_doc:
            for field in FIELDS_TO_SPOT_CHECK:
                if str(csv_row.get(field)) != str(db_doc.get(field)):
                    mismatches += 1
                    break

    return mismatches

# ─── Positions Validation ────────────────────────────────────────────────────

def validate_positions():
    print("\n" + "-" * 40)
    print("  POSITIONS")
    print("-" * 40)

    csv_df = load_csv(POSITIONS_CSV)
    if csv_df.empty: return 0

    csv_symbols = set(csv_df.drop_duplicates(subset=["symbol"], keep="last")["symbol"].tolist())
    db_symbols = set(positions_col.distinct("symbol"))

    count_ok = len(csv_symbols) == len(db_symbols)
    symbols_ok = csv_symbols == db_symbols

    print(f"  Count      : CSV={len(csv_symbols)}  DB={len(db_symbols)}  {pass_fail(count_ok)}")
    print(f"  Symbols    : {pass_fail(symbols_ok)}")

    return 0 if (count_ok and symbols_ok) else 1

# ─── Metadata Validation ─────────────────────────────────────────────────────

def validate_metadata():
    print("\n" + "-" * 40)
    print("  METADATA & REGISTRIES")
    print("-" * 40)
    
    issues = 0
    
    # 1. Instrument Mapping
    csv_mapping = load_csv(MAPPING_CSV)
    if not csv_mapping.empty:
        csv_syms = set(csv_mapping["ipo_symbol"].tolist())
        db_syms = set(instrument_keys_col.distinct("ipo_symbol"))
        ok = csv_syms == db_syms
        print(f"  Mappings   : CSV={len(csv_syms)}  DB={len(db_syms)}  {pass_fail(ok)}")
        if not ok: issues += 1
        
    # 2. Listing Data
    csv_listing = load_csv(LISTING_DATA_CSV)
    if not csv_listing.empty:
        # Filter comments
        csv_listing = csv_listing[~csv_listing['symbol'].astype(str).str.startswith('#')]
        csv_listing = csv_listing[csv_listing['symbol'].notna()]
        csv_syms = set(csv_listing["symbol"].tolist())
        db_syms = set(listing_data_col.distinct("symbol"))
        ok = csv_syms == db_syms
        print(f"  ListingData: CSV={len(csv_syms)}  DB={len(db_syms)}  {pass_fail(ok)}")
        if not ok: issues += 1

    # 3. IPO Discovery
    csv_ipos = load_csv(RECENT_IPO_CSV)
    if not csv_ipos.empty:
        csv_syms = set(csv_ipos["symbol"].tolist())
        db_syms = set(ipos_col.distinct("symbol"))
        ok = csv_syms == db_syms
        print(f"  Discovery  : CSV={len(csv_syms)}  DB={len(db_syms)}  {pass_fail(ok)}")
        if not ok: issues += 1
        
    return issues

# ─── Logs Validation ─────────────────────────────────────────────────────────

def validate_logs(today_only=False):
    print("\n" + "-" * 40)
    print(f"  LOGS {'(today only)' if today_only else '(backfill total)'}")
    print("-" * 40)

    log_dir = os.path.join("logs", today_ist)
    csv_count = 0
    if os.path.exists(log_dir):
        for fname in os.listdir(log_dir):
            if fname.endswith(".jsonl"):
                with open(os.path.join(log_dir, fname), "r", encoding="utf-8") as f:
                    csv_count += sum(1 for line in f if line.strip())

    if today_only:
        today_utc_start = datetime.now(IST).replace(hour=0, minute=0, second=0, microsecond=0).astimezone(timezone.utc)
        db_count = logs_col.count_documents({"source": "live", "created_at": {"$gte": today_utc_start}})
    else:
        db_count = logs_col.count_documents({"source": "live"})

    count_ok = csv_count == db_count
    print(f"  Count      : JSONL={csv_count}  DB={db_count}  {pass_fail(count_ok)}")
    
    if csv_count > 0 and db_count > 0:
        # Spot check hashes for the first 5 entries in today's logs
        print("  Hash Check (sample of 5):")
        if os.path.exists(log_dir):
            checked = 0
            for fname in os.listdir(log_dir):
                if checked >= 5: break
                if fname.endswith(".jsonl"):
                    with open(os.path.join(log_dir, fname), "r", encoding="utf-8") as f:
                        for line in f:
                            if checked >= 5: break
                            entry = json.loads(line)
                            # Re-generate log_id for validation if not in entry (though it should be in DB)
                            # Actually, we can't easily re-gen without the logic from db.py
                            # But we can check if DB has a record for this symbol/action/time
                            from db import generate_log_id
                            # Extract fields
                            s = entry.get("symbol")
                            a = entry.get("action")
                            t = entry.get("timestamp")
                            v = entry.get("version", "1.0.0")
                            d = entry.get("details", {})
                            sc = entry.get("scanner", "unknown")
                            
                            lid = generate_log_id(sc, s, a, t, v, d)
                            match = logs_col.find_one({"log_id": lid})
                            print(f"    {'✅' if match else '❌'} {s}/{a} at {t}")
                            if not match: count_ok = False
                            checked += 1

    return 0 if count_ok else 1

# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--today-logs-only", action="store_true", help="Compare only today's log entries")
    args = parser.parse_args()

    print(f"\n{'='*45}")
    print(f"  CSV vs MongoDB Validator — {today_ist}")
    print(f"{'='*45}")

    total_issues = 0
    total_issues += validate_signals()
    total_issues += validate_positions()
    total_issues += validate_metadata()
    total_issues += validate_logs(args.today_logs_only)

    print(f"\n{'='*45}")
    if total_issues == 0:
        print(f"  ✅ ALL CHECKS PASSED — Zero mismatches")
    else:
        print(f"  ❌ {total_issues} issue(s) found — Do NOT proceed to Phase 3")
    print(f"{'='*45}\n")
    sys.exit(0 if total_issues == 0 else 1)

if __name__ == "__main__":
    main()
