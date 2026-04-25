import os
import json
import glob
import sys
import argparse
from datetime import datetime, timezone
import pandas as pd
from pymongo import MongoClient
import logging
from db import generate_log_id, ensure_indexes, make_utc

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)

MONGO_URI = os.getenv("MONGO_URI", "")
client = MongoClient(MONGO_URI) if MONGO_URI else None
db = client["ipo_scanner_v2"] if client else None

signals_col = db["signals"] if db is not None else None
positions_col = db["positions"] if db is not None else None
logs_col = db["logs"] if db is not None else None

def safe_date_parse(val):
    if pd.isna(val) or val == "":
        return None
    if isinstance(val, str):
        try:
            return datetime.fromisoformat(val.replace('Z', '+00:00'))
        except ValueError:
            try:
                return datetime.strptime(val, "%Y-%m-%d")
            except ValueError:
                return None
    return val

def run_logs_backfill(dry_run=False):
    logger.info("Starting Logs Backfill...")
    log_files = glob.glob("logs/**/*.jsonl", recursive=True)
    inserted = 0
    duplicates = 0
    failures = 0
    
    sample_docs = []
    
    for fpath in log_files:
        with open(fpath, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    data = json.loads(line)
                    ts_str = data.get("timestamp")
                    if not ts_str:
                        continue
                        
                    try:
                        clean_ts_str = ts_str.replace(" IST", "")
                        candle_ts = datetime.strptime(clean_ts_str, "%Y-%m-%d %H:%M:%S")
                    except ValueError:
                        candle_ts = datetime.now(timezone.utc)
                        
                    candle_ts = make_utc(candle_ts)
                        
                    details = data.get("details", {})
                    log_id = generate_log_id(
                        data.get("scanner", "unknown"),
                        data.get("symbol", "unknown"),
                        data.get("action", "unknown"),
                        candle_ts,
                        data.get("version", "unknown"),
                        details
                    )
                    
                    doc = {
                        "log_id": log_id,
                        "timestamp": datetime.now(timezone.utc), 
                        "candle_timestamp": candle_ts,
                        "symbol": data.get("symbol"),
                        "action": data.get("action"),
                        "scanner": data.get("scanner"),
                        "version": data.get("version"),
                        "source": "backfill",
                        "details": details,
                        "created_at": datetime.now(timezone.utc)
                    }
                    
                    if len(sample_docs) < 3:
                        sample_docs.append(doc)
                    
                    if not dry_run and logs_col is not None:
                        try:
                            logs_col.insert_one(doc)
                            inserted += 1
                        except Exception as e:
                            if "duplicate key error" in str(e).lower():
                                duplicates += 1
                            else:
                                logger.error(f"Error inserting log: {e}")
                                failures += 1
                    else:
                        inserted += 1 # simulate insert
                except Exception as e:
                    logger.error(f"Error parsing log line in {fpath}: {e}")
                    failures += 1
                    
    logger.info(f"Logs Backfill Complete. Inserted: {inserted}, Skipped Duplicates: {duplicates}, Failures: {failures}")
    return inserted, duplicates, failures, sample_docs

def run_signals_backfill(dry_run=False):
    logger.info("Starting Signals Backfill...")
    if not os.path.exists("ipo_signals.csv"):
        logger.warning("ipo_signals.csv not found")
        return 0, 0, 0, []
        
    df = pd.read_csv("ipo_signals.csv")
    inserted = 0
    duplicates = 0
    failures = 0
    total_rows = len(df)
    
    sample_docs = []
    
    for _, row in df.iterrows():
        try:
            doc = row.dropna().to_dict()
            
            if 'signal_date' in doc:
                doc['signal_date'] = make_utc(safe_date_parse(doc['signal_date']))
            if 'exit_date' in doc:
                doc['exit_date'] = make_utc(safe_date_parse(doc['exit_date']))
                
            doc["source"] = "backfill"
            doc["created_at"] = datetime.now(timezone.utc)
            doc["updated_at"] = datetime.now(timezone.utc)
            
            signal_id = doc.get("signal_id")
            if not signal_id:
                sym = doc.get("symbol", "UNK")
                sdate = doc.get("signal_date")
                sdate_str = sdate.strftime("%Y%m%d") if sdate else "NODATE"
                stype = doc.get("signal_type", "UNKNOWN")
                signal_id = f"{stype}_{sym}_{sdate_str}"
                doc["signal_id"] = signal_id
            else:
                if "CONSOL_" in signal_id and "_LIVE" in signal_id:
                    parts = signal_id.split("_")
                    if len(parts) >= 3:
                        signal_id = f"CONSOL_{parts[1]}_{parts[2]}"
                        doc["signal_id"] = signal_id
                elif "LISTING_" in signal_id:
                    parts = signal_id.split("_")
                    if len(parts) >= 3:
                        signal_id = f"LISTING_{parts[1]}_{parts[2]}"
                        doc["signal_id"] = signal_id
                        
            if len(sample_docs) < 3:
                sample_docs.append(doc)
                        
            if not dry_run and signals_col is not None:
                try:
                    res = signals_col.update_one(
                        {"signal_id": doc["signal_id"]},
                        {"$set": doc},
                        upsert=True
                    )
                    if res.upserted_id:
                        inserted += 1
                    else:
                        duplicates += 1
                except Exception as e:
                    logger.error(f"Error inserting signal {signal_id}: {e}")
                    failures += 1
            else:
                inserted += 1
        except Exception as e:
            logger.error(f"Error processing signal row: {e}")
            failures += 1
            
    logger.info(f"Signals Backfill Complete. Total Rows: {total_rows}, Upserted New: {inserted}, Updated/Skipped: {duplicates}, Failures: {failures}")
    return total_rows, inserted + duplicates, failures, sample_docs

def run_positions_backfill(dry_run=False):
    logger.info("Starting Positions Backfill...")
    if not os.path.exists("ipo_positions.csv"):
        logger.warning("ipo_positions.csv not found")
        return 0, 0, 0, []
        
    df = pd.read_csv("ipo_positions.csv")
    total_symbols = df["symbol"].nunique()
    inserted = 0
    failures = 0
    
    # We confirm that positions acts as a state table.
    latest_df = df.drop_duplicates(subset=['symbol'], keep='last')
    
    sample_docs = []
    
    for _, row in latest_df.iterrows():
        try:
            doc = row.dropna().to_dict()
            
            if 'entry_date' in doc:
                doc['entry_date'] = make_utc(safe_date_parse(doc['entry_date']))
            if 'exit_date' in doc:
                doc['exit_date'] = make_utc(safe_date_parse(doc['exit_date']))
                
            doc["source"] = "backfill"
            doc["updated_at"] = datetime.now(timezone.utc)
            
            symbol = doc.get("symbol")
            if not symbol:
                continue
                
            if len(sample_docs) < 3:
                sample_docs.append(doc)
                
            if not dry_run and positions_col is not None:
                try:
                    positions_col.update_one(
                        {"symbol": symbol},
                        {"$set": doc, "$setOnInsert": {"created_at": datetime.now(timezone.utc)}},
                        upsert=True
                    )
                    inserted += 1
                except Exception as e:
                    logger.error(f"Error processing position row: {e}")
                    failures += 1
            else:
                inserted += 1
        except Exception as e:
            logger.error(f"Error processing position row logic: {e}")
            failures += 1
            
    logger.info(f"Positions Backfill Complete. Unique Symbols: {total_symbols}, Upserted: {inserted}, Failures: {failures}")
    return total_symbols, inserted, failures, sample_docs

def print_samples(name, samples):
    logger.info(f"--- {name} Samples ---")
    for s in samples:
        # Simplify datetimes for printing
        safe_s = {k: (v.isoformat() if isinstance(v, datetime) else v) for k, v in s.items()}
        logger.info(json.dumps(safe_s, default=str))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="Parse CSVs but do not insert into DB")
    args = parser.parse_args()

    if args.dry_run:
        logger.info("=== DRY RUN MODE: No DB operations will occur ===")
    elif db is None:
        logger.error("❌ MONGO_URI not set. Use --dry-run or configure environment.")
        sys.exit(1)
    else:
        logger.info("Ensuring indexes...")
        ensure_indexes()
    
    logger.info("Executing Phase 1 Backfill...")
    l_ins, l_dup, l_fail, l_samp = run_logs_backfill(args.dry_run)
    sig_total, sig_processed, s_fail, s_samp = run_signals_backfill(args.dry_run)
    pos_total, pos_processed, p_fail, p_samp = run_positions_backfill(args.dry_run)
    
    logger.info("="*50)
    logger.info("VALIDATION REPORT")
    logger.info("="*50)
    
    print_samples("Signals", s_samp)
    print_samples("Positions", p_samp)
    print_samples("Logs", l_samp)
    
    if not args.dry_run and db is not None:
        db_signals = signals_col.count_documents({})
        db_unique_signals = len(signals_col.distinct("signal_id"))
        db_positions = positions_col.count_documents({})
        db_logs = logs_col.count_documents({"source": "backfill"})
        
        logger.info(f"Signals validation: CSV Rows = {sig_total}, Processed = {sig_processed}, DB Count = {db_signals}, Unique DB IDs = {db_unique_signals}")
        if sig_total != sig_processed or db_unique_signals != db_signals:
            logger.error("❌ Signals processing count mismatch or unique ID clash!")
        else:
            logger.info("✅ Signals processing count matches perfectly.")
            
        logger.info(f"Positions validation: CSV Unique Symbols = {pos_total}, DB Count = {db_positions}")
        logger.info(f"Logs backfilled: {db_logs} (Duplicates Skipped: {l_dup})")
    else:
        logger.info("Dry run completed. Validate the samples above.")

if __name__ == "__main__":
    main()
