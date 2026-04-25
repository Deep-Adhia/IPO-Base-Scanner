"""
backfill_instrument_keys.py
One-off script to migrate existing Upstox instrument mappings from CSV to MongoDB.
"""
import os
import pandas as pd
import logging
from db import upsert_instrument_key, ensure_indexes
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)

MAPPING_CSV = "ipo_upstox_mapping.csv"

def backfill():
    if not os.path.exists(MAPPING_CSV):
        logger.error(f"{MAPPING_CSV} not found. Nothing to backfill.")
        return

    logger.info(f"Reading mappings from {MAPPING_CSV}...")
    df = pd.read_csv(MAPPING_CSV, encoding='utf-8')
    
    logger.info("Ensuring indexes in MongoDB...")
    ensure_indexes()
    
    count = 0
    for _, row in df.iterrows():
        try:
            # Reconstruct ISIN if possible from instrument_key (format: NSE_EQ|INE...)
            key = row['instrument_key']
            isin = key.split('|')[-1] if '|' in key else None
            
            upsert_instrument_key(
                ipo_symbol=row['ipo_symbol'],
                instrument_key=key,
                isin=isin,
                name=row.get('name', row['ipo_symbol']),
                match_type=row.get('match_type', 'exact')
            )
            count += 1
        except Exception as e:
            logger.error(f"Failed to backfill {row['ipo_symbol']}: {e}")

    logger.info(f"Successfully backfilled {count} mappings to MongoDB.")

if __name__ == "__main__":
    backfill()
