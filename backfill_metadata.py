"""
backfill_metadata.py
Backfill script for ipos and listing_data collections.
"""
import os
import logging
from db import upsert_ipo, upsert_listing_data, ensure_indexes
from datetime import datetime
from dotenv import load_dotenv
import pandas as pd

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)

RECENT_IPO_CSV = "recent_ipo_symbols.csv"
LISTING_DATA_CSV = "ipo_listing_data.csv"

def backfill():
    ensure_indexes()
    
    # 1. Backfill IPO discovery records
    if os.path.exists(RECENT_IPO_CSV):
        logger.info(f"Backfilling IPOs from {RECENT_IPO_CSV}...")
        try:
            df_ipos = pd.read_csv(RECENT_IPO_CSV, encoding='utf-8')
            for _, row in df_ipos.iterrows():
                upsert_ipo(
                    symbol=row['symbol'],
                    listing_date=row.get('listing_date'),
                    name=row.get('company', row['symbol'])
                )
            logger.info(f"✅ Backfilled {len(df_ipos)} IPO records.")
        except Exception as e:
            logger.error(f"Error backfilling IPOs: {e}")
            
    # 2. Backfill listing day metrics
    if os.path.exists(LISTING_DATA_CSV):
        logger.info(f"Backfilling Listing Data from {LISTING_DATA_CSV}...")
        try:
            df_listing = pd.read_csv(LISTING_DATA_CSV, encoding='utf-8')
            # Remove comment rows
            df_listing = df_listing[~df_listing['symbol'].astype(str).str.startswith('#')]
            df_listing = df_listing[df_listing['symbol'].notna() & (df_listing['symbol'] != '')]
            
            for _, row in df_listing.iterrows():
                upsert_listing_data(row['symbol'], row.to_dict())
            logger.info(f"✅ Backfilled {len(df_listing)} listing data records.")
        except Exception as e:
            logger.error(f"Error backfilling listing data: {e}")

if __name__ == "__main__":
    backfill()
