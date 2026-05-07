import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
from pymongo import MongoClient
from dotenv import load_dotenv
import yfinance as yf
import time
import logging

# Institutional Imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from core.repository import MongoRepository
from integration.signal_builder import SignalBuilder

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

load_dotenv()

def fetch_yfinance_history(symbol, signal_date):
    """Fetch history for enrichment around the signal date."""
    ticker_sym = f"{symbol}.NS"
    ticker = yf.Ticker(ticker_sym)
    
    # Fetch info for sector/industry (cached if possible)
    try:
        info = ticker.info
        sector = info.get('sector', 'Unknown')
        industry = info.get('industry', 'Unknown')
    except:
        sector, industry = 'Unknown', 'Unknown'

    # We need at least 30 days of history before the signal date for enrichment
    start_date = (signal_date - timedelta(days=60)).strftime('%Y-%m-%d')
    end_date = (signal_date + timedelta(days=5)).strftime('%Y-%m-%d')
    
    try:
        df = ticker.history(start=start_date, end=end_date, interval='1d')
        if df.empty:
            return None, sector, industry
        
        # Standardize columns for SignalBuilder
        df = df.reset_index()
        df = df.rename(columns={
            'Date': 'DATE',
            'Open': 'OPEN',
            'High': 'HIGH',
            'Low': 'LOW',
            'Close': 'CLOSE',
            'Volume': 'VOLUME'
        })
        # Remove timezone info for easier handling
        if hasattr(df['DATE'].dt, 'tz_localize'):
            df['DATE'] = df['DATE'].dt.tz_localize(None)
        
        return df, sector, industry
    except Exception as e:
        logger.error(f"Error fetching {symbol} from yfinance: {e}")
        return None, sector, industry

def backfill_v1_to_v2():
    client = MongoClient(os.getenv("MONGO_URI"))
    db = client['ipo_scanner_v2']
    v1_col = db['signals']
    
    repo = MongoRepository()
    builder = SignalBuilder()
    
    # Get all V1 signals (excluding watchlist)
    v1_signals = list(v1_col.find({"signal_type": {"$ne": "WATCHLIST"}}))
    logger.info(f"🚀 Starting backfill for {len(v1_signals)} signals...")
    
    success_count = 0
    fail_count = 0
    
    for v1 in v1_signals:
        symbol = v1.get('symbol')
        signal_date = v1.get('signal_date')
        
        if not signal_date:
            logger.warning(f"⏭️ Skipping {symbol}: No signal_date")
            continue
            
        # Ensure signal_date is datetime
        if isinstance(signal_date, str):
            signal_date = pd.to_datetime(signal_date)
            
        logger.info(f"🔍 Processing {symbol} ({signal_date.strftime('%Y-%m-%d')})...")
        
        # Check if already in V2
        # Use a similar deterministic ID logic to see if we should skip
        # (Though backfill IDs might differ, we can check by symbol and date)
        # For now, let's just proceed or check existence
        
        history, sector, industry = fetch_yfinance_history(symbol, signal_date)
        if history is None or history.empty:
            logger.warning(f"❌ Failed to get history for {symbol}")
            fail_count += 1
            continue
            
        # Find the closest candle to the signal date
        # Signal date might be after market hours, so we find the candle on or before it
        history['date_only'] = history['DATE'].dt.date
        target_date = signal_date.date()
        
        match = history[history['date_only'] == target_date]
        if match.empty:
            # Try finding the latest one before signal_date
            match = history[history['date_only'] < target_date]
            if match.empty:
                logger.warning(f"❌ No matching candle for {symbol} on {target_date}")
                fail_count += 1
                continue
            candle = match.iloc[-1]
        else:
            candle = match.iloc[0]
            
        try:
            # Map V1 fields to SignalBuilder payload
            raw_payload = {
                "symbol": symbol,
                "entry": v1.get("entry_price"),
                "stop": v1.get("stop_loss"),
                "target": v1.get("target_price"),
                "scanner": "listing_day" if v1.get("signal_type") == "LISTING_DAY_BREAKOUT" else "ipo_base",
                "sector": sector,
                "industry": industry,
                "log_id": str(v1.get("_id")),
                "metrics": {
                    "v1_grade": v1.get("grade"),
                    "v1_score": v1.get("score")
                }
            }
            
            # Add pnl and exit info to raw_payload so it's preserved in features if needed
            # But the SignalBuilder doesn't store outcome. 
            # We will handle outcomes separately or add them to the V2 doc manually.
            
            signal_v2 = builder.build_signal(
                raw_payload=raw_payload,
                candle=candle,
                history=history,
                base_candles=history, # Fallback
                scanner_version="backfill-1.0"
            )
            
            # Save to V2
            repo.save_signal(signal_v2)
            
            # Update the V2 doc with the outcome from V1
            db.signals_v2.update_one(
                {"signal_id": signal_v2.signal_id},
                {"$set": {
                    "outcome": {
                        "exit_price": v1.get("exit_price", 0),
                        "pnl_pct": v1.get("pnl_pct", 0),
                        "status": v1.get("status", "CLOSED"),
                        "days_held": v1.get("days_held", 0)
                    }
                }}
            )
            
            success_count += 1
            logger.info(f"✅ Backfilled {symbol} -> {signal_v2.signal_id}")
            
        except Exception as e:
            logger.error(f"💥 Failed to build signal for {symbol}: {e}")
            fail_count += 1
            
        # Rate limiting for yfinance
        time.sleep(1)
        
    logger.info(f"\n✨ Backfill Complete!")
    logger.info(f"Success: {success_count}")
    logger.info(f"Failed:  {fail_count}")

if __name__ == "__main__":
    import traceback
    try:
        backfill_v1_to_v2()
    except Exception:
        traceback.print_exc()
