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

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

load_dotenv()

def reconstruct_outcomes():
    client = MongoClient(os.getenv("MONGO_URI"))
    db = client['ipo_scanner_v2']
    signals_col = db['signals_v2']
    
    # Get all backfilled signals
    signals = list(signals_col.find({"scanner_version": "backfill-1.0"}))
    logger.info(f"🧪 Starting Synthetic Outcome Reconstruction for {len(signals)} signals...")
    
    success_count = 0
    
    for sig in signals:
        symbol = sig.get('symbol')
        breakout_date = sig.get('breakout_date')
        entry_price = sig.get('entry_price')
        stop_price = sig.get('stop_price')
        target_price = sig.get('target_price')
        
        if not breakout_date or not entry_price:
            continue
            
        logger.info(f"📊 Analyzing trajectory for {symbol} starting {breakout_date.strftime('%Y-%m-%d')}...")
        
        # Fetch 90 days of data after breakout to catch "Super Winners"
        ticker = yf.Ticker(f"{symbol}.NS")
        start_date = breakout_date.strftime('%Y-%m-%d')
        end_date = (breakout_date + timedelta(days=90)).strftime('%Y-%m-%d')
        
        try:
            df = ticker.history(start=start_date, end=end_date)
            if df.empty:
                logger.warning(f"❌ No post-breakout data for {symbol}")
                continue
                
            # Calculate performance
            # 1. Did it hit stop loss first?
            lows = df['Low'].values
            highs = df['High'].values
            
            hit_stop = False
            hit_target = False
            max_high = entry_price
            min_low = entry_price
            
            for i in range(len(df)):
                curr_low = lows[i]
                curr_high = highs[i]
                
                max_high = max(max_high, curr_high)
                min_low = min(min_low, curr_low)
                
                if curr_low <= stop_price and not hit_target:
                    hit_stop = True
                    # We continue to see if it recovered, but for "Outcome" we track if stop hit first
                    
                if curr_high >= target_price:
                    hit_target = True

            max_runup_pct = ((max_high - entry_price) / entry_price * 100) if entry_price > 0 else 0
            max_drawdown_pct = ((min_low - entry_price) / entry_price * 100) if entry_price > 0 else 0
            
            # Determine synthetic status
            status = "FAILED"
            if hit_target or max_runup_pct > 15:
                status = "SUCCESS"
            elif hit_stop:
                status = "STOPPED_OUT"
            elif max_runup_pct > 5:
                status = "SCRATCH"

            # Update MongoDB
            signals_col.update_one(
                {"_id": sig["_id"]},
                {"$set": {
                    "outcome": {
                        "max_runup_pct": round(max_runup_pct, 2),
                        "max_drawdown_pct": round(max_drawdown_pct, 2),
                        "hit_target": hit_target,
                        "hit_stop": hit_stop,
                        "status": status,
                        "reconstructed_at": datetime.now(timezone.utc)
                    }
                }}
            )
            success_count += 1
            logger.info(f"✅ {symbol}: Max Runup {max_runup_pct:.1f}% | Status: {status}")
            
        except Exception as e:
            logger.error(f"💥 Error analyzing {symbol}: {e}")
            
        time.sleep(1) # Be kind to yfinance
        
    logger.info(f"✨ Reconstruction Complete! Processed {success_count} signals.")

if __name__ == "__main__":
    reconstruct_outcomes()
