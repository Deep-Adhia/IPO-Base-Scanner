import os
import sys
from datetime import datetime, timezone
from dotenv import load_dotenv

# Mock the logger so we don't crash
import logging
logging.basicConfig(level=logging.INFO)

# Add current and parent directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Load environment
load_dotenv()

def test_listing_day_telemetry():
    """Verify that Listing Day breakouts now flow into signals_v2."""
    from listing_day_breakout_scanner import save_breakout_signal
    
    import pandas as pd
    import numpy as np
    
    # Create mock history
    dates = pd.date_range(end=datetime.now(), periods=30)
    history = pd.DataFrame({
        'DATE': dates,
        'OPEN': np.random.uniform(90, 110, 30),
        'HIGH': np.random.uniform(90, 110, 30),
        'LOW': np.random.uniform(90, 110, 30),
        'CLOSE': np.random.uniform(90, 110, 30),
        'VOLUME': np.random.uniform(100000, 1000000, 30)
    })
    candle = history.iloc[-1]

    mock_breakout = {
        'symbol': 'TELEMETRY_TEST',
        'entry_price': 100.0,
        'stop_loss': 90.0,
        'target_price': 130.0,
        'listing_day_high': 95.0,
        'tier': 'Tier 1',
        'leader_score': 85,
        'has_volume_caution': False,
        'volume_ratio': 3.5,
        'ipo_age': 5,
        'candle_timestamp': datetime.now(timezone.utc),
        '_candle': candle,
        '_history': history,
        '_base_candles': history
    }
    
    print("Testing save_breakout_signal integration...")
    success = save_breakout_signal(mock_breakout)
    
    if success:
        print("\n[SUCCESS] Signal processed by scanner.")
        
        # Verify in MongoDB
        from pymongo import MongoClient
        client = MongoClient(os.getenv("MONGO_URI"))
        db = client["ipo_scanner_v2"]
        
        v2_doc = db.signals_v2.find_one({"symbol": "TELEMETRY_TEST"})
        if v2_doc:
            print("\n[VERIFIED] Rich telemetry found in signals_v2:")
            print(f"  - Scanner: {v2_doc.get('scanner')}")
            print(f"  - Features: {list(v2_doc.get('features', {}).keys())}")
            print(f"  - Market Context: {v2_doc.get('market_context')}")
            
            # Clean up (Commented out for verification)
            # db.signals_v2.delete_one({"symbol": "TELEMETRY_TEST"})
            # db.signals.delete_one({"symbol": "TELEMETRY_TEST"})
            # db.logs.delete_many({"symbol": "TELEMETRY_TEST"})
            # print("\n[CLEANED] Test data removed.")
            print("\n[KEEPING] Test data kept for verify_today_results.py")
        else:
            print("\n[FAILED] Signal did not reach signals_v2.")
    else:
        print("\n[FAILED] save_breakout_signal returned False.")

if __name__ == "__main__":
    test_listing_day_telemetry()
