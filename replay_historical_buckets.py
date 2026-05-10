"""
replay_historical_buckets.py
Applies the new 3-bucket logic to historical signals in MongoDB.
Outputs the distribution of ALIGNED vs EXTENDED vs BROKEN.
"""
import os
import pandas as pd
from pymongo import MongoClient
from dotenv import load_dotenv

load_dotenv()

# Constants
BUCKET_ALIGNED  = "ALIGNED"
BUCKET_EXTENDED = "EXTENDED"
BUCKET_BROKEN   = "BROKEN"

def categorize_signal_bucket(metrics: dict, days_since_listing: int) -> str:
    prng = metrics.get("prng", 0)
    vol_ratio = metrics.get("vol_ratio", 0)
    
    if prng > 45.0 or days_since_listing > 750:
        return BUCKET_BROKEN

    if prng > 25.0 or vol_ratio < 1.2:
        return BUCKET_EXTENDED
    
    return BUCKET_ALIGNED

def replay():
    client = MongoClient(os.getenv("MONGO_URI"))
    db = client['ipo_scanner_v2']
    signals = list(db.signals.find())
    
    if not signals:
        print("No signals found in DB.")
        return

    df = pd.DataFrame(signals)
    results = []
    
    for _, row in df.iterrows():
        # Handle different schema versions
        details = row.get("details", {})
        snapshot = row.get("decision_snapshot", {})
        if not isinstance(snapshot, (dict, str)):
            snapshot = {}
        
        if isinstance(snapshot, str):
            import ast
            try: snapshot = ast.literal_eval(snapshot)
            except: snapshot = {}
        
        m_snapshot = snapshot.get("metrics_snapshot", {}) if isinstance(snapshot, dict) else {}
        
        prng = m_snapshot.get("prng", row.get("prng", details.get("prng", 0)))
        vol_ratio = m_snapshot.get("vol_ratio", row.get("vol_ratio", details.get("vol_ratio", 0)))
        
        metrics = {
            "prng": float(prng) if prng is not None else 0,
            "vol_ratio": float(vol_ratio) if vol_ratio is not None else 0
        }
        
        # Calculate IPO age (approximate from signal_date and listing_date)
        try:
            sd = pd.to_datetime(row.get("signal_date")).date()
            ld = pd.to_datetime(row.get("listing_date")).date()
            age = (sd - ld).days
        except:
            age = 0
            
        bucket = categorize_signal_bucket(metrics, age)
        results.append(bucket)

    df['bucket'] = results
    summary = df['bucket'].value_counts()
    
    print(f"\n{'='*40}")
    print(" HISTORICAL BUCKET REPLAY (n=714)")
    print(f"{'='*40}")
    print(summary)
    print(f"{'='*40}\n")

if __name__ == "__main__":
    replay()
