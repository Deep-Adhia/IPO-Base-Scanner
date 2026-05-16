#!/usr/bin/env python3
"""
reset_all_data.py

Reset all MongoDB collections to initial state.
This script will:
- Clear all positions
- Clear all signals (or archive them in DB)
- Reset listing data
- Reset watchlist
- Reset instrument mappings (optional)
"""

import os
from datetime import datetime, timezone
import pandas as pd
from db import (
    signals_col, positions_col, listing_data_col, 
    watchlist_col, instrument_keys_col, logs_col
)

def archive_collection_to_csv(col, prefix):
    """Archive existing collection to a CSV file before resetting"""
    if col is None:
        return None
        
    try:
        docs = list(col.find({}, {"_id": 0}))
        if not docs:
            return None
            
        df = pd.DataFrame(docs)
        archive_name = f"{prefix}_archive_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        df.to_csv(archive_name, index=False, encoding='utf-8')
        print(f"✅ Archived {len(df)} records from {col.name} to {archive_name}")
        return archive_name
    except Exception as e:
        print(f"⚠️ Could not archive {prefix}: {e}")
    return None

def reset_collection(col, name):
    """Delete all documents in a collection"""
    if col is None:
        print(f"❌ {name} collection unavailable")
        return
        
    try:
        archive_collection_to_csv(col, name)
        result = col.delete_many({})
        print(f"✅ Reset {name}: Deleted {result.deleted_count} records")
    except Exception as e:
        print(f"❌ Failed to reset {name}: {e}")

def show_summary():
    """Show summary of what will be reset"""
    print("\n" + "="*60)
    print("📊 CURRENT MONGODB SUMMARY")
    print("="*60)
    
    cols = {
        "Signals": signals_col,
        "Positions": positions_col,
        "Listing Data": listing_data_col,
        "Watchlist": watchlist_col,
        "Mappings": instrument_keys_col,
        "Logs": logs_col
    }
    
    for name, col in cols.items():
        if col is not None:
            count = col.count_documents({})
            print(f"📋 {name}: {count} records")
        else:
            print(f"📋 {name}: Collection unavailable")
    
    print("="*60 + "\n")

def main():
    """Main reset function"""
    print("\n" + "="*60)
    print("🔄 IPO SCANNER MONGODB RESET")
    print("="*60)
    print(f"⏰ Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60 + "\n")
    
    # Show current summary
    show_summary()
    
    # Confirm reset
    print("⚠️  WARNING: This will reset all IPO data in MongoDB!")
    print("📦 Collections will be archived to CSV files automatically")
    response = input("Type 'RESET' to confirm: ")
    
    if response != "RESET":
        print("❌ Reset cancelled")
        return
    
    print("\n🔄 Starting reset process...\n")
    
    # Reset collections
    reset_collection(signals_col, "signals")
    reset_collection(positions_col, "positions")
    reset_collection(listing_data_col, "listing_data")
    reset_collection(watchlist_col, "watchlist")
    reset_collection(logs_col, "logs")
    
    # Optional mapping reset
    mapping_response = input("\nDo you also want to reset instrument mappings? (y/N): ")
    if mapping_response.lower() == 'y':
        reset_collection(instrument_keys_col, "instrument_keys")
    
    print("\n" + "="*60)
    print("✅ RESET COMPLETE")
    print("="*60)
    print("\n📦 All selected collections have been reset")
    print("💾 Archival CSVs created for safety")
    print("🔄 You can now run the scanners fresh\n")

if __name__ == "__main__":
    main()
