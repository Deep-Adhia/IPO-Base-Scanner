#!/usr/bin/env python3
"""
reset_all_data.py

Reset all CSV files and positions to initial state.
This script will:
- Clear all positions (set status to CLOSED or remove)
- Clear all signals (or archive them)
- Reset listing data (optional)
- Reset watchlist (optional)
- Keep IPO symbol lists intact
"""

import os
import pandas as pd
from datetime import datetime
import shutil

# CSV file paths
SIGNALS_CSV = "ipo_signals.csv"
POSITIONS_CSV = "ipo_positions.csv"
LISTING_DATA_CSV = "ipo_listing_data.csv"
WATCHLIST_CSV = "watchlist.csv"
IPO_SYMBOLS_CSV = "recent_ipo_symbols.csv"
UPSTOX_MAPPING_CSV = "ipo_upstox_mapping.csv"

def backup_csv(filename):
    """Create backup of CSV file before resetting"""
    if os.path.exists(filename):
        backup_name = f"{filename}.backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        shutil.copy2(filename, backup_name)
        print(f"✅ Backed up {filename} to {backup_name}")
        return backup_name
    return None

def reset_signals():
    """Reset signals CSV to empty with headers"""
    backup_csv(SIGNALS_CSV)
    
    df = pd.DataFrame(columns=[
        "signal_id", "symbol", "signal_date", "entry_price", "grade", "score",
        "stop_loss", "target_price", "status", "exit_date", "exit_price", 
        "pnl_pct", "days_held", "signal_type"
    ])
            df.to_csv(SIGNALS_CSV, index=False, encoding='utf-8')
    print(f"✅ Reset {SIGNALS_CSV}")

def reset_positions():
    """Reset positions CSV to empty with headers"""
    backup_csv(POSITIONS_CSV)
    
    df = pd.DataFrame(columns=[
        "symbol", "entry_date", "entry_price", "grade", "current_price",
        "stop_loss", "trailing_stop", "pnl_pct", "days_held", "status"
    ])
            df.to_csv(POSITIONS_CSV, index=False, encoding='utf-8')
    print(f"✅ Reset {POSITIONS_CSV}")

def reset_listing_data():
    """Reset listing data CSV (keep structure, clear data)"""
    backup_csv(LISTING_DATA_CSV)
    
    # Initialize with comment header
    with open(LISTING_DATA_CSV, 'w', encoding='utf-8') as f:
        f.write("# IPO Listing Day Data\n")
        f.write("# Format: symbol,listing_date,listing_day_high,listing_day_low,listing_day_close,listing_day_volume,status,last_updated\n")
    
    df = pd.DataFrame(columns=[
        "symbol", "listing_date", "listing_day_high", "listing_day_low",
        "listing_day_close", "listing_day_volume", "status", "last_updated"
    ])
    df.to_csv(LISTING_DATA_CSV, mode='a', index=False, encoding='utf-8')
    print(f"✅ Reset {LISTING_DATA_CSV}")

def reset_watchlist():
    """Reset watchlist CSV to empty with headers"""
    backup_csv(WATCHLIST_CSV)
    
    df = pd.DataFrame(columns=["symbol", "added_date", "status", "notes"])
    df.to_csv(WATCHLIST_CSV, index=False, encoding='utf-8')
    print(f"✅ Reset {WATCHLIST_CSV}")

def archive_signals():
    """Archive existing signals to a separate file"""
    if os.path.exists(SIGNALS_CSV):
        try:
            df = pd.read_csv(SIGNALS_CSV, encoding='utf-8')
            if not df.empty:
                archive_name = f"ipo_signals_archive_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                df.to_csv(archive_name, index=False, encoding='utf-8')
                print(f"✅ Archived {len(df)} signals to {archive_name}")
                return archive_name
        except Exception as e:
            print(f"⚠️ Could not archive signals: {e}")
    return None

def archive_positions():
    """Archive existing positions to a separate file"""
    if os.path.exists(POSITIONS_CSV):
        try:
            df = pd.read_csv(POSITIONS_CSV, encoding='utf-8')
            if not df.empty:
                archive_name = f"ipo_positions_archive_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                df.to_csv(archive_name, index=False, encoding='utf-8')
                print(f"✅ Archived {len(df)} positions to {archive_name}")
                return archive_name
        except Exception as e:
            print(f"⚠️ Could not archive positions: {e}")
    return None

def show_summary():
    """Show summary of what will be reset"""
    print("\n" + "="*60)
    print("📊 CURRENT DATA SUMMARY")
    print("="*60)
    
    if os.path.exists(SIGNALS_CSV):
        try:
            df = pd.read_csv(SIGNALS_CSV, encoding='utf-8')
            active_signals = len(df[df['status'] == 'ACTIVE']) if 'status' in df.columns else len(df)
            total_signals = len(df)
            print(f"📋 Signals: {total_signals} total ({active_signals} active)")
        except:
            print("📋 Signals: Could not read")
    else:
        print("📋 Signals: File does not exist")
    
    if os.path.exists(POSITIONS_CSV):
        try:
            df = pd.read_csv(POSITIONS_CSV, encoding='utf-8')
            active_positions = len(df[df['status'] == 'ACTIVE']) if 'status' in df.columns else len(df)
            total_positions = len(df)
            print(f"💼 Positions: {total_positions} total ({active_positions} active)")
        except:
            print("💼 Positions: Could not read")
    else:
        print("💼 Positions: File does not exist")
    
    if os.path.exists(LISTING_DATA_CSV):
        try:
            df = pd.read_csv(LISTING_DATA_CSV, comment='#', encoding='utf-8')
            print(f"📅 Listing Data: {len(df)} entries")
        except:
            print("📅 Listing Data: Could not read")
    else:
        print("📅 Listing Data: File does not exist")
    
    if os.path.exists(WATCHLIST_CSV):
        try:
            df = pd.read_csv(WATCHLIST_CSV, encoding='utf-8')
            active_watchlist = len(df[df['status'] == 'ACTIVE']) if 'status' in df.columns else len(df)
            print(f"👀 Watchlist: {active_watchlist} active symbols")
        except:
            print("👀 Watchlist: Could not read")
    else:
        print("👀 Watchlist: File does not exist")
    
    print("="*60 + "\n")

def main():
    """Main reset function"""
    print("\n" + "="*60)
    print("🔄 IPO SCANNER DATA RESET")
    print("="*60)
    print(f"⏰ Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60 + "\n")
    
    # Show current summary
    show_summary()
    
    # Confirm reset
    print("⚠️  WARNING: This will reset all positions and signals!")
    print("📦 Backups will be created automatically")
    response = input("Type 'RESET' to confirm: ")
    
    if response != "RESET":
        print("❌ Reset cancelled")
        return
    
    print("\n🔄 Starting reset process...\n")
    
    # Archive existing data
    archive_signals()
    archive_positions()
    
    # Reset all CSVs
    reset_signals()
    reset_positions()
    reset_listing_data()
    reset_watchlist()
    
    print("\n" + "="*60)
    print("✅ RESET COMPLETE")
    print("="*60)
    print("\n📦 All data has been reset to initial state")
    print("💾 Backups have been created with timestamps")
    print("🔄 You can now run the scanners fresh\n")

if __name__ == "__main__":
    main()

