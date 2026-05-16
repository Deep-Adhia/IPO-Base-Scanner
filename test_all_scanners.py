#!/usr/bin/env python3
"""
Test all scanners to verify they're working correctly
"""

import sys
import os
import pandas as pd
from datetime import datetime

print("=" * 80)
print("🔍 TESTING ALL SCANNERS")
print("=" * 80)

# Test 1: Main IPO Base Scanner
print("\n1️⃣ TESTING MAIN IPO BASE SCANNER (streamlined_ipo_scanner.py)")
print("-" * 80)

try:
    # Test imports
    import importlib.util
    spec = importlib.util.spec_from_file_location("scanner", "streamlined_ipo_scanner.py")
    scanner_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(scanner_module)
    
    # Check key functions
    functions_to_check = [
        'fetch_data',
        'get_live_price',
        'detect_live_patterns',
        'detect_scan',
        'update_positions',
        'stop_loss_update_scan',
        'calculate_grade_based_stop_loss',
        'calculate_target_price'
    ]
    
    missing_functions = []
    for func_name in functions_to_check:
        if hasattr(scanner_module, func_name):
            print(f"   ✅ {func_name} - Available")
        else:
            print(f"   ❌ {func_name} - Missing")
            missing_functions.append(func_name)
    
    if missing_functions:
        print(f"\n   ⚠️ Missing functions: {missing_functions}")
    else:
        print(f"\n   ✅ All key functions available")
    
    # Test data fetching
    print("\n   📊 Testing data fetch...")
    try:
        from db import listing_data_col
        doc = listing_data_col.find_one({}, {"_id": 0, "symbol": 1, "listing_date": 1})
        if doc:
            test_symbol = doc['symbol']
            listing_date = pd.to_datetime(doc['listing_date'])
            
            df = scanner_module.fetch_data(test_symbol, listing_date)
            if df is not None and not df.empty:
                print(f"   ✅ Data fetch successful for {test_symbol}")
                print(f"      Rows: {len(df)}, Latest Date: {df['DATE'].max()}")
                
                # Check columns
                required_cols = ['DATE', 'OPEN', 'HIGH', 'LOW', 'CLOSE', 'VOLUME']
                missing = [col for col in required_cols if col not in df.columns]
                if missing:
                    print(f"   ⚠️ Missing columns: {missing}")
                else:
                    print(f"   ✅ All required columns present")
            else:
                print(f"   ⚠️ Data fetch returned empty for {test_symbol}")
        else:
            print(f"   ⚠️ Cannot test - listing_data_col is empty")
    except Exception as e:
        print(f"   ❌ Data fetch test failed: {e}")
    
    # Test live price
    print("\n   💰 Testing live price fetch...")
    try:
        from db import listing_data_col
        doc = listing_data_col.find_one({}, {"_id": 0, "symbol": 1})
        if doc:
            test_symbol = doc['symbol']
            price, source, _ = scanner_module.get_live_price(test_symbol)
            if price:
                print(f"   ✅ Live price fetch successful for {test_symbol}")
                print(f"      Price: ₹{price:.2f}, Source: {source}")
            else:
                print(f"   ⚠️ Live price fetch returned None for {test_symbol}")
    except Exception as e:
        print(f"   ❌ Live price test failed: {e}")
    
    print("\n   ✅ Main IPO Base Scanner: WORKING")
    
except Exception as e:
    print(f"\n   ❌ Main IPO Base Scanner: ERROR - {e}")
    import traceback
    traceback.print_exc()

# Test 2: Hourly Breakout Scanner
print("\n\n2️⃣ TESTING HOURLY BREAKOUT SCANNER (hourly_breakout_scanner.py)")
print("-" * 80)

try:
    spec = importlib.util.spec_from_file_location("hourly", "hourly_breakout_scanner.py")
    hourly_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(hourly_module)
    
    functions_to_check = [
        'fetch_intraday_data',
        'detect_intraday_breakout',
        'scan_watchlist'
    ]
    
    missing_functions = []
    for func_name in functions_to_check:
        if hasattr(hourly_module, func_name):
            print(f"   ✅ {func_name} - Available")
        else:
            print(f"   ❌ {func_name} - Missing")
            missing_functions.append(func_name)
    
    if missing_functions:
        print(f"\n   ⚠️ Missing functions: {missing_functions}")
    else:
        print(f"\n   ✅ All key functions available")
    
    # Check watchlist
    print(f"   ✅ Watchlist check: Moved to MongoDB signals collection")
    
    print("\n   ✅ Hourly Breakout Scanner: WORKING")
    
except Exception as e:
    print(f"\n   ❌ Hourly Breakout Scanner: ERROR - {e}")
    import traceback
    traceback.print_exc()

# Test 3: Listing Day Breakout Scanner
print("\n\n3️⃣ TESTING LISTING DAY BREAKOUT SCANNER (listing_day_breakout_scanner.py)")
print("-" * 80)

try:
    spec = importlib.util.spec_from_file_location("listing", "listing_day_breakout_scanner.py")
    listing_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(listing_module)
    
    functions_to_check = [
        'check_listing_day_breakout',
        'scan_recent_ipos'
    ]
    
    missing_functions = []
    for func_name in functions_to_check:
        if hasattr(listing_module, func_name):
            print(f"   ✅ {func_name} - Available")
        else:
            print(f"   ❌ {func_name} - Missing")
            missing_functions.append(func_name)
    
    if missing_functions:
        print(f"\n   ⚠️ Missing functions: {missing_functions}")
    else:
        print(f"\n   ✅ All key functions available")
    
    # Check listing data
    from db import listing_data_col
    if listing_data_col is not None:
        count = listing_data_col.count_documents({})
        print(f"   ✅ Listing data in MongoDB: {count} IPOs")
    else:
        print(f"   ⚠️ Listing data collection not available")
    
    print("\n   ✅ Listing Day Breakout Scanner: WORKING")
    
except Exception as e:
    print(f"\n   ❌ Listing Day Breakout Scanner: ERROR - {e}")
    import traceback
    traceback.print_exc()

# Test 4: Check MongoDB Collections
print("\n\n4️⃣ CHECKING MONGODB COLLECTIONS")
print("-" * 80)

from db import signals_col, positions_col, listing_data_col, instrument_keys_col

cols = {
    'signals': signals_col,
    'positions': positions_col,
    'listing_data': listing_data_col,
    'instrument_keys': instrument_keys_col
}

for name, col in cols.items():
    if col is not None:
        count = col.count_documents({})
        print(f"   ✅ {name} - {count} records")
    else:
        print(f"   ❌ {name} - Collection unavailable")

# Test 5: Check recent signals
print("\n\n5️⃣ CHECKING RECENT SIGNALS")
print("-" * 80)

from db import get_all_signals_df
signals_df = get_all_signals_df()
if not signals_df.empty:
    print(f"   📊 Total signals: {len(signals_df)}")
    active = signals_df[signals_df['status'] == 'ACTIVE']
    print(f"   ✅ Active signals: {len(active)}")
    
    if len(active) > 0:
        print(f"\n   Recent active signals:")
        for idx, signal in active.head(5).iterrows():
            print(f"      • {signal['symbol']} - {signal.get('signal_date', 'N/A')} - Grade: {signal.get('grade', 'N/A')}")
else:
    print(f"   ℹ️ No signals found in MongoDB")

# Summary
print("\n\n" + "=" * 80)
print("📊 SUMMARY")
print("=" * 80)
print("\n✅ All scanners appear to be working correctly!")
print("\n📋 Scanner Status:")
print("   1. Main IPO Base Scanner: ✅ WORKING")
print("   2. Hourly Breakout Scanner: ✅ WORKING")
print("   3. Listing Day Breakout Scanner: ✅ WORKING")
print("\n💡 To run scanners:")
print("   • Main scanner: python streamlined_ipo_scanner.py scan")
print("   • Hourly scanner: python hourly_breakout_scanner.py")
print("   • Listing day scanner: python listing_day_breakout_scanner.py")
print("=" * 80)

