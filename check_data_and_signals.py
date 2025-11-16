#!/usr/bin/env python3
"""
Check for new signals and verify OHLC data updates
"""

import pandas as pd
import sys
import os
from datetime import datetime, timedelta

# Import from main scanner
import importlib.util
spec = importlib.util.spec_from_file_location("scanner", "streamlined-ipo-scanner.py")
scanner_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(scanner_module)

fetch_data = scanner_module.fetch_data
get_live_price = scanner_module.get_live_price
logger = scanner_module.logger

SIGNALS_CSV = "ipo_signals.csv"
POSITIONS_CSV = "ipo_positions.csv"

def check_signals():
    """Check existing signals and their data freshness"""
    print("📊 CHECKING SIGNALS")
    print("=" * 80)
    
    if not os.path.exists(SIGNALS_CSV):
        print("❌ No signals file found!")
        return
    
    signals_df = pd.read_csv(SIGNALS_CSV, encoding='utf-8')
    if signals_df.empty:
        print("✅ No signals found (file is empty)")
        return
    
    print(f"\n📈 Found {len(signals_df)} signals\n")
    
    for idx, signal in signals_df.iterrows():
        symbol = signal['symbol']
        signal_date = pd.to_datetime(signal['signal_date']).date()
        entry_price = signal['entry_price']
        status = signal.get('status', 'ACTIVE')
        
        today = datetime.today().date()
        days_old = (today - signal_date).days
        
        print(f"\n{'='*80}")
        print(f"📊 Signal: {symbol} ({status})")
        print(f"   Signal Date: {signal_date} ({days_old} days ago)")
        print(f"   Entry Price: ₹{entry_price:.2f}")
        
        # Check if we can fetch fresh data
        try:
            # Get listing date from mapping
            listing_map = {}
            if os.path.exists('ipo_listing_data.csv'):
                listing_df = pd.read_csv('ipo_listing_data.csv', encoding='utf-8')
                if not listing_df.empty and 'symbol' in listing_df.columns:
                    listing_map = dict(zip(listing_df['symbol'], pd.to_datetime(listing_df['listing_date'])))
            
            listing_date = listing_map.get(symbol)
            if listing_date:
                df = fetch_data(symbol, listing_date)
                if df is not None and not df.empty:
                    latest_date = df['DATE'].max()
                    if hasattr(latest_date, 'date'):
                        latest_date = latest_date.date()
                    
                    days_data_old = (today - latest_date).days
                    
                    print(f"   📅 Latest Data Date: {latest_date} ({days_data_old} days old)")
                    print(f"   📊 Latest OHLC: O={df['OPEN'].iloc[-1]:.2f} H={df['HIGH'].iloc[-1]:.2f} L={df['LOW'].iloc[-1]:.2f} C={df['CLOSE'].iloc[-1]:.2f}")
                    
                    if days_data_old <= 1:
                        print(f"   ✅ Data is fresh!")
                    elif days_data_old <= 2:
                        print(f"   ⚠️ Data is 1-2 days old (may be weekend/holiday)")
                    else:
                        print(f"   ❌ Data is stale ({days_data_old} days old)")
                    
                    # Check live price
                    try:
                        live_price, source = get_live_price(symbol)
                        if live_price:
                            print(f"   💰 Live Price: ₹{live_price:.2f} ({source})")
                            close_diff = abs(live_price - df['CLOSE'].iloc[-1])
                            close_diff_pct = (close_diff / df['CLOSE'].iloc[-1] * 100) if df['CLOSE'].iloc[-1] > 0 else 0
                            print(f"   📊 Close vs Live: {close_diff_pct:.2f}% difference")
                    except:
                        pass
                else:
                    print(f"   ❌ Could not fetch data")
            else:
                print(f"   ⚠️ No listing date found")
        except Exception as e:
            print(f"   ❌ Error: {e}")

def check_positions():
    """Check existing positions and their data freshness"""
    print("\n\n" + "=" * 80)
    print("💰 CHECKING POSITIONS")
    print("=" * 80)
    
    if not os.path.exists(POSITIONS_CSV):
        print("❌ No positions file found!")
        return
    
    positions_df = pd.read_csv(POSITIONS_CSV, encoding='utf-8')
    if positions_df.empty:
        print("✅ No positions found (file is empty)")
        return
    
    print(f"\n💰 Found {len(positions_df)} positions\n")
    
    for idx, pos in positions_df.iterrows():
        symbol = pos['symbol']
        entry_date = pd.to_datetime(pos['entry_date']).date()
        entry_price = pos['entry_price']
        current_price = pos.get('current_price', entry_price)
        status = pos.get('status', 'ACTIVE')
        
        today = datetime.today().date()
        days_held = (today - entry_date).days
        
        print(f"\n{'='*80}")
        print(f"💰 Position: {symbol} ({status})")
        print(f"   Entry Date: {entry_date} ({days_held} days ago)")
        print(f"   Entry Price: ₹{entry_price:.2f}")
        print(f"   Stored Current Price: ₹{current_price:.2f}")
        
        # Check if we can fetch fresh data
        try:
            # Get listing date from mapping
            listing_map = {}
            if os.path.exists('ipo_listing_data.csv'):
                listing_df = pd.read_csv('ipo_listing_data.csv', encoding='utf-8')
                if not listing_df.empty and 'symbol' in listing_df.columns:
                    listing_map = dict(zip(listing_df['symbol'], pd.to_datetime(listing_df['listing_date'])))
            
            listing_date = listing_map.get(symbol)
            if listing_date:
                df = fetch_data(symbol, listing_date)
                if df is not None and not df.empty:
                    latest_date = df['DATE'].max()
                    if hasattr(latest_date, 'date'):
                        latest_date = latest_date.date()
                    
                    days_data_old = (today - latest_date).days
                    
                    print(f"   📅 Latest Data Date: {latest_date} ({days_data_old} days old)")
                    print(f"   📊 Latest OHLC: O={df['OPEN'].iloc[-1]:.2f} H={df['HIGH'].iloc[-1]:.2f} L={df['LOW'].iloc[-1]:.2f} C={df['CLOSE'].iloc[-1]:.2f}")
                    
                    if days_data_old <= 1:
                        print(f"   ✅ Data is fresh!")
                    elif days_data_old <= 2:
                        print(f"   ⚠️ Data is 1-2 days old (may be weekend/holiday)")
                    else:
                        print(f"   ❌ Data is stale ({days_data_old} days old)")
                    
                    # Check live price
                    try:
                        live_price, source = get_live_price(symbol)
                        if live_price:
                            print(f"   💰 Live Price: ₹{live_price:.2f} ({source})")
                            stored_diff = abs(live_price - current_price)
                            stored_diff_pct = (stored_diff / current_price * 100) if current_price > 0 else 0
                            print(f"   📊 Stored vs Live: {stored_diff_pct:.2f}% difference")
                            
                            close_diff = abs(live_price - df['CLOSE'].iloc[-1])
                            close_diff_pct = (close_diff / df['CLOSE'].iloc[-1] * 100) if df['CLOSE'].iloc[-1] > 0 else 0
                            print(f"   📊 Close vs Live: {close_diff_pct:.2f}% difference")
                    except:
                        pass
                else:
                    print(f"   ❌ Could not fetch data")
            else:
                print(f"   ⚠️ No listing date found")
        except Exception as e:
            print(f"   ❌ Error: {e}")

def verify_ohlc_data():
    """Verify OHLC data structure and freshness"""
    print("\n\n" + "=" * 80)
    print("🔍 VERIFYING OHLC DATA STRUCTURE")
    print("=" * 80)
    
    # Check a sample symbol
    print("\n📊 Testing data fetch for sample symbols...")
    
    # Get recent IPO symbols
    if os.path.exists('recent_ipo_symbols.csv'):
        symbols_df = pd.read_csv('recent_ipo_symbols.csv', encoding='utf-8')
        if not symbols_df.empty and 'symbol' in symbols_df.columns:
            test_symbols = symbols_df['symbol'].head(3).tolist()
            
            # Get listing dates
            listing_map = {}
            if os.path.exists('ipo_listing_data.csv'):
                listing_df = pd.read_csv('ipo_listing_data.csv', encoding='utf-8')
                if not listing_df.empty and 'symbol' in listing_df.columns:
                    listing_map = dict(zip(listing_df['symbol'], pd.to_datetime(listing_df['listing_date'])))
            
            for symbol in test_symbols:
                print(f"\n{'='*80}")
                print(f"🔍 Testing: {symbol}")
                
                listing_date = listing_map.get(symbol)
                if listing_date:
                    try:
                        df = fetch_data(symbol, listing_date)
                        if df is not None and not df.empty:
                            print(f"   ✅ Data fetched successfully")
                            print(f"   📊 Rows: {len(df)}")
                            print(f"   📅 Date Range: {df['DATE'].min()} to {df['DATE'].max()}")
                            
                            # Check columns
                            required_cols = ['DATE', 'OPEN', 'HIGH', 'LOW', 'CLOSE', 'VOLUME']
                            missing_cols = [col for col in required_cols if col not in df.columns]
                            if missing_cols:
                                print(f"   ❌ Missing columns: {missing_cols}")
                            else:
                                print(f"   ✅ All required columns present")
                            
                            # Check latest data
                            latest = df.iloc[-1]
                            latest_date = latest['DATE']
                            if hasattr(latest_date, 'date'):
                                latest_date = latest_date.date()
                            
                            today = datetime.today().date()
                            days_old = (today - latest_date).days
                            
                            print(f"   📅 Latest Date: {latest_date} ({days_old} days old)")
                            print(f"   📊 Latest OHLC: O={latest['OPEN']:.2f} H={latest['HIGH']:.2f} L={latest['LOW']:.2f} C={latest['CLOSE']:.2f} V={latest['VOLUME']:,.0f}")
                            
                            # Check data quality
                            if days_old <= 1:
                                print(f"   ✅ Data is fresh!")
                            elif days_old <= 2:
                                print(f"   ⚠️ Data is 1-2 days old")
                            else:
                                print(f"   ❌ Data is stale")
                            
                            # Check for null values
                            null_counts = df[required_cols].isnull().sum()
                            if null_counts.sum() > 0:
                                print(f"   ⚠️ Null values found:")
                                for col, count in null_counts.items():
                                    if count > 0:
                                        print(f"      {col}: {count}")
                            else:
                                print(f"   ✅ No null values")
                            
                            # Check data sorting
                            if df['DATE'].is_monotonic_increasing:
                                print(f"   ✅ Data is sorted correctly (ascending)")
                            else:
                                print(f"   ⚠️ Data may not be sorted correctly")
                        else:
                            print(f"   ❌ Could not fetch data")
                    except Exception as e:
                        print(f"   ❌ Error: {e}")
                else:
                    print(f"   ⚠️ No listing date found")
        else:
            print("   ⚠️ No symbols found in recent_ipo_symbols.csv")
    else:
        print("   ⚠️ recent_ipo_symbols.csv not found")

if __name__ == "__main__":
    print("🔍 CHECKING SIGNALS AND DATA UPDATES")
    print("=" * 80)
    print("This script checks:")
    print("1. Existing signals and their data freshness")
    print("2. Existing positions and their data freshness")
    print("3. OHLC data structure and quality")
    print("=" * 80)
    
    check_signals()
    check_positions()
    verify_ohlc_data()
    
    print("\n\n" + "=" * 80)
    print("✅ CHECK COMPLETE")
    print("=" * 80)

