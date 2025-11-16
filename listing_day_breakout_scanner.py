#!/usr/bin/env python3
"""
listing_day_breakout_scanner.py

IPO Listing Day Breakout Scanner:
- Tracks listing day high and low for each IPO
- Detects when stock breaks listing day high with volume
- Entry: When breaks listing day high with volume
- Stop Loss: Listing day low
- Target: Based on listing day range
"""

import os
import sys
import pandas as pd
import numpy as np
import requests
import time
from datetime import datetime, timedelta
from dotenv import load_dotenv
import logging

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import from main scanner
import importlib.util
spec = importlib.util.spec_from_file_location("scanner", "streamlined-ipo-scanner.py")
scanner_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(scanner_module)

fetch_data = scanner_module.fetch_data
send_telegram = scanner_module.send_telegram
logger = scanner_module.logger
get_live_price = scanner_module.get_live_price

# Load environment
load_dotenv()

# Configuration
LISTING_DATA_CSV = "ipo_listing_data.csv"
SIGNALS_CSV = "ipo_signals.csv"
POSITIONS_CSV = "ipo_positions.csv"
RECENT_IPO_CSV = "recent_ipo_symbols.csv"
MIN_VOLUME_MULTIPLIER = 1.5  # Minimum volume spike for breakout confirmation
MAX_ENTRY_ABOVE_HIGH_PCT = 5.0  # Maximum % above listing high for entry (prevents late entries - stricter)
MIN_RISK_REWARD = 1.0  # Minimum risk/reward ratio (1:1 minimum)
STOP_LOSS_PCT = 8.0  # Fixed stop loss % below entry (8%)
MIN_VOLUME_VS_LISTING_DAY = 1.2  # Minimum current volume vs listing day volume (1.2x = 20% higher)

def initialize_listing_data_csv():
    """Initialize listing data CSV if it doesn't exist"""
    if not os.path.exists(LISTING_DATA_CSV):
        df = pd.DataFrame(columns=[
            'symbol', 'listing_date', 'listing_day_high', 'listing_day_low',
            'listing_day_close', 'listing_day_volume', 'last_updated', 'status'
        ])
        df.to_csv(LISTING_DATA_CSV, index=False, encoding='utf-8')
        logger.info(f"Created {LISTING_DATA_CSV}")
    else:
        # Clean up any comment rows if file exists
        try:
            df = pd.read_csv(LISTING_DATA_CSV, encoding='utf-8')
            df = df[~df['symbol'].astype(str).str.startswith('#')]
            df = df[df['symbol'].notna() & (df['symbol'] != '')]
            df.to_csv(LISTING_DATA_CSV, index=False, encoding='utf-8')
        except:
            pass

def load_listing_data():
    """Load listing day data from CSV"""
    try:
        if not os.path.exists(LISTING_DATA_CSV):
            initialize_listing_data_csv()
            return pd.DataFrame()
        
        df = pd.read_csv(LISTING_DATA_CSV, encoding='utf-8')
        # Remove comment rows and empty rows
        df = df[~df['symbol'].astype(str).str.startswith('#')]
        df = df[df['symbol'].notna() & (df['symbol'] != '')]
        
        # Convert listing_date to date object if it exists
        if 'listing_date' in df.columns and not df.empty:
            df['listing_date'] = pd.to_datetime(df['listing_date']).dt.date
        
        return df
    except Exception as e:
        logger.error(f"Error loading listing data: {e}")
        return pd.DataFrame()

def save_listing_data(df):
    """Save listing data to CSV"""
    try:
        df.to_csv(LISTING_DATA_CSV, index=False, encoding='utf-8')
    except Exception as e:
        logger.error(f"Error saving listing data: {e}")

def get_listing_day_data(symbol, listing_date):
    """Fetch and extract listing day high/low from historical data"""
    try:
        logger.info(f"📊 Fetching listing day data for {symbol} (listing date: {listing_date})")
        
        # Convert listing_date to date object if needed (before calling fetch_data)
        if isinstance(listing_date, str):
            listing_date = pd.to_datetime(listing_date).date()
        elif hasattr(listing_date, 'date'):
            listing_date = listing_date.date()
        elif isinstance(listing_date, pd.Timestamp):
            listing_date = listing_date.date()
        
        # Fetch data starting from listing date
        df = fetch_data(symbol, listing_date)
        
        if df is None or df.empty:
            logger.warning(f"⚠️ No data available for {symbol} on listing date")
            return None
        
        # Find listing day data
        df['DATE'] = pd.to_datetime(df['DATE']).dt.date
        listing_day_data = df[df['DATE'] == listing_date]
        
        if listing_day_data.empty:
            # Try to get first day of data (might be listing day)
            listing_day_data = df.iloc[[0]]
            logger.info(f"Using first available day as listing day for {symbol}")
        
        if listing_day_data.empty:
            logger.warning(f"⚠️ Could not find listing day data for {symbol}")
            return None
        
        # Extract listing day metrics
        listing_day_high = float(listing_day_data['HIGH'].iloc[0])
        listing_day_low = float(listing_day_data['LOW'].iloc[0])
        listing_day_close = float(listing_day_data['CLOSE'].iloc[0])
        listing_day_volume = float(listing_day_data['VOLUME'].iloc[0])
        actual_listing_date = listing_day_data['DATE'].iloc[0]
        
        logger.info(f"✅ Listing day data for {symbol}:")
        logger.info(f"   High: ₹{listing_day_high:.2f}")
        logger.info(f"   Low: ₹{listing_day_low:.2f}")
        logger.info(f"   Close: ₹{listing_day_close:.2f}")
        logger.info(f"   Volume: {listing_day_volume:,.0f}")
        
        return {
            'symbol': symbol,
            'listing_date': actual_listing_date,
            'listing_day_high': listing_day_high,
            'listing_day_low': listing_day_low,
            'listing_day_close': listing_day_close,
            'listing_day_volume': listing_day_volume,
            'last_updated': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'status': 'ACTIVE'
        }
    
    except Exception as e:
        logger.error(f"Error getting listing day data for {symbol}: {e}")
        return None

def update_listing_data_for_new_ipos():
    """Check for new IPOs and add their listing day data"""
    try:
        # Load recent IPOs
        if not os.path.exists(RECENT_IPO_CSV):
            logger.warning(f"{RECENT_IPO_CSV} not found")
            return
        
        recent_ipos = pd.read_csv(RECENT_IPO_CSV, encoding='utf-8')
        recent_ipos['listing_date'] = pd.to_datetime(recent_ipos['listing_date']).dt.date
        
        # Load existing listing data
        listing_data = load_listing_data()
        
        if listing_data.empty:
            existing_symbols = set()
        else:
            existing_symbols = set(listing_data['symbol'].tolist())
        
        new_ipos = 0
        
        for _, row in recent_ipos.iterrows():
            symbol = row['symbol']
            listing_date = row['listing_date']
            
            # Skip if already exists
            if symbol in existing_symbols:
                continue
            
            # Get listing day data
            listing_info = get_listing_day_data(symbol, listing_date)
            
            if listing_info:
                # Add to listing data
                new_row = pd.DataFrame([listing_info])
                if listing_data.empty:
                    listing_data = new_row
                else:
                    listing_data = pd.concat([listing_data, new_row], ignore_index=True)
                
                new_ipos += 1
                logger.info(f"✅ Added listing data for {symbol}")
                
                # Small delay to avoid rate limiting
                time.sleep(0.5)
            else:
                logger.warning(f"⚠️ Could not get listing data for {symbol} - skipping")
        
        if new_ipos > 0:
            save_listing_data(listing_data)
            logger.info(f"✅ Updated listing data: {new_ipos} new IPOs added")
        else:
            logger.info("✅ No new IPOs to add")
    
    except Exception as e:
        logger.error(f"Error updating listing data: {e}")

def check_listing_day_breakout(symbol, listing_info):
    """Check if symbol has broken listing day high with volume"""
    try:
        listing_day_high = listing_info['listing_day_high']
        listing_day_low = listing_info['listing_day_low']
        listing_day_close = listing_info.get('listing_day_close', listing_day_high)  # Fallback to high if close not available
        listing_date = listing_info['listing_date']
        last_updated = listing_info.get('last_updated', 'N/A')  # When listing data was captured
        
        # Convert listing_date to date object if needed (from CSV it might be string)
        if isinstance(listing_date, str):
            listing_date = pd.to_datetime(listing_date).date()
        elif hasattr(listing_date, 'date'):
            listing_date = listing_date.date()
        elif isinstance(listing_date, pd.Timestamp):
            listing_date = listing_date.date()
        
        # Fetch current data
        df = fetch_data(symbol, listing_date)
        
        if df is None or df.empty:
            return None
        
        # Get latest data from historical
        latest = df.iloc[-1]
        historical_high = float(latest['HIGH'])
        current_volume = float(latest['VOLUME'])
        current_date = latest['DATE']
        
        # Validate data freshness - check if latest data is from today
        latest_date = latest['DATE']
        if isinstance(latest_date, pd.Timestamp):
            latest_date = latest_date.date()
        elif hasattr(latest_date, 'date'):
            latest_date = latest_date.date()
        else:
            latest_date = pd.to_datetime(latest_date).date()
        
        today_date = datetime.today().date()
        days_old = (today_date - latest_date).days
        
        # CRITICAL: Get LIVE price FIRST for accurate breakout detection
        current_price = None
        current_high = None
        price_source = "Historical Close"
        
        try:
            live_price, live_source = get_live_price(symbol)
            if live_price is not None and live_price > 0:
                current_price = live_price
                current_high = live_price  # Use live price as current high for breakout detection
                price_source = f"Live ({live_source})"
                logger.info(f"✅ Using live price for {symbol} breakout detection: ₹{current_price:.2f} from {live_source}")
        except Exception as e:
            logger.debug(f"Could not get live price for {symbol}: {e}")
        
        # Fallback to historical data if live price unavailable
        if current_price is None:
            current_price = float(latest['CLOSE'])
            current_high = historical_high  # Use historical high
            price_source = f"Historical Close ({latest_date.strftime('%Y-%m-%d')})"
            
            # Warn if data is stale
            if days_old > 1:
                logger.warning(f"⚠️ Using stale data for {symbol}: {days_old} days old ({latest_date})")
            elif days_old == 0:
                logger.info(f"✅ Using today's historical close for {symbol}: ₹{current_price:.2f}")
            else:
                logger.info(f"⚠️ Using yesterday's close for {symbol}: ₹{current_price:.2f} (market may be closed)")
        
        # Log breakout level comparison
        logger.info(f"📊 {symbol} Breakout Level Check:")
        logger.info(f"   Listing Day High: ₹{listing_day_high:.2f}")
        logger.info(f"   Current High: ₹{current_high:.2f} ({price_source})")
        logger.info(f"   Breakout Required: Current High > ₹{listing_day_high:.2f}")
        
        # Calculate average volume (last 10 days excluding listing day)
        if len(df) > 1:
            recent_df = df.tail(10)
            avg_volume = recent_df['VOLUME'].mean()
        else:
            avg_volume = current_volume
        
        # Check for breakout
        is_breakout = False
        breakout_conditions = []
        
        # Condition 1: Price breaks listing day high
        if current_high > listing_day_high:
            is_breakout = True
            breakout_conditions.append(f"Price broke listing day high ({current_high:.2f} > {listing_day_high:.2f})")
        
        # Condition 2: Volume confirmation
        volume_spike = current_volume >= avg_volume * MIN_VOLUME_MULTIPLIER
        if volume_spike:
            breakout_conditions.append(f"Volume spike ({current_volume:,.0f} vs avg {avg_volume:,.0f})")
        
        if is_breakout and volume_spike:
            # Calculate entry, stop loss, and target
            entry_price = current_price  # Use current price as entry
            
            # CRITICAL FIX: Calculate target based on ENTRY price, not listing day high
            # This ensures target is always above entry price
            listing_range = listing_day_high - listing_day_low
            listing_range_pct = (listing_range / listing_day_high * 100) if listing_day_high > 0 else 0
            
            # Note: Listing day range is not used for rejection - listing day low is last support level
            # Stop loss is purely percentage-based (8% below entry), not based on listing day low
            
            # Calculate how far above listing high the entry is
            entry_above_high = entry_price - listing_day_high
            entry_above_high_pct = (entry_above_high / listing_day_high * 100) if listing_day_high > 0 else 0
            
            # FILTER 2: Only generate signals if entry is within reasonable distance of listing high
            # This prevents generating signals when breakout happened long ago
            if entry_above_high_pct > MAX_ENTRY_ABOVE_HIGH_PCT:
                logger.info(f"⏭️ Skipping {symbol}: Entry (₹{entry_price:.2f}) is {entry_above_high_pct:.1f}% above listing high (₹{listing_day_high:.2f}) - too far from breakout level, no point entering now")
                return None
            
            # Calculate days since listing (for display/information only - no filter)
            # Some IPOs correct for months before breaking listing day high - this is still valid
            today_date = datetime.today().date()
            if isinstance(listing_date, str):
                listing_date_obj = pd.to_datetime(listing_date).date()
            elif hasattr(listing_date, 'date'):
                listing_date_obj = listing_date.date()
            else:
                listing_date_obj = listing_date
            
            days_since_listing = (today_date - listing_date_obj).days
            # No time filter - IPOs that correct for months and then break listing day high are valid
            
            # FILTER 3: Volume confirmation - current volume should be elevated vs listing day
            volume_vs_listing_day = current_volume / listing_day_volume if listing_day_volume > 0 else 0
            if volume_vs_listing_day < MIN_VOLUME_VS_LISTING_DAY:
                logger.info(f"⏭️ Skipping {symbol}: Current volume ({current_volume:,.0f}) is only {volume_vs_listing_day:.1f}x listing day volume (min: {MIN_VOLUME_VS_LISTING_DAY:.1f}x) - insufficient volume confirmation")
                return None
            
            # CRITICAL FIX: Stop loss is 8% below entry (fixed percentage, NOT based on listing day low)
            # Listing day low is the last support level (reference only), but stop loss is purely entry-based
            # Always use 8% below entry regardless of entry distance from listing high
            stop_loss_pct = 0.08  # Fixed 8% below entry
            
            # Calculate stop loss purely based on entry price percentage
            stop_loss = entry_price * (1 - stop_loss_pct)
            stop_loss_pct_below_entry = stop_loss_pct * 100
            
            logger.info(f"📊 {symbol} Stop Loss: Using 8% below entry (₹{stop_loss:.2f})")
            logger.info(f"   Entry: ₹{entry_price:.2f}")
            logger.info(f"   Listing Day High: ₹{listing_day_high:.2f} (Entry is {entry_above_high_pct:.1f}% above)")
            logger.info(f"   Listing Day Low: ₹{listing_day_low:.2f} (Last support level - reference only)")
            logger.info(f"   Listing Day Range: {listing_range_pct:.1f}% (High-Low spread)")
            logger.info(f"   Selected Stop: ₹{stop_loss:.2f} (8% below entry)")
            logger.info(f"   Days Since Listing: {days_since_listing} days ({'Fresh' if days_since_listing <= 5 else 'Moderate' if days_since_listing <= 15 else 'Mature'})")
            logger.info(f"   Volume Confirmation: {volume_vs_listing_day:.1f}x listing day volume ✅")
            
            # Target calculation: Use entry price + percentage of listing range
            # This ensures target is always above entry
            # Use 50-100% of listing range as target, depending on how close entry is to listing high
            if entry_above_high_pct <= 2.0:
                # Entry is very close to listing high - use larger target (100% of range)
                target_multiplier = 1.0
            elif entry_above_high_pct <= 5.0:
                # Entry is moderately above - use medium target (75% of range)
                target_multiplier = 0.75
            else:
                # Entry is further above - use smaller target (50% of range)
                target_multiplier = 0.5
            
            target_price = entry_price + (listing_range * target_multiplier)
            
            # Risk/Reward
            risk = entry_price - stop_loss
            reward = target_price - entry_price
            risk_reward = reward / risk if risk > 0 else 0
            
            # FILTER: Minimum risk/reward ratio (reward must be at least equal to risk)
            if risk_reward < MIN_RISK_REWARD:
                logger.info(f"⏭️ Skipping {symbol}: Risk/Reward ratio ({risk_reward:.2f}) is below minimum ({MIN_RISK_REWARD:.1f})")
                return None
            
            # Validation: Risk should be exactly 8% (since stop loss is fixed at 8%)
            risk_pct = (risk / entry_price * 100) if entry_price > 0 else 0
            if abs(risk_pct - STOP_LOSS_PCT) > 0.1:  # Allow small floating point differences
                logger.warning(f"⚠️ {symbol}: Risk ({risk_pct:.1f}%) doesn't match expected stop loss ({STOP_LOSS_PCT:.1f}%) - unexpected")
            
            # Days since listing already calculated above
            
            # Calculate gain from listing day close
            gain_from_listing_close = ((current_price - listing_day_close) / listing_day_close * 100) if listing_day_close > 0 else 0
            
            return {
                'symbol': symbol,
                'listing_date': listing_date,
                'listing_day_high': listing_day_high,
                'listing_day_low': listing_day_low,
                'listing_day_close': listing_day_close,
                'current_price': current_price,
                'current_high': current_high,
                'entry_price': round(entry_price, 2),
                'stop_loss': round(stop_loss, 2),
                'target_price': round(target_price, 2),
                'volume_spike': round(current_volume / avg_volume, 2),
                'volume_vs_listing_day': round(volume_vs_listing_day, 2),
                'listing_range_pct': round(listing_range_pct, 2),
                'risk_reward': round(risk_reward, 2),
                'breakout_date': current_date,
                'breakout_conditions': ' | '.join(breakout_conditions),
                'price_source': price_source,
                'days_since_listing': days_since_listing,
                'gain_from_listing_close': round(gain_from_listing_close, 2),
                'entry_above_high_pct': round(entry_above_high_pct, 2),
                'target_multiplier': round(target_multiplier, 2),
                'last_updated': last_updated
            }
        
        return None
    
    except Exception as e:
        logger.error(f"Error checking breakout for {symbol}: {e}")
        return None

def format_listing_breakout_alert(breakout_data):
    """Format listing day breakout alert"""
    symbol = breakout_data['symbol']
    entry = breakout_data['entry_price']
    stop = breakout_data['stop_loss']
    target = breakout_data['target_price']
    listing_high = breakout_data['listing_day_high']
    listing_low = breakout_data['listing_day_low']
    listing_close = breakout_data.get('listing_day_close', listing_high)
    current_high = breakout_data['current_high']
    current_price = breakout_data['current_price']
    vol_spike = breakout_data['volume_spike']
    rr = breakout_data['risk_reward']
    conditions = breakout_data['breakout_conditions']
    days_since_listing = breakout_data.get('days_since_listing', 0)
    gain_from_listing = breakout_data.get('gain_from_listing_close', 0)
    price_source = breakout_data.get('price_source', 'Historical Close')
    entry_above_high_pct = breakout_data.get('entry_above_high_pct', 0)
    target_multiplier = breakout_data.get('target_multiplier', 0.5)
    volume_vs_listing_day = breakout_data.get('volume_vs_listing_day', 0)
    listing_range_pct = breakout_data.get('listing_range_pct', 0)
    last_updated = breakout_data.get('last_updated', 'N/A')
    breakout_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    # Determine freshness based on days since listing
    if days_since_listing <= 5:
        freshness = "🟢 Very Fresh"
        freshness_desc = "Fresh breakout - early stage"
    elif days_since_listing <= 30:
        freshness = "🟡 Moderate"
        freshness_desc = "Moderate - correction phase"
    elif days_since_listing <= 90:
        freshness = "🟠 Mature"
        freshness_desc = "Mature - extended correction"
    else:
        freshness = "🔴 Extended"
        freshness_desc = "Extended correction - breaking out after months"
    
    # Format listing date
    listing_date = breakout_data['listing_date']
    if isinstance(listing_date, str):
        listing_date_str = listing_date
    elif hasattr(listing_date, 'strftime'):
        listing_date_str = listing_date.strftime('%Y-%m-%d')
    else:
        listing_date_str = str(listing_date)
    
    # Gain emoji
    gain_emoji = "📈" if gain_from_listing >= 0 else "📉"
    
    msg = f"""🎯 <b>LISTING DAY HIGH BREAKOUT!</b>

📊 Symbol: <b>{symbol}</b>
📋 Signal Type: <b>Listing Day Breakout</b>

⏰ <b>Timing Information:</b>
• Listing Date: {listing_date_str}
• Days Since Listing: {days_since_listing} days {freshness}
• Freshness: {freshness_desc}
• Base Time (Data Captured): {last_updated}
• Breakout Detected: {breakout_time}

💰 <b>Entry Details:</b>
• Current Price: ₹{current_price:,.2f} ({price_source})
• Entry: ₹{entry:,.2f} ({entry_above_high_pct:+.1f}% above listing high)
• Stop Loss: ₹{stop:,.2f} (8% below entry)
• Target: ₹{target:,.2f} (Entry + {target_multiplier*100:.0f}% of listing range)
• Risk:Reward: 1:{rr:.1f} ✅

📈 <b>Listing Day Metrics:</b>
• Listing Day High: ₹{listing_high:,.2f}
• Listing Day Low: ₹{listing_low:,.2f}
• Listing Day Close: ₹{listing_close:,.2f}
• Current High: ₹{current_high:,.2f} ✅ BROKEN!
• {gain_emoji} Gain from Listing Close: {gain_from_listing:+.2f}%

📊 <b>Breakout Confirmation:</b>
• Volume Spike: {vol_spike:.1f}x (vs recent average)
• Volume vs Listing Day: {volume_vs_listing_day:.1f}x ✅
• Listing Day Range: {listing_range_pct:.1f}% (High-Low spread)
• {conditions}

⚠️ <b>Action Required:</b> Enter position - Listing day high broken with volume!"""
    
    return msg

def save_breakout_signal(breakout_data):
    """Save breakout signal to signals CSV"""
    try:
        today = datetime.now().date()
        signal_id = f"LISTING_{breakout_data['symbol']}_{today.strftime('%Y%m%d')}_{datetime.now().strftime('%H%M')}"
        
        # Check if signal already exists
        if os.path.exists(SIGNALS_CSV):
            existing_signals = pd.read_csv(SIGNALS_CSV, encoding='utf-8')
            # Add signal_type column if it doesn't exist (for backward compatibility)
            if 'signal_type' not in existing_signals.columns:
                existing_signals['signal_type'] = 'UNKNOWN'
            # Check if we already have a listing day breakout signal for this symbol today
            today_signals = existing_signals[
                (existing_signals['symbol'] == breakout_data['symbol']) &
                (pd.to_datetime(existing_signals['signal_date']).dt.date == today) &
                (existing_signals['signal_id'].str.contains('LISTING_', na=False))
            ]
            if len(today_signals) > 0:
                logger.info(f"Listing day breakout signal already exists for {breakout_data['symbol']} today")
                return False
        else:
            existing_signals = pd.DataFrame()
        
        # Check if symbol already has active position (prevent duplicates)
        try:
            if os.path.exists(POSITIONS_CSV):
                existing_positions = pd.read_csv(POSITIONS_CSV, encoding='utf-8')
                if not existing_positions.empty:
                    active_positions = existing_positions[existing_positions['status'] == 'ACTIVE']
                    if breakout_data['symbol'] in active_positions['symbol'].tolist():
                        logger.info(f"⏭️ Skipping {breakout_data['symbol']} - already has active position")
                        return False
        except:
            pass
        
        # Create new signal
        new_signal = {
            "signal_id": signal_id,
            "symbol": breakout_data['symbol'],
            "signal_date": today,
            "entry_price": breakout_data['entry_price'],
            "grade": "LISTING_BREAKOUT",
            "score": 100,  # High score for listing day breakout
            "stop_loss": breakout_data['stop_loss'],
            "target_price": breakout_data['target_price'],
            "status": "ACTIVE",
            "exit_date": "",
            "exit_price": 0,
            "pnl_pct": 0,
            "days_held": 0,
            "signal_type": "LISTING_DAY_BREAKOUT"
        }
        
        # Append to CSV
        new_df = pd.DataFrame([new_signal])
        if existing_signals.empty:
            new_df.to_csv(SIGNALS_CSV, index=False, encoding='utf-8')
        else:
            # Ensure signal_type column exists in existing_signals
            if 'signal_type' not in existing_signals.columns:
                existing_signals['signal_type'] = 'UNKNOWN'
            pd.concat([existing_signals, new_df], ignore_index=True).to_csv(SIGNALS_CSV, index=False, encoding='utf-8')
        
        logger.info(f"✅ Saved breakout signal for {breakout_data['symbol']}")
        return True
    
    except Exception as e:
        logger.error(f"Error saving signal: {e}")
        return False

def add_position(breakout_data):
    """Add position to positions CSV"""
    try:
        today = datetime.now().date()
        
        # Check if position already exists
        if os.path.exists(POSITIONS_CSV):
            existing_positions = pd.read_csv(POSITIONS_CSV)
            active_positions = existing_positions[existing_positions['status'] == 'ACTIVE']
            if breakout_data['symbol'] in active_positions['symbol'].tolist():
                logger.info(f"Position already exists for {breakout_data['symbol']}")
                return False
        else:
            existing_positions = pd.DataFrame()
        
        # Check if position already exists (double check)
        try:
            if os.path.exists(POSITIONS_CSV):
                existing_positions = pd.read_csv(POSITIONS_CSV, encoding='utf-8')
                if not existing_positions.empty:
                    active_positions = existing_positions[existing_positions['status'] == 'ACTIVE']
                    if breakout_data['symbol'] in active_positions['symbol'].tolist():
                        logger.info(f"⏭️ Position already exists for {breakout_data['symbol']}")
                        return False
        except:
            pass
        
        # Create new position
        new_position = {
            "symbol": breakout_data['symbol'],
            "entry_date": today,
            "entry_price": breakout_data['entry_price'],
            "grade": "LISTING_BREAKOUT",
            "current_price": breakout_data['entry_price'],
            "stop_loss": breakout_data['stop_loss'],
            "trailing_stop": breakout_data['stop_loss'],
            "pnl_pct": 0,
            "days_held": 0,
            "status": "ACTIVE"
        }
        
        # Append to CSV
        new_df = pd.DataFrame([new_position])
        if existing_positions.empty:
            new_df.to_csv(POSITIONS_CSV, index=False, encoding='utf-8')
        else:
            pd.concat([existing_positions, new_df], ignore_index=True).to_csv(POSITIONS_CSV, index=False, encoding='utf-8')
        
        logger.info(f"✅ Added position for {breakout_data['symbol']}")
        return True
    
    except Exception as e:
        logger.error(f"Error adding position: {e}")
        return False

def update_listing_status(symbol, status):
    """Update status of listing data entry"""
    try:
        listing_data = load_listing_data()
        if not listing_data.empty and symbol in listing_data['symbol'].tolist():
            listing_data.loc[listing_data['symbol'] == symbol, 'status'] = status
            listing_data.loc[listing_data['symbol'] == symbol, 'last_updated'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            save_listing_data(listing_data)
    except Exception as e:
        logger.error(f"Error updating listing status: {e}")

def scan_listing_day_breakouts():
    """Main function to scan for listing day breakouts"""
    logger.info("🚀 Starting Listing Day Breakout Scan...")
    logger.info("=" * 60)
    
    # Initialize CSV
    initialize_listing_data_csv()
    
    # Update listing data for new IPOs
    logger.info("📊 Step 1: Updating listing data for new IPOs...")
    update_listing_data_for_new_ipos()
    
    # Load active listings
    logger.info("\n📊 Step 2: Scanning for breakouts...")
    listing_data = load_listing_data()
    
    if listing_data.empty:
        logger.warning("No listing data available")
        return
    
    # Filter for active listings
    active_listings = listing_data[listing_data['status'] == 'ACTIVE']
    
    if active_listings.empty:
        logger.info("No active listings to monitor")
        return
    
    logger.info(f"📋 Monitoring {len(active_listings)} active listings...")
    
    breakouts_found = 0
    
    for idx, listing_info in active_listings.iterrows():
        symbol = listing_info['symbol']
        logger.info(f"\n🔍 Checking {symbol}...")
        
        try:
            # Check for breakout
            breakout = check_listing_day_breakout(symbol, listing_info)
            
            if breakout:
                logger.info(f"🎯 BREAKOUT DETECTED for {symbol}!")
                logger.info(f"   Entry: ₹{breakout['entry_price']:.2f}")
                logger.info(f"   Stop Loss: ₹{breakout['stop_loss']:.2f}")
                logger.info(f"   Target: ₹{breakout['target_price']:.2f}")
                
                # Save signal
                if save_breakout_signal(breakout):
                    # Add position
                    add_position(breakout)
                    
                    # Update listing status
                    update_listing_status(symbol, 'BREAKOUT')
                    
                    # Send alert
                    alert_msg = format_listing_breakout_alert(breakout)
                    send_telegram(alert_msg)
                    
                    breakouts_found += 1
                
                # Small delay
                time.sleep(0.5)
            else:
                logger.info(f"✅ {symbol}: No breakout yet (Current price below listing day high)")
            
            # Rate limiting
            time.sleep(0.3)
        
        except Exception as e:
            logger.error(f"Error scanning {symbol}: {e}")
            continue
    
    logger.info(f"\n{'='*60}")
    logger.info(f"✅ Scan complete: {breakouts_found} breakouts found")
    
    # Send summary
    if breakouts_found > 0:
        summary = f"""📊 <b>Listing Day Breakout Scan Summary</b>

🔍 Listings Monitored: {len(active_listings)}
🎯 Breakouts Found: {breakouts_found}
⏰ Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

{'🎉 New breakouts detected! Check alerts above.' if breakouts_found > 0 else '✅ No breakouts at this time.'}"""
        send_telegram(summary)

def main():
    """Main function"""
    try:
        print("IPO Listing Day Breakout Scanner")
        print("=" * 60)
        print(f"Date: {datetime.now().strftime('%Y-%m-%d')}")
        print(f"Time: {datetime.now().strftime('%H:%M:%S')}")
        print("=" * 60)
    except:
        # Fallback for systems with encoding issues
        print("IPO Listing Day Breakout Scanner")
        print("=" * 60)
        print(f"Date: {datetime.now().strftime('%Y-%m-%d')}")
        print(f"Time: {datetime.now().strftime('%H:%M:%S')}")
        print("=" * 60)
    
    scan_listing_day_breakouts()

if __name__ == "__main__":
    main()

