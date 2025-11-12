#!/usr/bin/env python3
"""
hourly_breakout_scanner.py

Hourly intraday breakout scanner for watchlist symbols:
- Reads symbols from watchlist.csv
- Fetches intraday data (5-minute candles)
- Detects real-time breakouts
- Sends alerts for immediate action
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

# Load environment
load_dotenv()

# Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)

# Configuration
WATCHLIST_CSV = "watchlist.csv"
SIGNALS_CSV = "ipo_signals.csv"
POSITIONS_CSV = "ipo_positions.csv"
INTRADAY_INTERVAL = "5minute"  # 1minute, 5minute, 15minute, 30minute, 60minute
LOOKBACK_DAYS = 5  # Days of intraday data to fetch
MIN_VOLUME_MULTIPLIER = 1.5  # Minimum volume spike for breakout

# Telegram configuration
BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

def send_telegram(msg):
    """Send Telegram notification"""
    if not BOT_TOKEN or not CHAT_ID:
        logger.warning(f"[Telegram disabled] BOT_TOKEN: {'SET' if BOT_TOKEN else 'MISSING'}, CHAT_ID: {'SET' if CHAT_ID else 'MISSING'}")
        return
    
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    
    try:
        response = requests.post(url, json={
            "chat_id": CHAT_ID, 
            "text": msg, 
            "parse_mode": "HTML",
            "disable_notification": False
        }, timeout=10)
        
        if response.status_code == 200:
            logger.info("✅ Telegram message sent successfully!")
        else:
            logger.error(f"❌ Telegram API error: {response.status_code} - {response.text}")
    except Exception as e:
        logger.error(f"❌ Telegram error: {e}")

def load_watchlist():
    """Load symbols from watchlist.csv"""
    try:
        if not os.path.exists(WATCHLIST_CSV):
            logger.warning(f"Watchlist file {WATCHLIST_CSV} not found. Creating template...")
            df = pd.DataFrame(columns=['symbol', 'added_date', 'notes', 'status'])
            df.to_csv(WATCHLIST_CSV, index=False, encoding='utf-8')
            return []
        
        df = pd.read_csv(WATCHLIST_CSV, encoding='utf-8')
        
        # Filter for active symbols only
        active_symbols = df[df['status'] == 'ACTIVE']['symbol'].tolist()
        
        # Remove comments and empty lines
        active_symbols = [s for s in active_symbols if not str(s).startswith('#') and str(s).strip()]
        
        logger.info(f"📋 Loaded {len(active_symbols)} active symbols from watchlist")
        return active_symbols
    
    except Exception as e:
        logger.error(f"Error loading watchlist: {e}")
        return []

def fetch_intraday_data_upstox(symbol, interval=INTRADAY_INTERVAL):
    """Fetch intraday data from Upstox API"""
    try:
        # Load IPO mappings
        if not os.path.exists('ipo_upstox_mapping.csv'):
            logger.warning("IPO mapping file not found")
            return None
        
        mapping_df = pd.read_csv('ipo_upstox_mapping.csv', encoding='utf-8')
        symbol_mapping = dict(zip(mapping_df['ipo_symbol'], mapping_df['instrument_key']))
        
        if symbol not in symbol_mapping:
            logger.warning(f"Symbol {symbol} not found in Upstox mapping")
            return None
        
        instrument_key = symbol_mapping[symbol]
        
        # Get Upstox credentials
        access_token = os.getenv('UPSTOX_ACCESS_TOKEN')
        if not access_token:
            logger.warning("Upstox access token not found")
            return None
        
        # Prepare API request
        headers = {
            'Accept': 'application/json',
            'Api-Version': '2.0',
            'Authorization': f'Bearer {access_token}'
        }
        
        # Calculate date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=LOOKBACK_DAYS)
        
        # Format dates for Upstox API
        from_str = start_date.strftime('%Y-%m-%d')
        to_str = end_date.strftime('%Y-%m-%d')
        
        # Upstox intraday API endpoint
        url = f"https://api.upstox.com/v2/historical-candle/{instrument_key}/{interval}/{to_str}/{from_str}"
        
        logger.info(f"🔄 Fetching intraday data for {symbol} ({interval})...")
        response = requests.get(url, headers=headers, timeout=30)
        
        # Handle rate limiting
        if response.status_code == 429:
            logger.warning(f"⚠️ Rate limited for {symbol}, waiting 1 second...")
            time.sleep(1)
            response = requests.get(url, headers=headers, timeout=30)
        
        if response.status_code == 200:
            data = response.json()
            if 'data' in data and 'candles' in data['data']:
                candles = data['data']['candles']
                if candles:
                    # Convert to DataFrame
                    df = pd.DataFrame(candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close'])
                    
                    # Handle timestamp conversion
                    try:
                        df['DATE'] = pd.to_datetime(df['timestamp'], unit='s')
                    except:
                        try:
                            df['DATE'] = pd.to_datetime(df['timestamp'])
                        except:
                            df['DATE'] = pd.to_datetime(df['timestamp'], format='%Y-%m-%d %H:%M:%S')
                    
                    df.columns = ['timestamp', 'OPEN', 'HIGH', 'LOW', 'CLOSE', 'VOLUME', 'IGNORE', 'DATE']
                    df = df[['DATE', 'OPEN', 'HIGH', 'LOW', 'CLOSE', 'VOLUME']]
                    df['LTP'] = df['CLOSE']
                    
                    # Sort by date
                    df = df.sort_values('DATE').reset_index(drop=True)
                    
                    logger.info(f"✅ Got {len(df)} intraday candles for {symbol}")
                    return df
        
        logger.warning(f"⚠️ No intraday data for {symbol}")
        return None
        
    except Exception as e:
        logger.warning(f"⚠️ Upstox API error for {symbol}: {e}")
        return None

def compute_rsi(close, period=14):
    """Calculate RSI"""
    delta = close.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def detect_intraday_breakout(df, symbol):
    """Detect intraday breakout patterns"""
    if df is None or len(df) < 20:
        return None
    
    try:
        # Get recent data (last 2 days of intraday candles)
        # For 5-minute candles: 2 days = ~96 candles (assuming 9:15 AM to 3:30 PM = 6.25 hours = 75 candles/day)
        recent_df = df.tail(150)  # Last 150 candles (~2 days)
        
        if len(recent_df) < 20:
            return None
        
        # Calculate recent high and low
        recent_high = recent_df['HIGH'].max()
        recent_low = recent_df['LOW'].min()
        recent_range = recent_high - recent_low
        
        # Current price
        current_price = recent_df['CLOSE'].iloc[-1]
        current_high = recent_df['HIGH'].iloc[-1]
        current_volume = recent_df['VOLUME'].iloc[-1]
        
        # Average volume
        avg_volume = recent_df['VOLUME'].mean()
        
        # Check for breakout above recent high
        breakout_threshold = recent_high * 0.995  # 0.5% below recent high to catch breakouts
        
        # Breakout conditions:
        # 1. Current high breaks above recent high
        # 2. Volume spike (at least 1.5x average)
        # 3. Price is above recent high
        is_breakout = False
        breakout_strength = 0
        
        if current_high > recent_high:
            is_breakout = True
            breakout_strength += 1
            logger.info(f"🔥 {symbol}: Price broke above recent high! ({current_high:.2f} > {recent_high:.2f})")
        
        if current_volume >= avg_volume * MIN_VOLUME_MULTIPLIER:
            breakout_strength += 1
            logger.info(f"📊 {symbol}: Volume spike detected! ({current_volume:,.0f} vs avg {avg_volume:,.0f})")
        
        # Calculate RSI for momentum confirmation
        rsi = compute_rsi(recent_df['CLOSE'])
        current_rsi = rsi.iloc[-1] if len(rsi) > 0 else 50
        
        if current_rsi > 60:  # Strong momentum
            breakout_strength += 1
        
        # Only trigger if we have a clear breakout
        if is_breakout and breakout_strength >= 2:
            # Calculate consolidation range
            consolidation_low = recent_df['LOW'].tail(50).min()  # Last 50 candles
            consolidation_high = recent_df['HIGH'].tail(50).max()
            consolidation_range = consolidation_high - consolidation_low
            
            # Entry price (current price or next candle open)
            entry_price = current_price
            
            # Stop loss (below consolidation low)
            stop_loss = consolidation_low * 0.98  # 2% below consolidation low
            
            # Target (based on consolidation range)
            target_price = consolidation_high + (consolidation_range * 0.5)  # 50% above consolidation high
            
            # Risk/Reward
            risk = entry_price - stop_loss
            reward = target_price - entry_price
            risk_reward = reward / risk if risk > 0 else 0
            
            return {
                'symbol': symbol,
                'entry_price': round(entry_price, 2),
                'stop_loss': round(stop_loss, 2),
                'target_price': round(target_price, 2),
                'current_price': round(current_price, 2),
                'recent_high': round(recent_high, 2),
                'consolidation_low': round(consolidation_low, 2),
                'consolidation_high': round(consolidation_high, 2),
                'volume_spike': round(current_volume / avg_volume, 2),
                'rsi': round(current_rsi, 2),
                'risk_reward': round(risk_reward, 2),
                'breakout_strength': breakout_strength,
                'timestamp': datetime.now()
            }
        
        return None
    
    except Exception as e:
        logger.error(f"Error detecting breakout for {symbol}: {e}")
        return None

def format_intraday_alert(breakout_data):
    """Format intraday breakout alert"""
    symbol = breakout_data['symbol']
    entry = breakout_data['entry_price']
    stop = breakout_data['stop_loss']
    target = breakout_data['target_price']
    current = breakout_data['current_price']
    rsi = breakout_data['rsi']
    vol_spike = breakout_data['volume_spike']
    rr = breakout_data['risk_reward']
    strength = breakout_data['breakout_strength']
    
    msg = f"""⚡ <b>INTRADAY BREAKOUT DETECTED</b>

📊 Symbol: <b>{symbol}</b>
💰 Current Price: ₹{current:,.2f}
🎯 Entry: ₹{entry:,.2f}
🛑 Stop Loss: ₹{stop:,.2f}
📈 Target: ₹{target:,.2f}
📊 Risk:Reward: 1:{rr:.1f}

📊 <b>Breakout Metrics:</b>
• RSI: {rsi:.1f}
• Volume Spike: {vol_spike:.1f}x
• Breakout Strength: {strength}/3

⏰ Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

⚠️ <b>Action Required:</b> Review immediately for entry opportunity"""
    
    return msg

def save_breakout_signal(breakout_data):
    """Save breakout signal to CSV"""
    try:
        # Check if signal already exists (same symbol, same day)
        today = datetime.now().date()
        signal_id = f"{breakout_data['symbol']}_INTRADAY_{today.strftime('%Y%m%d')}_{datetime.now().strftime('%H%M')}"
        
        # Load existing signals
        if os.path.exists(SIGNALS_CSV):
            existing_signals = pd.read_csv(SIGNALS_CSV, encoding='utf-8')
            # Check if we already have a signal for this symbol today
            today_signals = existing_signals[
                (existing_signals['symbol'] == breakout_data['symbol']) &
                (pd.to_datetime(existing_signals['signal_date']).dt.date == today)
            ]
            if len(today_signals) > 0:
                logger.info(f"Signal already exists for {breakout_data['symbol']} today")
                return False
        else:
            existing_signals = pd.DataFrame()
        
        # Create new signal
        new_signal = {
            "signal_id": signal_id,
            "symbol": breakout_data['symbol'],
            "signal_date": today,
            "entry_price": breakout_data['entry_price'],
            "grade": "INTRADAY",
            "score": breakout_data['breakout_strength'] * 10,
            "stop_loss": breakout_data['stop_loss'],
            "target_price": breakout_data['target_price'],
            "status": "ACTIVE",
            "exit_date": "",
            "exit_price": 0,
            "pnl_pct": 0,
            "days_held": 0
        }
        
        # Append to CSV
        new_df = pd.DataFrame([new_signal])
        if existing_signals.empty:
            new_df.to_csv(SIGNALS_CSV, index=False, encoding='utf-8')
        else:
            pd.concat([existing_signals, new_df], ignore_index=True).to_csv(SIGNALS_CSV, index=False, encoding='utf-8')
        
        logger.info(f"✅ Saved breakout signal for {breakout_data['symbol']}")
        return True
    
    except Exception as e:
        logger.error(f"Error saving signal: {e}")
        return False

def scan_watchlist():
    """Scan all symbols in watchlist for breakouts"""
    logger.info("🚀 Starting hourly intraday breakout scan...")
    logger.info("=" * 60)
    
    # Load watchlist
    symbols = load_watchlist()
    
    if not symbols:
        logger.warning("No active symbols in watchlist")
        return
    
    logger.info(f"📋 Scanning {len(symbols)} symbols...")
    
    breakouts_found = 0
    
    for i, symbol in enumerate(symbols, 1):
        logger.info(f"\n[{i}/{len(symbols)}] Scanning {symbol}...")
        
        try:
            # Fetch intraday data
            df = fetch_intraday_data_upstox(symbol)
            
            if df is None or df.empty:
                logger.warning(f"⚠️ No data for {symbol}")
                continue
            
            # Detect breakout
            breakout = detect_intraday_breakout(df, symbol)
            
            if breakout:
                logger.info(f"🎯 BREAKOUT DETECTED for {symbol}!")
                
                # Save signal
                if save_breakout_signal(breakout):
                    # Send alert
                    alert_msg = format_intraday_alert(breakout)
                    send_telegram(alert_msg)
                    breakouts_found += 1
                
                # Small delay to avoid rate limiting
                time.sleep(0.5)
            else:
                logger.info(f"✅ {symbol}: No breakout detected")
            
            # Rate limiting between symbols
            time.sleep(0.3)
        
        except Exception as e:
            logger.error(f"Error scanning {symbol}: {e}")
            continue
    
    logger.info(f"\n{'='*60}")
    logger.info(f"✅ Scan complete: {breakouts_found} breakouts found")
    
    # Send summary
    if breakouts_found > 0:
        summary = f"""📊 <b>Hourly Breakout Scan Summary</b>

🔍 Symbols Scanned: {len(symbols)}
🎯 Breakouts Found: {breakouts_found}
⏰ Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

{'🎉 New breakouts detected! Check alerts above.' if breakouts_found > 0 else '✅ No new breakouts at this time.'}"""
        send_telegram(summary)

def main():
    """Main function"""
    print("⚡ Hourly Intraday Breakout Scanner")
    print("=" * 60)
    print(f"📅 Date: {datetime.now().strftime('%Y-%m-%d')}")
    print(f"⏰ Time: {datetime.now().strftime('%H:%M:%S')}")
    print("=" * 60)
    
    scan_watchlist()

if __name__ == "__main__":
    main()

