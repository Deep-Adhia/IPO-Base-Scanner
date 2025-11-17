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
        
        # Check if DataFrame is empty or missing required columns
        if df.empty:
            logger.info("📋 Watchlist is empty")
            return []
        
        if 'status' not in df.columns or 'symbol' not in df.columns:
            logger.warning("📋 Watchlist missing required columns (symbol, status)")
            return []
        
        # Filter for active symbols only
        active_symbols = df[df['status'] == 'ACTIVE']['symbol'].tolist()
        
        # Remove comments and empty lines
        active_symbols = [s for s in active_symbols if not str(s).startswith('#') and str(s).strip() and pd.notna(s)]
        
        logger.info(f"📋 Loaded {len(active_symbols)} active symbols from watchlist")
        return active_symbols
    
    except Exception as e:
        logger.error(f"Error loading watchlist: {e}")
        return []

# Try to import yfinance for intraday data
try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False

def fetch_intraday_data_yfinance(symbol, interval=INTRADAY_INTERVAL):
    """Fetch intraday data from yfinance API with rate limiting"""
    if not YFINANCE_AVAILABLE:
        return None
    
    try:
        # Rate limiting: 200ms minimum delay
        time.sleep(0.2)
        
        # Map interval to yfinance format
        interval_map = {
            '1minute': '1m',
            '5minute': '5m',
            '15minute': '15m',
            '30minute': '30m',
            '60minute': '1h'
        }
        yf_interval = interval_map.get(interval, '5m')
        
        # NSE symbols need .NS suffix
        ticker_symbol = f"{symbol}.NS"
        ticker = yf.Ticker(ticker_symbol)
        
        # Fetch intraday data (max 7 days for intraday)
        period = min(LOOKBACK_DAYS, 7)
        df = ticker.history(period=f"{period}d", interval=yf_interval)
        
        if df.empty:
            return None
        
        # Rename columns to match expected format
        df = df.reset_index()
        df.columns = ['DATE', 'OPEN', 'HIGH', 'LOW', 'CLOSE', 'VOLUME']
        df['LTP'] = df['CLOSE']
        
        # Ensure DATE is datetime (should already be from yfinance, but verify)
        if not pd.api.types.is_datetime64_any_dtype(df['DATE']):
            df['DATE'] = pd.to_datetime(df['DATE'])
        
        # Sort by date (ascending - oldest to newest)
        df = df.sort_values('DATE').reset_index(drop=True)
        
        logger.info(f"✅ Got {len(df)} intraday candles from yfinance for {symbol}")
        return df
        
    except Exception as e:
        logger.warning(f"⚠️ yfinance error for {symbol}: {e}")
        return None

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
                    
                    # Ensure DATE is datetime (should already be, but verify for consistency)
                    if not pd.api.types.is_datetime64_any_dtype(df['DATE']):
                        df['DATE'] = pd.to_datetime(df['DATE'])
                    
                    # Sort by date (ascending - oldest to newest)
                    df = df.sort_values('DATE').reset_index(drop=True)
                    
                    logger.info(f"✅ Got {len(df)} intraday candles for {symbol}")
                    return df
        
        logger.warning(f"⚠️ No intraday data for {symbol}")
        return None
        
    except Exception as e:
        logger.warning(f"⚠️ Upstox API error for {symbol}: {e}")
        return None

def fetch_intraday_data(symbol, interval=INTRADAY_INTERVAL):
    """
    Fetch intraday data from multiple sources with fallback:
    1. Upstox API (if available)
    2. yfinance (fallback)
    
    Returns DataFrame or None
    """
    # Try Upstox first
    df = fetch_intraday_data_upstox(symbol, interval)
    if df is not None and not df.empty:
        return df
    
    # Fallback to yfinance
    logger.info(f"⚠️ Upstox failed, trying yfinance for {symbol}...")
    df = fetch_intraday_data_yfinance(symbol, interval)
    if df is not None and not df.empty:
        return df
    
    logger.warning(f"⚠️ Could not fetch intraday data for {symbol} from any source")
        return None

def compute_rsi(close, period=14):
    """Calculate RSI"""
    delta = close.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def detect_intraday_breakout(df, symbol):
    """Detect intraday breakout patterns using LIVE prices for accurate detection"""
    if df is None or len(df) < 20:
        return None
    
    try:
        # Get recent data (last 2 days of intraday candles)
        # For 5-minute candles: 2 days = ~96 candles (assuming 9:15 AM to 3:30 PM = 6.25 hours = 75 candles/day)
        recent_df = df.tail(150)  # Last 150 candles (~2 days)
        
        if len(recent_df) < 20:
            return None
        
        # Calculate consolidation levels from historical data (exclude last candle for accurate range)
        historical_df = recent_df.iloc[:-1] if len(recent_df) > 1 else recent_df
        
        # Calculate recent high and low from historical data (excluding current candle)
        recent_high = historical_df['HIGH'].max() if len(historical_df) > 0 else recent_df['HIGH'].max()
        recent_low = historical_df['LOW'].min() if len(historical_df) > 0 else recent_df['LOW'].min()
        
        # Calculate consolidation range (last 50 candles, excluding current)
        consolidation_df = historical_df.tail(50) if len(historical_df) >= 50 else historical_df
        consolidation_low = consolidation_df['LOW'].min() if len(consolidation_df) > 0 else recent_low
        consolidation_high = consolidation_df['HIGH'].max() if len(consolidation_df) > 0 else recent_high
        
        # Get historical data for volume/RSI calculations
        current_volume = recent_df['VOLUME'].iloc[-1]
        avg_volume = historical_df['VOLUME'].mean() if len(historical_df) > 0 else current_volume
        
        # CRITICAL: Get LIVE price for accurate breakout detection
        live_price = None
        live_source = "Historical"
        try:
            # Import get_live_price from main scanner
            import importlib.util
            spec = importlib.util.spec_from_file_location("scanner", "streamlined-ipo-scanner.py")
            scanner_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(scanner_module)
            get_live_price = scanner_module.get_live_price
            
            live_price, live_source = get_live_price(symbol)
            if live_price is not None and live_price > 0:
                logger.info(f"✅ Using live price for {symbol} breakout detection: ₹{live_price:.2f} ({live_source})")
        except Exception as e:
            logger.debug(f"Could not get live price for {symbol}: {e}")
        
        # Use live price if available, otherwise use latest historical close
        if live_price is not None:
            current_price = live_price
            current_high = live_price  # For breakout detection, use live price as current high
        else:
            current_price = float(recent_df['CLOSE'].iloc[-1])
            current_high = float(recent_df['HIGH'].iloc[-1])
            live_source = "Historical"  # Ensure source is set to Historical when using historical data
            logger.warning(f"⚠️ Using historical price for {symbol}: ₹{current_price:.2f}")
        
        # Breakout conditions:
        # 1. Current price/high breaks above recent high (using LIVE price if available)
        # 2. Volume spike (at least 1.5x average)
        # 3. RSI momentum confirmation
        is_breakout = False
        breakout_strength = 0
        
        # Check if price breaks above recent high
        if current_high > recent_high:
            is_breakout = True
            breakout_strength += 1
            logger.info(f"🔥 {symbol}: Price broke above recent high! ({current_high:.2f} > {recent_high:.2f}) [Using: {live_source}]")
        
        # Volume confirmation
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
            consolidation_range = consolidation_high - consolidation_low
            
            # Entry price: Use LIVE price if available, otherwise current price
            entry_price = current_price
            
            # Stop loss (below consolidation low) - use proper calculation
            # Use the lower (more conservative) stop loss: 2% below consolidation low OR 5% of range below, whichever is LOWER
            stop_loss_1 = consolidation_low * 0.98  # 2% below consolidation low
            stop_loss_2 = consolidation_low - (consolidation_range * 0.05)  # 5% of range below
            stop_loss = min(stop_loss_1, stop_loss_2)  # Use the lower (more conservative) stop
            
            # Target (based on consolidation range) - add 50% of range above consolidation high
            target_price = consolidation_high + (consolidation_range * 0.5)
            
            # Risk/Reward
            risk = entry_price - stop_loss
            reward = target_price - entry_price
            risk_reward = reward / risk if risk > 0 else 0
            
            logger.info(f"📊 {symbol} Breakout Levels:")
            logger.info(f"   Consolidation: ₹{consolidation_low:.2f} - ₹{consolidation_high:.2f}")
            logger.info(f"   Recent High: ₹{recent_high:.2f}")
            logger.info(f"   Entry: ₹{entry_price:.2f} ({live_source})")
            logger.info(f"   Stop Loss: ₹{stop_loss:.2f}")
            logger.info(f"   Target: ₹{target_price:.2f}")
            logger.info(f"   Risk:Reward: 1:{risk_reward:.2f}")
            
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
                'price_source': live_source,
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
    price_source = breakout_data.get('price_source', 'Historical')
    
    # Add emoji for price source
    source_emojis = {
        'upstox': '🚀',
        'yfinance': '📈',
        'jugaad': '📊',
        'Historical': '📊'
    }
    emoji = source_emojis.get(price_source.lower(), '💰')
    
    msg = f"""⚡ <b>INTRADAY BREAKOUT DETECTED</b>

📊 Symbol: <b>{symbol}</b>
💰 Current Price: ₹{current:,.2f} ({emoji} {price_source})
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
            # Fetch intraday data (tries Upstox first, then yfinance)
            df = fetch_intraday_data(symbol)
            
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

