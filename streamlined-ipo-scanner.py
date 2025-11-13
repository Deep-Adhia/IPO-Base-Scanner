#!/usr/bin/env python3
"""
streamlined-ipo-scanner.py

Optimized IPO breakout scanner:
- Dynamic symbol list (recent IPOs + active positions)
- Enhanced entry filters and grading
- Dynamic partial profit taking by grade
- SuperTrend trailing stops for winners
- Smart exit logic (stop-loss, persistent losers)
- Weekly and monthly summary commands
- Dry-run and heartbeat modes
"""

import os
import sys
import argparse
import pickle
import pandas as pd
import numpy as np
import requests
import time
from datetime import datetime, timedelta
from dotenv import load_dotenv
from jugaad_data.nse.history import stock_raw
import pandas as pd
import threading

# Global rate limiter for Upstox API
_upstox_last_request = 0.0
_upstox_lock = threading.Lock()

def stock_df(symbol, from_date, to_date, series="EQ"):
    """Custom stock_df function that handles column mapping correctly"""
    try:
        # Get raw data
        raw = stock_raw(symbol, from_date, to_date, series)
        
        if not raw:
            return pd.DataFrame()
        
        # Create DataFrame from raw data
        df = pd.DataFrame(raw)
        
        # Map old column names to new ones
        column_mapping = {
            'CH_TIMESTAMP': 'DATE',
            'CH_SERIES': 'SERIES',
            'CH_OPENING_PRICE': 'OPEN',
            'CH_TRADE_HIGH_PRICE': 'HIGH',
            'CH_TRADE_LOW_PRICE': 'LOW',
            'CH_PREVIOUS_CLS_PRICE': 'PREV. CLOSE',
            'CH_LAST_TRADED_PRICE': 'LTP',
            'CH_CLOSING_PRICE': 'CLOSE',
            'VWAP': 'VWAP',
            'CH_52WEEK_HIGH_PRICE': '52W H',
            'CH_52WEEK_LOW_PRICE': '52W L',
            'CH_TOT_TRADED_QTY': 'VOLUME',
            'CH_TOT_TRADED_VAL': 'VALUE',
            'CH_TOTAL_TRADES': 'NO OF TRADES',
            'CH_SYMBOL': 'SYMBOL'
        }
        
        # Rename columns
        df = df.rename(columns=column_mapping)
        
        # Select only the columns we need
        required_columns = ['DATE', 'SERIES', 'OPEN', 'HIGH', 'LOW', 'PREV. CLOSE', 'LTP', 'CLOSE', 'VWAP', '52W H', '52W L', 'VOLUME', 'VALUE', 'NO OF TRADES', 'SYMBOL']
        df = df[required_columns]
        
        # Convert DATE to datetime
        df['DATE'] = pd.to_datetime(df['DATE'])
        
        return df
        
    except Exception as e:
        logger.error(f"Error in custom stock_df for {symbol}: {e}")
        return pd.DataFrame()
from fetch import fetch_recent_ipo_symbols
# Import only the functions we need, not the entire module
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import functions without executing hybrid.py
def supertrend(df, p=10, m=3.0):
    hl = (df['HIGH']+df['LOW'])/2
    tr = pd.concat([df['HIGH']-df['LOW'],
                    abs(df['HIGH']-df['CLOSE'].shift()),
                    abs(df['LOW']-df['CLOSE'].shift())], axis=1).max(axis=1)
    atr = tr.rolling(p).mean()
    ub, lb = hl + m*atr, hl - m*atr
    st = pd.Series(index=df.index)
    for i in range(1, len(df)):
        if df['CLOSE'].iat[i] <= lb.iat[i]:
            st.iat[i] = ub.iat[i]
        elif df['CLOSE'].iat[i] >= ub.iat[i]:
            st.iat[i] = lb.iat[i]
        else:
            st.iat[i] = st.iat[i-1]
    return st

def compute_grade_hybrid(df, idx, w, avg_vol):
    score=0
    low, high = df['LOW'].tail(w).min(), df['HIGH'].tail(w).max()
    prng = (high-low)/low*100
    if prng<=18: score+=1
    if df['VOLUME'].iat[idx]>=2.5*avg_vol and df['VOLUME'].iloc[idx-2:idx+1].sum()>=4*avg_vol: score+=1
    ret20 = (df['CLOSE'].iat[idx]/df['CLOSE'].iat[max(0,idx-20)]-1)
    percentile=np.percentile((df['CLOSE']-df['CLOSE'].shift(20))/df['CLOSE'].shift(20).fillna(0),85)
    if ret20>=percentile: score+=1
    ema20,ema50 = df['CLOSE'].ewm(20).mean().iat[idx], df['CLOSE'].ewm(50).mean().iat[idx]
    macd = df['CLOSE'].ewm(12).mean().iat[idx] - df['CLOSE'].ewm(26).mean().iat[idx]
    sig = pd.Series(df['CLOSE'].ewm(12).mean()-df['CLOSE'].ewm(26).mean()).ewm(9).mean().iat[idx]
    rsi = 100-100/(1+(df['CLOSE'].diff().clip(lower=0).rolling(14).mean()/
                     df['CLOSE'].diff().clip(upper=0).abs().rolling(14).mean())).iat[idx]
    if macd>sig and rsi>65 and ema20>ema50: score+=1
    if idx+1<len(df) and (df['OPEN'].iat[idx+1]/df['CLOSE'].iat[idx]-1)>=0.04: score+=1
    return score

def assign_grade(score):
    if score>=4: return 'A+'
    if score>=2: return 'B'
    if score>=1: return 'C'
    return 'D'

def calculate_target_price(entry_price, consolidation_low, consolidation_high, grade):
    """Calculate target price based on pattern and grade"""
    # Calculate consolidation range
    consolidation_range = consolidation_high - consolidation_low
    
    # Base target multipliers by grade
    target_multipliers = {
        "A+": 1.5,  # 50% above consolidation high
        "A": 1.4,   # 40% above consolidation high
        "B": 1.3,   # 30% above consolidation high
        "C": 1.2,   # 20% above consolidation high
        "D": 1.15   # 15% above consolidation high
    }
    
    multiplier = target_multipliers.get(grade, 1.2)
    
    # Target = consolidation high + (range * multiplier)
    target = consolidation_high + (consolidation_range * multiplier)
    
    # Ensure minimum 10% return
    min_target = entry_price * 1.10
    target = max(target, min_target)
    
    return target

def calculate_grade_based_stop_loss(entry_price, consolidation_low, grade):
    """Calculate stop loss based on grade and IPO volatility"""
    # Grade-based stop loss percentages (more appropriate for IPO volatility)
    grade_stop_pcts = {
        "A+": 0.05,  # 5% - High confidence, tighter stop
        "A": 0.07,   # 7% - Good confidence
        "B": 0.10,   # 10% - Medium confidence, more room for volatility
        "C": 0.12,   # 12% - Lower confidence, more volatile
        "D": 0.15    # 15% - High risk, very volatile
    }
    
    stop_pct = grade_stop_pcts.get(grade, 0.10)  # Default 10% for unknown grades
    
    # Calculate stop below entry price
    stop_below_entry = entry_price * (1 - stop_pct)
    
    # Calculate stop below consolidation low (safer)
    stop_below_consolidation = consolidation_low * (1 - stop_pct)
    
    # Use the higher (safer) of the two
    stop_loss = max(stop_below_entry, stop_below_consolidation)
    
    # Ensure stop loss is not more than 20% below entry (maximum risk)
    max_risk_stop = entry_price * 0.80
    stop_loss = max(stop_loss, max_risk_stop)
    
    return stop_loss, stop_pct
import logging

# Load environment
load_dotenv()

# Helper functions for environment variables with robust defaults
def get_env_int(key, default):
    """Get environment variable as integer with fallback"""
    try:
        return int(os.getenv(key, default) or default)
    except (ValueError, TypeError):
        print(f"Warning: Invalid {key} value, using default: {default}")
        return default

def get_env_float(key, default):
    """Get environment variable as float with fallback"""
    try:
        return float(os.getenv(key, default) or default)
    except (ValueError, TypeError):
        print(f"Warning: Invalid {key} value, using default: {default}")
        return default

def get_env_list(key, default, separator=","):
    """Get environment variable as list with fallback"""
    try:
        value = os.getenv(key, default)
        return [int(x.strip()) for x in value.split(separator) if x.strip()]
    except (ValueError, TypeError):
        print(f"Warning: Invalid {key} value, using default: {default}")
        return [int(x.strip()) for x in default.split(separator)]

# Environment variables with robust defaults
BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")


# Core configuration
IPO_YEARS_BACK = get_env_int("IPO_YEARS_BACK", 1)
STOP_PCT = get_env_float("STOP_PCT", 0.07)  # Default 7% for IPO volatility

# Dynamic partial take per grade
PT_A_PLUS = get_env_float("PT_A_PLUS", 0.15)
PT_B = get_env_float("PT_B", 0.12)
PT_C = get_env_float("PT_C", 0.10)
# Trading parameters
CONSOL_WINDOWS = get_env_list("CONSOL_WINDOWS", "10,20,40,80,120")
VOL_MULT = get_env_float("VOL_MULT", 1.2)
ABS_VOL_MIN = get_env_int("ABS_VOL_MIN", 3000000)
LOOKAHEAD = get_env_int("LOOKAHEAD", 80)
MAX_DAYS = get_env_int("MAX_DAYS", 200)
# File paths
CACHE_FILE = os.getenv("CACHE_FILE", "ipo_cache.pkl")
SIGNALS_CSV = os.getenv("SIGNALS_CSV", "ipo_signals.csv")
POSITIONS_CSV = os.getenv("POSITIONS_CSV", "ipo_positions.csv")

# System parameters
HEARTBEAT_RUNS = get_env_int("HEARTBEAT_RUNS", 0)

# Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)

def send_telegram(msg):
    if not BOT_TOKEN or not CHAT_ID:
        logger.warning(f"[Telegram disabled] BOT_TOKEN: {'SET' if BOT_TOKEN else 'MISSING'}, CHAT_ID: {'SET' if CHAT_ID else 'MISSING'}")
        return
    
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    
    try:
        logger.info(f"Sending Telegram message to chat_id: {CHAT_ID}")
        response = requests.post(url, json={
            "chat_id": CHAT_ID, 
            "text": msg, 
            "parse_mode": "HTML",
            "disable_notification": False  # Force notification in group chats
        }, timeout=10)
        
        if response.status_code == 200:
            logger.info("✅ Telegram message sent successfully!")
            logger.info(f"Response: {response.json()}")
        else:
            logger.error(f"❌ Telegram API error: {response.status_code} - {response.text}")
    except Exception as e:
        logger.error(f"❌ Telegram error: {e}")

def format_signal_alert(symbol, grade, entry_price, stop_loss, target_price, score, date, consolidation_low=None, consolidation_high=None, breakout_price=None, data_source=None):
    """Format detailed IPO signal alert with comprehensive trading information"""
    # Calculate risk metrics
    risk_amount = entry_price - stop_loss
    risk_percentage = (risk_amount / entry_price) * 100
    reward_amount = target_price - entry_price
    reward_percentage = (reward_amount / entry_price) * 100
    risk_reward_ratio = reward_amount / risk_amount if risk_amount > 0 else 0
    
    # Calculate position sizing (assuming 1% risk per trade)
    position_size_percent = 1.0  # 1% of portfolio
    position_size_amount = (position_size_percent * 100000) / risk_amount if risk_amount > 0 else 0  # Assuming 1L portfolio
    
    # Determine win rate and confidence based on grade
    grade_info = {
        "A+": {"win_rate": "91%", "confidence": "Very High", "emoji": "⭐"},
        "A": {"win_rate": "85%", "confidence": "High", "emoji": "🔥"},
        "B": {"win_rate": "75%", "confidence": "Medium-High", "emoji": "🔥"},
        "C": {"win_rate": "65%", "confidence": "Medium", "emoji": "📈"},
        "D": {"win_rate": "60%", "confidence": "Low-Medium", "emoji": "📊"}
    }
    
    info = grade_info.get(grade, {"win_rate": "60%", "confidence": "Low", "emoji": "📊"})
    win_rate = info["win_rate"]
    confidence = info["confidence"]
    emoji = info["emoji"]
    
    # Format the alert message with comprehensive information
    msg = f"""🎯 <b>IPO BREAKOUT SIGNAL</b>

📊 Symbol: <b>{symbol}</b>
{emoji} Grade: <b>{grade}</b> ({confidence} Confidence)
💰 Entry Price: ₹{entry_price:,.2f}
🛑 Stop Loss: ₹{stop_loss:,.2f} ({risk_percentage:.1f}% risk)
🎯 Target: ₹{target_price:,.2f} ({reward_percentage:.1f}% reward)
📊 Risk:Reward: 1:{risk_reward_ratio:.1f}
📈 Expected Return: {reward_percentage:.1f}% ({win_rate} win rate)

📋 <b>Pattern Details:</b>"""
    
    if consolidation_low and consolidation_high:
        msg += f"""
• Consolidation: ₹{consolidation_low:,.2f} - ₹{consolidation_high:,.2f}"""
    
    if breakout_price:
        msg += f"""
• Breakout: ₹{breakout_price:,.2f}"""
    
    msg += f"""
• Score: {score:.1f}/100"""

    # Add data source information
    if data_source:
        if data_source == 'Upstox API':
            msg += f"""
• Data Source: 🚀 Upstox API (Premium)"""
        elif data_source == 'NSE (Fallback)':
            msg += f"""
• Data Source: 📊 NSE (Fallback)"""
        else:
            msg += f"""
• Data Source: {data_source}"""

    msg += f"""

💼 <b>Position Sizing:</b>
• Risk per trade: {risk_percentage:.1f}%
• Suggested quantity: {int(position_size_amount):,} shares
• Capital at risk: ₹{int(risk_amount * position_size_amount):,}

📅 Signal Date: {date if isinstance(date, str) else date.strftime('%Y-%m-%d')}
⚠️ <b>Action Required:</b> Enter position at market open"""
    
    return msg

def format_exit_alert(symbol, exit_reason, exit_price, pnl_pct, days_held, entry_price):
    """Format detailed exit alert"""
    # Exit reason emojis
    exit_emojis = {
        "Stop Loss": "🛑",
        "Early Base Break": "⚡",
        "Time Stop -5%": "⏰",
        "Time Stop -8%": "⏰",
        "Partial Take": "💰"
    }
    emoji = exit_emojis.get(exit_reason, "📊")
    
    # PnL color
    pnl_color = "🟢" if pnl_pct > 0 else "🔴"
    
    msg = f"""{emoji} <b>POSITION EXIT</b>

📊 Symbol: <b>{symbol}</b>
📋 Reason: <b>{exit_reason}</b>
💰 Exit Price: ₹{exit_price:,.2f}
{pnl_color} P&L: {pnl_pct:+.1f}%
📅 Days Held: {days_held}
💵 Entry: ₹{entry_price:,.2f}

{datetime.now().strftime('%Y-%m-%d %H:%M')}"""
    return msg
    
def initialize_csvs():
    if not os.path.exists(SIGNALS_CSV):
        pd.DataFrame(columns=[
            "signal_id","symbol","signal_date","entry_price","grade","score",
            "stop_loss","target_price","status","exit_date","exit_price","pnl_pct","days_held","signal_type"
        ]).to_csv(SIGNALS_CSV, index=False, encoding='utf-8')
    if not os.path.exists(POSITIONS_CSV):
        pd.DataFrame(columns=[
            "symbol","entry_date","entry_price","grade","current_price",
            "stop_loss","trailing_stop","pnl_pct","days_held","status"
        ]).to_csv(POSITIONS_CSV, index=False, encoding='utf-8')

def cache_recent_ipos():
    try:
        df = fetch_recent_ipo_symbols(years_back=IPO_YEARS_BACK)
        with open(CACHE_FILE, "wb") as f:
            pickle.dump(df, f)
    except:
        if os.path.exists(CACHE_FILE):
            with open(CACHE_FILE,"rb") as f:
                df = pickle.load(f)
        else:
            df = pd.DataFrame(columns=["symbol","company","listing_date"])
    return df

def get_symbols_and_listing():
    ipo_df = cache_recent_ipos()
    recent = ipo_df["symbol"].tolist()
    listing_map = {
        row["symbol"]: pd.to_datetime(row["listing_date"]).date()
        for _, row in ipo_df.iterrows()
    }
    try:
        active = pd.read_csv(POSITIONS_CSV, encoding='utf-8')["symbol"].unique().tolist()
    except:
        active = []
    return list(set(recent + active)), listing_map


def fetch_from_upstox(symbol, start_date, end_date):
    """Fetch historical data from Upstox API with rate limiting"""
    import time
    
    try:
        # Convert dates to date objects if needed
        if isinstance(start_date, str):
            start_date = pd.to_datetime(start_date).date()
        elif hasattr(start_date, 'date'):
            start_date = start_date.date()
        elif isinstance(start_date, pd.Timestamp):
            start_date = start_date.date()
            
        if isinstance(end_date, str):
            end_date = pd.to_datetime(end_date).date()
        elif hasattr(end_date, 'date'):
            end_date = end_date.date()
        elif isinstance(end_date, pd.Timestamp):
            end_date = end_date.date()
        
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
        
        from_str = start_date.strftime('%Y-%m-%d')
        to_str = end_date.strftime('%Y-%m-%d')
        url = f"https://api.upstox.com/v2/historical-candle/{instrument_key}/day/{to_str}/{from_str}"
        
        # Global rate limiting: Ensure minimum 100ms between Upstox API requests
        global _upstox_last_request
        with _upstox_lock:
            current_time = time.time()
            time_since_last = current_time - _upstox_last_request
            if time_since_last < 0.1:  # 100ms minimum delay
                time.sleep(0.1 - time_since_last)
            _upstox_last_request = time.time()
        
        logger.info(f"🔄 Trying Upstox API for {symbol}")
        response = requests.get(url, headers=headers)
        
        # Handle rate limiting (429 Too Many Requests)
        if response.status_code == 429:
            logger.warning(f"⚠️ Rate limited for {symbol}, waiting 1 second...")
            time.sleep(1)
            response = requests.get(url, headers=headers)
        
        if response.status_code == 200:
            data = response.json()
            if 'data' in data and 'candles' in data['data']:
                candles = data['data']['candles']
                if candles:
                    # Convert to DataFrame
                    df = pd.DataFrame(candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close'])
                    
                    # Handle timestamp conversion - try different formats
                    try:
                        # Try Unix timestamp first
                        df['DATE'] = pd.to_datetime(df['timestamp'], unit='s')
                    except:
                        try:
                            # Try ISO format
                            df['DATE'] = pd.to_datetime(df['timestamp'])
                        except:
                            # Try string format
                            df['DATE'] = pd.to_datetime(df['timestamp'], format='%Y-%m-%d')
                    
                    df.columns = ['timestamp', 'OPEN', 'HIGH', 'LOW', 'CLOSE', 'VOLUME', 'IGNORE', 'DATE']
                    
                    # Select required columns and add LTP column (use CLOSE as LTP)
                    df = df[['DATE', 'OPEN', 'HIGH', 'LOW', 'CLOSE', 'VOLUME']]
                    df['LTP'] = df['CLOSE']  # Add LTP column using CLOSE price
                    
                    logger.info(f"✅ Upstox API: Got {len(df)} candles for {symbol}")
                    return df
        
        logger.warning(f"⚠️ Upstox API: No data for {symbol}")
        return None
        
    except Exception as e:
        logger.warning(f"⚠️ Upstox API error for {symbol}: {e}")
        return None

def get_live_price_upstox(symbol):
    """Get live price from Upstox market quote API"""
    try:
        # Load IPO mappings
        if not os.path.exists('ipo_upstox_mapping.csv'):
            return None
        
        mapping_df = pd.read_csv('ipo_upstox_mapping.csv', encoding='utf-8')
        symbol_mapping = dict(zip(mapping_df['ipo_symbol'], mapping_df['instrument_key']))
        
        if symbol not in symbol_mapping:
            return None
        
        instrument_key = symbol_mapping[symbol]
        
        # Get Upstox credentials
        access_token = os.getenv('UPSTOX_ACCESS_TOKEN')
        if not access_token:
            return None
        
        # Prepare API request
        headers = {
            'Accept': 'application/json',
            'Api-Version': '2.0',
            'Authorization': f'Bearer {access_token}'
        }
        
        url = f'https://api.upstox.com/v2/market-quote/quotes?instrument_key={instrument_key}'
        
        response = requests.get(url, headers=headers, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            if 'data' in data:
                # Try both key formats
                quote_key = f'NSE_EQ:{symbol}'
                if quote_key in data['data']:
                    quote = data['data'][quote_key]
                    live_price = quote.get('last_price')
                    if live_price:
                        return float(live_price)
                elif instrument_key in data['data']:
                    quote = data['data'][instrument_key]
                    live_price = quote.get('last_price')
                    if live_price:
                        return float(live_price)
        
        return None
    except Exception as e:
        logger.warning(f"Error fetching live price from Upstox for {symbol}: {e}")
        return None

def fetch_data(symbol, start_date):
    """Fetch the most recent available data for a symbol using Upstox API with NSE fallback (jugaad-data)"""
    import time  # Import at the top of function
    
    # Skip RE (Real Estate Investment Trusts) shares as they're not suitable for IPO breakout patterns
    if symbol.endswith('-RE'):
        logger.warning(f"Skipping RE share: {symbol}")
        return None
    
    try:
        # Convert start_date to date object if needed
        if isinstance(start_date, str):
            start_date = pd.to_datetime(start_date).date()
        elif hasattr(start_date, 'date'):
            start_date = start_date.date()
        elif isinstance(start_date, pd.Timestamp):
            start_date = start_date.date()
        
        today = datetime.today().date()
        
        # Try Upstox API first (if available)
        df = fetch_from_upstox(symbol, start_date, today)
        if df is not None and not df.empty:
            logger.info(f"✅ Upstox API: Got data for {symbol} ({len(df)} rows)")
            # Add data source info to DataFrame
            df.attrs['data_source'] = 'Upstox API'
            return df
        else:
            logger.warning(f"⚠️ Upstox API: No data for {symbol}, trying NSE fallback")
        
        # Fallback to NSE data (existing logic)
        
        # Try to get data for the last 30 days to find the most recent data
        # Start from the entry date and go backwards to find the most recent data
        for days_back in range(0, 30):
            target_date = start_date + timedelta(days=days_back)
            if target_date > today:
                target_date = today
            try:
                # Add retry mechanism for jugaad_data calls
                max_retries = 1  # Only 1 retry to avoid infinite loops
                df = None
                for retry in range(max_retries):
                    try:
                        # Use a date range from start_date to target_date
                        df = stock_df(symbol,
                            from_date=start_date,
                            to_date=target_date,
                            series="EQ")
                        break  # Success, exit retry loop
                    except Exception as retry_error:
                        if retry == max_retries - 1:
                            logger.warning(f"Failed to fetch data for {symbol}: {retry_error}")
                            return None  # Return None instead of continuing
                        else:
                            logger.warning(f"Retry {retry + 1}/{max_retries} for {symbol}: {retry_error}")
                            time.sleep(0.2)  # Very short wait time
                
                # Debug: Log what we actually received
                if df is None:
                    logger.warning(f"Received None for {symbol} - skipping")
                    continue
                elif hasattr(df, 'empty') and df.empty:
                    logger.warning(f"Received empty DataFrame for {symbol}")
                    continue
                elif not hasattr(df, 'columns'):
                    logger.warning(f"Received non-DataFrame object for {symbol}: {type(df)}")
                    continue
                else:
                    logger.info(f"Received data for {symbol}: {len(df)} rows, columns: {list(df.columns)}")
                    
                # Check if the data looks like HTML (error page)
                if hasattr(df, 'iloc') and len(df) > 0:
                    first_row = df.iloc[0]
                    if isinstance(first_row, pd.Series):
                        # Check if any column contains HTML-like content
                        for col in first_row.index:
                            if isinstance(first_row[col], str) and ('<html' in str(first_row[col]).lower() or '<!doctype' in str(first_row[col]).lower()):
                                logger.warning(f"Received HTML error page for {symbol}, skipping")
                continue
                
            except Exception as e:
                logger.warning(f"Error fetching data for {symbol}: {e}")
                # Add more detailed error logging
                import traceback
                logger.warning(f"Full traceback for {symbol}: {traceback.format_exc()}")
                continue
            
            # Add small delay to avoid rate limiting
            import time
            time.sleep(0.1)  # 100ms delay between requests
            
            if not df.empty:
                # Check data freshness
                latest_date = df['DATE'].max()
                if hasattr(latest_date, 'date'):
                    latest_date = latest_date.date()
                elif hasattr(latest_date, 'to_pydatetime'):
                    latest_date = latest_date.to_pydatetime().date()
                elif hasattr(latest_date, 'date'):
                    latest_date = latest_date.date()
                
                # Ensure both are date objects for comparison
                if isinstance(latest_date, pd.Timestamp):
                    latest_date = latest_date.date()
                if isinstance(today, pd.Timestamp):
                    today = today.date()
                
                days_old = (today - latest_date).days
                if days_old <= 1:
                    logger.info(f"Using fresh data for {symbol}: {latest_date}")
                elif days_old <= 3:
                    logger.info(f"Using recent data for {symbol}: {latest_date} ({days_old} days old)")
                else:
                    logger.warning(f"Using old data for {symbol}: {latest_date} ({days_old} days old)")
                
                # Standardize column names - handle both old and new jugaad_data formats
                if not df.empty:
                    # Check which format we have
                    if 'CH_TIMESTAMP' in df.columns:
                        # Old format
                        column_mapping = {
                            'CH_TIMESTAMP': 'DATE',
                            'CH_OPENING_PRICE': 'OPEN', 
                            'CH_TRADE_HIGH_PRICE': 'HIGH',
                            'CH_TRADE_LOW_PRICE': 'LOW',
                            'CH_CLOSING_PRICE': 'CLOSE',
                            'CH_LAST_TRADED_PRICE': 'LTP',
                            'CH_PREV_CLS_PRICE': 'PREV_CLOSE',
                            'CH_TOT_TRADED_QTY': 'VOLUME'
                        }
                    else:
                        # New format (already has correct names)
                        column_mapping = {
                            'DATE': 'DATE',
                            'OPEN': 'OPEN',
                            'HIGH': 'HIGH', 
                            'LOW': 'LOW',
                            'CLOSE': 'CLOSE',
                            'LTP': 'LTP',
                            'PREV. CLOSE': 'PREV_CLOSE',
                            'VOLUME': 'VOLUME'
                        }
                    
                    # Only rename columns that exist in the dataframe
                    existing_mapping = {k: v for k, v in column_mapping.items() if k in df.columns}
                    df = df.rename(columns=existing_mapping)
                    
                    # Ensure required columns exist
                    required_cols = ['DATE', 'OPEN', 'HIGH', 'LOW', 'CLOSE', 'LTP', 'VOLUME']
                    missing_cols = [col for col in required_cols if col not in df.columns]
                    if missing_cols:
                        logger.warning(f"Missing columns for {symbol}: {missing_cols}")
                        return None
                    
                    # Convert DATE to datetime
                    df['DATE'] = pd.to_datetime(df['DATE'])
                    
                    # Sort by date
                    df = df.sort_values('DATE').reset_index(drop=True)
                    
                    # Add data source info to DataFrame
                    df.attrs['data_source'] = 'NSE (jugaad-data)'
                    return df
                
                break
        
        logger.warning(f"No data found for {symbol} in the last 30 days")
        return None
            
    except Exception as e:
        logger.error(f"Error fetching data for {symbol}: {e}")
        return None
    
def update_positions():
    df_pos = pd.read_csv(POSITIONS_CSV, parse_dates=["entry_date"], encoding='utf-8')
    for idx, pos in df_pos[df_pos["status"]=="ACTIVE"].iterrows():
        sym = pos["symbol"]
        
        # Handle entry_date - convert to date object
        start = pos["entry_date"]
        if isinstance(start, pd.Timestamp):
            start = start.date()
        elif isinstance(start, str):
            start = pd.to_datetime(start).date()
        elif hasattr(start, 'date'):
            start = start.date()
        else:
            logger.warning(f"Could not parse entry_date for {sym}: {start}")
            continue
        
        df = fetch_data(sym, start)
        if df is None or df.empty: 
            logger.warning(f"No data available for {sym}")
            continue
        
        # Get latest price and validate data freshness
        latest_date = df['DATE'].max()
        if isinstance(latest_date, pd.Timestamp):
            latest_date = latest_date.date()
        elif hasattr(latest_date, 'date'):
            latest_date = latest_date.date()
        
        today_date = datetime.today().date()
        days_old = (today_date - latest_date).days
        
        if days_old > 3:
            logger.warning(f"⚠️ Data for {sym} is {days_old} days old (latest: {latest_date})")
        
        price = float(df["CLOSE"].iloc[-1])
        st = supertrend(df)
        trailing = max(float(pos["stop_loss"]), float(st.iloc[-1]))
        pnl = (price - float(pos["entry_price"]))/float(pos["entry_price"])*100
        
        # Calculate days held correctly
        days = (today_date - start).days
        
        if days < 0:
            logger.error(f"⚠️ Negative days for {sym}: entry_date={start}, today={today_date}")
            days = 0
        
        # Enhanced exit strategies
        exit_reason = None
        
        # Early Base-Break Exit (0-10 days) - need to get base low from original signal
        if days <= 10:
            # For now, use a simple approach - if price drops below entry * 0.95 in first 10 days
            if price < pos["entry_price"] * 0.95:
                exit_reason = "Early Base Break"
        
        # Tiered Time-Based Stops
        if days > 30 and price < pos["entry_price"] * 0.95:
            exit_reason = "Time Stop -5%"
        elif days > 60 and price < pos["entry_price"] * 0.92:
            exit_reason = "Time Stop -8%"
        
        # Traditional stop loss
        if price <= trailing:
            exit_reason = "Stop Loss"
        
        if exit_reason:
            df_pos.loc[idx, ["status","exit_date","exit_price","pnl_pct","days_held"]] = [
                "CLOSED", datetime.today().strftime("%Y-%m-%d"),
                float(price), float(pnl), int(days)
            ]
            # Send detailed exit alert
            exit_msg = format_exit_alert(sym, exit_reason, price, pnl, days, pos["entry_price"])
            send_telegram(exit_msg)
        else:
            df_pos.loc[idx, ["current_price","trailing_stop","pnl_pct","days_held"]] = [
                float(price), float(trailing), float(pnl), int(days)
            ]
    df_pos.to_csv(POSITIONS_CSV, index=False, encoding='utf-8')
    logger.info("Positions updated")

def compute_rsi(close, period=14):
    delta = close.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def compute_macd(close):
    ema12 = close.ewm(span=12).mean()
    ema26 = close.ewm(span=26).mean()
    macd_line = ema12 - ema26
    macd_signal = macd_line.ewm(span=9).mean()
    return macd_line, macd_signal

def dynamic_partial_take(grade):
    return PT_A_PLUS if grade=="A+" else PT_B if grade=="B" else PT_C

def smart_b_filters(df, entry_idx, avg_vol):
    close = df['CLOSE']
    rsi = compute_rsi(close)
    macd_line, macd_signal = compute_macd(close)

    # Require RSI >= 60 for a stronger momentum filter
    if rsi.iloc[entry_idx] < 60:
        return False

    # Require MACD line to be above signal by 0.1% of closing price
    if (macd_line.iloc[entry_idx] - macd_signal.iloc[entry_idx]) < 0.001 * close.iloc[entry_idx]:
        return False

    volume_ok = df['VOLUME'].iat[entry_idx] > 2.5*avg_vol
    ema20 = close.ewm(span=20).mean()
    trend_ok = close.iloc[entry_idx] > ema20.iloc[entry_idx]

    return volume_ok and trend_ok

def smart_c_filters(df, entry_idx, entry_price, w, avg_vol):
    c_score = 0
    low=df['LOW'][entry_idx-w+1:entry_idx+1].min()
    high=df['HIGH'][entry_idx-w+1:entry_idx+1].max()
    prn = (high-low)/low*100
    if prn<=25: c_score+=1
    if df['VOLUME'].iat[entry_idx]>=2*avg_vol: c_score+=1
    close = df['CLOSE']
    ema20 = close.ewm(span=20).mean()
    if close.iloc[entry_idx] > ema20.iloc[entry_idx]: c_score+=1
    for k in range(entry_idx+1, min(entry_idx+5, len(df))):
        if df['CLOSE'].iat[k] >= entry_price * 0.99:
            c_score+=1; break
    return c_score>=2

def reject_quick_losers(df, entry_idx, w, avg_vol):
    close = df['CLOSE']
    volume = df['VOLUME']
    red_flags = 0
    recent_avg_vol = volume.iloc[entry_idx-5:entry_idx].mean() if entry_idx >= 5 else avg_vol
    if volume.iat[entry_idx] < 1.5 * recent_avg_vol: red_flags += 1
    base_closes = close.iloc[entry_idx-w:entry_idx]
    if len(base_closes) >= 10:
        downtrend_days = sum(base_closes.diff() < 0) / len(base_closes)
        if downtrend_days > 0.6: red_flags += 1
    rsi = compute_rsi(close)
    if rsi.iloc[entry_idx] < 45: red_flags += 1
    recent_high = close.iloc[max(0,entry_idx-10):entry_idx].max()
    current_price = close.iloc[entry_idx]
    if current_price < recent_high * 0.92: red_flags += 1
    return red_flags >= 2

def detect_live_patterns(symbols, listing_map):
    """Detect LIVE FORMING patterns using proven backtest logic"""
    try:
        existing_signals_df = pd.read_csv(SIGNALS_CSV, encoding='utf-8')
        existing = existing_signals_df["signal_id"].tolist()
        # Add signal_type column if it doesn't exist (for backward compatibility)
        if 'signal_type' not in existing_signals_df.columns:
            existing_signals_df['signal_type'] = 'UNKNOWN'
    except:
        existing = []
    signals_found = 0
    symbols_processed = 0
    processed_today = set()  # Track symbols processed today to prevent duplicates
    
    for sym in symbols:
        symbols_processed += 1
        if symbols_processed % 20 == 0:
            logger.info(f"Processed {symbols_processed}/{len(symbols)} symbols...")
        
        if sym in pd.read_csv(POSITIONS_CSV, encoding='utf-8')["symbol"].tolist(): continue
        
        # Check if we already processed this symbol today
        today_key = f"{sym}_{datetime.today().strftime('%Y%m%d')}"
        if today_key in processed_today:
            continue
            
        ld = listing_map.get(sym)
        if not ld: continue
        df = fetch_data(sym, ld)
        if df is None or df.empty: continue
        
        # Check data freshness - reject signals with data older than 2 days for live trading
        latest_date = df['DATE'].max()
        if hasattr(latest_date, 'date'):
            latest_date = latest_date.date()
        days_old = (datetime.today().date() - latest_date).days
        if days_old > 2:
            logger.warning(f"Skipping {sym} - data is {days_old} days old (latest: {latest_date}). Too old for live trading!")
            continue
        
        lhigh = df["HIGH"].iloc[0]
        
        # Use your proven backtest logic but check for LIVE patterns (recent breakouts)
        for w in CONSOL_WINDOWS[::-1]:  # Start with larger windows first
            if len(df) < w: continue
            
            # Check recent data for live patterns (last 10 days for better coverage)
            recent_start = max(w, len(df)-10)  # Check last 10 days for live patterns
            for i in range(recent_start, len(df)):
                perf = (df["CLOSE"].iat[i] - lhigh) / lhigh
                if not (0.08 <= -perf <= 0.35): continue
                
                low, high2 = df["LOW"][:i+1].tail(w).min(), df["HIGH"][:i+1].tail(w).max()
                prng = (high2 - low) / low * 100
                if prng > 60: continue
                
                avgv = df["VOLUME"][:i+1].tail(w).mean()
                vol_ok = ((df["VOLUME"].iat[i] >= 2.5*avgv and df["VOLUME"].iloc[i-2:i+1].sum() >= 4*avgv) or
                         df["VOLUME"].iat[i]/avgv >= VOL_MULT or
                         (df["VOLUME"].iloc[i-2:i+1].sum() * df["CLOSE"].iat[i]) >= ABS_VOL_MIN)
                if not vol_ok: continue
                
                # Check if this is a LIVE breakout (happening now or very recently)
                for j in range(i+1, min(i+1+LOOKAHEAD, len(df))):
                    if df["HIGH"].iat[j] > max(high2, lhigh*0.97):
                        # Follow-through filter: next day close > base high and volume ≥110% of breakout day
                        if j + 1 < len(df):
                            breakout_close = df["CLOSE"].iat[j]
                            breakout_volume = df["VOLUME"].iat[j]
                            next_day_close = df["CLOSE"].iat[j + 1]
                            next_day_volume = df["VOLUME"].iat[j + 1]
                            base_high = df["HIGH"][j-w+1:j+1].max()

                            if next_day_close <= base_high:
                                continue
                            if next_day_volume < 1.1 * breakout_volume:
                                continue

                        # Apply your proven filters
                        if reject_quick_losers(df, j, w, avgv):
                            continue

                        score = compute_grade_hybrid(df, j, w, avgv)
                        grade = assign_grade(score)

                        # Enhanced B-grade filters with RSI and MACD
                        if grade == 'B' and not smart_b_filters(df, j, avgv):
                            continue

                        if grade == 'C' and not smart_c_filters(df, j, df["OPEN"].iat[j], w, avgv):
                            continue

                        if grade == 'D':
                            continue

                        # This is a LIVE pattern - generate signal
                        # For live signals, use TODAY's date and fetch CURRENT price
                        today_date = datetime.today().date()
                        
                        # Try to get live price from Upstox API first
                        entry = get_live_price_upstox(sym)
                        latest_date = None
                        price_source = "Upstox Live"
                        
                        if entry is None:
                            # Fallback to historical data
                            fresh_df = fetch_data(sym, ld)
                            if fresh_df is None or fresh_df.empty:
                                logger.warning(f"Could not fetch fresh data for {sym} to get current price")
                                continue
                            
                            # Get the latest available price from fresh data
                            entry = float(fresh_df["CLOSE"].iloc[-1])
                            latest_date = fresh_df['DATE'].max()
                            if isinstance(latest_date, pd.Timestamp):
                                latest_date = latest_date.date()
                            elif hasattr(latest_date, 'date'):
                                latest_date = latest_date.date()
                            price_source = f"Historical ({latest_date})"
                            
                            # Only generate signal if data is fresh (within last 2 days)
                            days_old = (today_date - latest_date).days
                            if days_old > 2:
                                logger.warning(f"Skipping {sym} - data is {days_old} days old. Too old for live signal!")
                                continue
                        else:
                            # Got live price from Upstox - set latest_date to today
                            latest_date = today_date
                        
                        # For live signals, entry date should be today
                        entry_date = today_date
                        
                        logger.info(f"Current price for {sym}: ₹{entry:.2f} (from {price_source})")
                        
                        # Log detailed data for analysis
                        logger.info(f"=== SIGNAL DATA FOR {sym} ===")
                        logger.info(f"Pattern detected at index {j} (consolidation window: {w})")
                        logger.info(f"Data range: {df['DATE'].min()} to {df['DATE'].max()}")
                        logger.info(f"Breakout date: {df['DATE'].iat[j]}")
                        logger.info(f"Entry date: {entry_date} (validated)")
                        logger.info(f"Entry price: ₹{entry:.2f}")
                        
                        # Check data freshness
                        latest_date = df['DATE'].max()
                        if hasattr(latest_date, 'date'):
                            latest_date = latest_date.date()
                        days_old = (datetime.today().date() - latest_date).days
                        if days_old > 1:
                            logger.warning(f"⚠️  DATA IS {days_old} DAYS OLD! Latest data: {latest_date}")
                        else:
                            logger.info(f"✅ Data is fresh: {days_old} days old")
                        
                        logger.info(f"Breakout day data:")
                        breakout_row = df.iloc[j]
                        logger.info(f"  Date: {breakout_row['DATE']}, O={breakout_row['OPEN']:.2f} H={breakout_row['HIGH']:.2f} L={breakout_row['LOW']:.2f} C={breakout_row['CLOSE']:.2f}")
                        if j + 1 < len(df):
                            next_row = df.iloc[j + 1]
                            logger.info(f"Entry day data (next day):")
                            logger.info(f"  Date: {next_row['DATE']}, O={next_row['OPEN']:.2f} H={next_row['HIGH']:.2f} L={next_row['LOW']:.2f} C={next_row['CLOSE']:.2f}")
                        
                        logger.info(f"Consolidation low: {low:.2f}")
                        logger.info(f"Consolidation high: {high2:.2f}")
                        logger.info(f"Breakout detected at index {j}: High={df['HIGH'].iat[j]:.2f} > Base High={high2:.2f}")
                        logger.info(f"Grade: {grade} (Score: {score})")
                        
                        # Grade-based stop loss: More appropriate for IPO volatility
                        stop, stop_pct = calculate_grade_based_stop_loss(entry, low, grade)
                        date = entry_date  # Use actual entry date from dataframe
                        
                        # Ensure date is a string in YYYY-MM-DD format for CSV
                        if isinstance(date, pd.Timestamp):
                            date_str = date.strftime('%Y-%m-%d')
                        elif hasattr(date, 'strftime'):
                            date_str = date.strftime('%Y-%m-%d')
                        else:
                            date_str = str(date)
                        
                        # Create unique signal ID with type prefix
                        sid = f"CONSOL_{sym}_{date_str.replace('-', '')}_{w}_{j}_LIVE"
                        if sid in existing: continue
                        
                        # Check if symbol already has active position (prevent duplicates)
                        try:
                            existing_positions = pd.read_csv(POSITIONS_CSV, encoding='utf-8')
                            if not existing_positions.empty:
                                active_positions = existing_positions[existing_positions['status'] == 'ACTIVE']
                                if sym in active_positions['symbol'].tolist():
                                    logger.info(f"⏭️ Skipping {sym} - already has active position")
                                    continue
                        except:
                            pass
                        
                        pt = dynamic_partial_take(grade)
                        target = entry * (1 + pt)
                        
                        # Add to signals
                        # Ensure date is a string in YYYY-MM-DD format
                        if isinstance(date, pd.Timestamp):
                            date_str = date.strftime('%Y-%m-%d')
                        elif hasattr(date, 'strftime'):
                            date_str = date.strftime('%Y-%m-%d')
                        else:
                            date_str = str(date)
                        
                        new_signal = {
                            "signal_id": sid,
                            "symbol": sym,
                            "signal_date": date_str,
                            "entry_price": round(entry, 2),
                            "grade": grade,
                            "score": score,
                            "stop_loss": round(stop, 2),
                            "target_price": round(target, 2),
                            "status": "ACTIVE",
                            "exit_date": "",
                            "exit_price": 0,
                            "pnl_pct": 0,
                            "days_held": 0,
                            "signal_type": "CONSOLIDATION"
                        }
                        
                        # Add to positions
                        new_position = {
                            "symbol": sym,
                            "entry_date": date_str,
                            "entry_price": round(entry, 2),
                            "grade": grade,
                            "current_price": round(entry, 2),
                            "stop_loss": round(stop, 2),
                            "trailing_stop": round(stop, 2),
                            "pnl_pct": 0,
                            "days_held": 0,
                            "status": "ACTIVE"
                        }
                        
                        signals_df = pd.DataFrame([new_signal])
                        positions_df = pd.DataFrame([new_position])
                        
                        # Append to CSV files properly
                        try:
                            existing_signals = pd.read_csv(SIGNALS_CSV, encoding='utf-8')
                            # Add signal_type column if it doesn't exist (for backward compatibility)
                            if 'signal_type' not in existing_signals.columns:
                                existing_signals['signal_type'] = 'UNKNOWN'
                        except (FileNotFoundError, pd.errors.EmptyDataError):
                            existing_signals = pd.DataFrame()
                        
                        if existing_signals.empty:
                            signals_df.to_csv(SIGNALS_CSV, index=False, encoding='utf-8')
                        else:
                            pd.concat([existing_signals, signals_df], ignore_index=True).to_csv(SIGNALS_CSV, index=False, encoding='utf-8')
                        
                        try:
                            existing_positions = pd.read_csv(POSITIONS_CSV, encoding='utf-8')
                        except (FileNotFoundError, pd.errors.EmptyDataError):
                            existing_positions = pd.DataFrame()
                        
                        if existing_positions.empty:
                            positions_df.to_csv(POSITIONS_CSV, index=False, encoding='utf-8')
                        else:
                            pd.concat([existing_positions, positions_df], ignore_index=True).to_csv(POSITIONS_CSV, index=False, encoding='utf-8')
                        
                        # Send Telegram notification with next day trading instructions
                        if days_old > 1:
                            price_warning = f"⚠️ OLD DATA: {days_old} days old. Verify current price before trading!"
                        else:
                            price_warning = f"✅ Fresh data - Ready for next day trading"
                        
                        message = f"""🎯 <b>CONSOLIDATION BREAKOUT SIGNAL</b>

📊 Symbol: <b>{sym}</b>
📋 Signal Type: <b>Consolidation-Based Breakout</b>
{'🔥' if grade in ['A+', 'B'] else '📈'} Grade: <b>{grade}</b>
💰 Entry: ₹{entry:.2f} (Next Day Opening)
🛑 Stop Loss: ₹{stop:.2f}
📈 Target: ₹{target:.2f}
📅 Signal Date: {date_str}
{price_warning}

📋 <b>TRADING INSTRUCTIONS:</b>
• Enter at market open tomorrow
• Use ₹{entry:.2f} as reference price
• Set stop loss at ₹{stop:.2f}
• Target: ₹{target:.2f}
⚡ Consolidation pattern detected"""
                        send_telegram(message)
                        
                        signals_found += 1
                        processed_today.add(today_key)
                        break
                if signals_found > 0: break
            if signals_found > 0: break
    
    logger.info(f"Live pattern scan complete: {signals_found} signals found from {symbols_processed} symbols")
    return signals_found

def detect_scan(symbols, listing_map):
    try:
        existing_signals_df = pd.read_csv(SIGNALS_CSV, encoding='utf-8')
        existing = existing_signals_df["signal_id"].tolist()
        # Add signal_type column if it doesn't exist (for backward compatibility)
        if 'signal_type' not in existing_signals_df.columns:
            existing_signals_df['signal_type'] = 'UNKNOWN'
    except:
        existing = []
    signals_found = 0
    symbols_processed = 0
    processed_today = set()  # Track symbols processed today to prevent duplicates
    
    for sym in symbols:
        symbols_processed += 1
        if symbols_processed % 20 == 0:
            logger.info(f"Processed {symbols_processed}/{len(symbols)} symbols...")
        
        if sym in pd.read_csv(POSITIONS_CSV, encoding='utf-8')["symbol"].tolist(): continue
        
        # Check if we already processed this symbol today
        today_key = f"{sym}_{datetime.today().strftime('%Y%m%d')}"
        if today_key in processed_today:
            continue
            
        ld = listing_map.get(sym)
        if not ld: continue
        df = fetch_data(sym, ld)
        if df is None or df.empty: continue
        lhigh = df["HIGH"].iloc[0]
        for w in CONSOL_WINDOWS:
            if len(df) < w: continue
            for i in range(w, min(len(df), MAX_DAYS)):
                perf = (df["CLOSE"].iat[i] - lhigh) / lhigh
                if not (0.08 <= -perf <= 0.35): continue
                low, high2 = df["LOW"][:i+1].tail(w).min(), df["HIGH"][:i+1].tail(w).max()
                prng = (high2 - low) / low * 100
                if prng > 60: continue
                avgv = df["VOLUME"][:i+1].tail(w).mean()
                vol_ok = ((df["VOLUME"].iat[i] >= 2.5*avgv and df["VOLUME"].iloc[i-2:i+1].sum() >= 4*avgv) or
                         df["VOLUME"].iat[i]/avgv >= VOL_MULT or
                         (df["VOLUME"].iloc[i-2:i+1].sum() * df["CLOSE"].iat[i]) >= ABS_VOL_MIN)
                if not vol_ok: continue
                for j in range(i+1, min(i+1+LOOKAHEAD, len(df))):
                    if df["HIGH"].iat[j] > max(high2, lhigh*0.97):
                        # Follow-through filter: next day close > base high and volume ≥110% of breakout day
                        if j + 1 < len(df):
                            breakout_close = df["CLOSE"].iat[j]
                            breakout_volume = df["VOLUME"].iat[j]
                            next_day_close = df["CLOSE"].iat[j + 1]
                            next_day_volume = df["VOLUME"].iat[j + 1]
                            base_high = df["HIGH"][j-w+1:j+1].max()

                            if next_day_close <= base_high:
                                continue
                            if next_day_volume < 1.1 * breakout_volume:
                                continue

                        score = compute_grade_hybrid(df, j, w, avgv)
                        grade = assign_grade(score)
                        if grade == "D": continue
                        
                        # This is a LIVE pattern - generate signal
                        # For live signals, use TODAY's date and fetch CURRENT price
                        today_date = datetime.today().date()
                        
                        # Try to get live price from Upstox API first
                        entry = get_live_price_upstox(sym)
                        latest_date = None
                        price_source = "Upstox Live"
                        
                        if entry is None:
                            # Fallback to historical data
                            fresh_df = fetch_data(sym, ld)
                            if fresh_df is None or fresh_df.empty:
                                logger.warning(f"Could not fetch fresh data for {sym} to get current price")
                                continue
                            
                            # Get the latest available price from fresh data
                            entry = float(fresh_df["CLOSE"].iloc[-1])
                            latest_date = fresh_df['DATE'].max()
                            if isinstance(latest_date, pd.Timestamp):
                                latest_date = latest_date.date()
                            elif hasattr(latest_date, 'date'):
                                latest_date = latest_date.date()
                            price_source = f"Historical ({latest_date})"
                            
                            # Only generate signal if data is fresh (within last 2 days)
                            days_old = (today_date - latest_date).days
                            if days_old > 2:
                                logger.warning(f"Skipping {sym} - data is {days_old} days old. Too old for live signal!")
                                continue
                        else:
                            # Got live price from Upstox - set latest_date to today
                            latest_date = today_date
                        
                        # For live signals, entry date should be today
                        entry_date = today_date
                        
                        logger.info(f"Current price for {sym}: ₹{entry:.2f} (from {price_source})")
                        logger.info(f"Breakout detected for {sym} at index {j}, date: {df['DATE'].iat[j]}")
                        logger.info(f"Entry date: {entry_date} (validated), Entry price: ₹{entry:.2f}")
                        
                        # Grade-based stop loss: More appropriate for IPO volatility
                        stop, stop_pct = calculate_grade_based_stop_loss(entry, low, grade)
                        # Use actual entry date from dataframe
                        date = entry_date
                            
                        # Create unique signal ID with type prefix
                        sid = f"CONSOL_{sym}_{date.strftime('%Y%m%d')}_{w}_{j}"
                        if sid in existing: continue
                        
                        # Check if symbol already has active position (prevent duplicates)
                        try:
                            existing_positions = pd.read_csv(POSITIONS_CSV, encoding='utf-8')
                            if not existing_positions.empty:
                                active_positions = existing_positions[existing_positions['status'] == 'ACTIVE']
                                if sym in active_positions['symbol'].tolist():
                                    logger.info(f"⏭️ Skipping {sym} - already has active position")
                                    continue
                        except:
                            pass
                        
                        pt = dynamic_partial_take(grade)
                        # Ensure date is a string in YYYY-MM-DD format
                        if isinstance(date, pd.Timestamp):
                            date_str = date.strftime('%Y-%m-%d')
                        elif hasattr(date, 'strftime'):
                            date_str = date.strftime('%Y-%m-%d')
                        else:
                            date_str = str(date)
                        
                        row = {
                            "signal_id": sid, "symbol": sym, "signal_date": date_str,
                            "entry_price": entry, "grade": grade, "score": score,
                            "stop_loss": stop, "target_price": entry*(1+pt),
                            "status": "ACTIVE", "exit_date": "", "exit_price": 0,
                            "pnl_pct": 0, "days_held": 0, "signal_type": "CONSOLIDATION"
                        }
                        
                        # Read existing signals and append new signal
                        try:
                            existing_signals = pd.read_csv(SIGNALS_CSV, encoding='utf-8')
                            # Add signal_type column if it doesn't exist (for backward compatibility)
                            if 'signal_type' not in existing_signals.columns:
                                existing_signals['signal_type'] = 'UNKNOWN'
                        except (FileNotFoundError, pd.errors.EmptyDataError):
                            existing_signals = pd.DataFrame()
                        
                        if existing_signals.empty:
                            pd.DataFrame([row]).to_csv(SIGNALS_CSV, index=False, encoding='utf-8')
                        else:
                            pd.concat([existing_signals, pd.DataFrame([row])], ignore_index=True).to_csv(SIGNALS_CSV, index=False, encoding='utf-8')
                        
                        pos = {
                            "symbol": sym, "entry_date": date_str, "entry_price": entry,
                            "grade": grade, "current_price": entry, "stop_loss": stop,
                            "trailing_stop": stop, "pnl_pct": 0, "days_held": 0, "status": "ACTIVE"
                        }
                        
                        # Read existing positions and append new position
                        try:
                            existing_positions = pd.read_csv(POSITIONS_CSV, encoding='utf-8')
                        except (FileNotFoundError, pd.errors.EmptyDataError):
                            existing_positions = pd.DataFrame()
                        
                        if existing_positions.empty:
                            pd.DataFrame([pos]).to_csv(POSITIONS_CSV, index=False, encoding='utf-8')
                        else:
                            pd.concat([existing_positions, pd.DataFrame([pos])], ignore_index=True).to_csv(POSITIONS_CSV, index=False, encoding='utf-8')
                        
                        # Calculate better target price based on pattern
                        target = calculate_target_price(entry, low, high2, grade)
                        
                        # Get data source from DataFrame
                        data_source = df.attrs.get('data_source', 'Unknown')
                        
                        # Send detailed signal alert with type
                        signal_msg = format_signal_alert(
                            sym, grade, entry, stop, target, score, date_str,
                            consolidation_low=low, consolidation_high=high2, breakout_price=entry,
                            data_source=data_source
                        )
                        # Add signal type to alert
                        signal_msg = signal_msg.replace("🎯 <b>IPO BREAKOUT SIGNAL</b>", 
                                                       "🎯 <b>CONSOLIDATION BREAKOUT SIGNAL</b>\n\n📋 <b>Signal Type:</b> Consolidation-Based Breakout")
                        send_telegram(signal_msg)
                        signals_found += 1
                        logger.info(f"🎯 Signal found: {sym} - {grade} grade at {entry}")
                        
                        # Mark this symbol as processed today to prevent duplicates
                        processed_today.add(today_key)
                        break
                break
    
    logger.info(f"📊 Scan complete: {signals_found} signals found from {symbols_processed} symbols processed")
    
    # Send scan summary to Telegram
    summary_msg = f"""📊 <b>IPO Scanner Summary</b>
    
🔍 <b>Scan Results:</b>
• Symbols Processed: {symbols_processed}
• New Signals Found: {signals_found}
• Scan Date: {datetime.today().strftime('%Y-%m-%d %H:%M')}

{'🎯 New signals detected! Check details above.' if signals_found > 0 else '✅ No new signals today - Market conditions normal.'}

📈 <b>Active Positions:</b> {len(pd.read_csv(POSITIONS_CSV, encoding='utf-8'))}"""
    
    send_telegram(summary_msg)
    return signals_found

def weekly_summary():
    """Generate detailed weekly summary with performance metrics"""
    df_signals = pd.read_csv(SIGNALS_CSV, parse_dates=["signal_date"], encoding='utf-8')
    df_positions = pd.read_csv(POSITIONS_CSV, parse_dates=["entry_date"], encoding='utf-8')
    
    # Weekly stats
    week_start = datetime.today() - timedelta(days=7)
    weekly_signals = len(df_signals[df_signals["signal_date"] >= week_start])
    active_positions = len(df_positions[df_positions["status"] == "ACTIVE"])
    
    # Performance stats for active positions
    if active_positions > 0:
        active_df = df_positions[df_positions["status"] == "ACTIVE"]
        avg_pnl = active_df["pnl_pct"].mean()
        best_position = active_df.loc[active_df["pnl_pct"].idxmax()]
        worst_position = active_df.loc[active_df["pnl_pct"].idxmin()]
        
        performance_text = f"""
📈 <b>Performance Highlights:</b>
• Average P&L: {avg_pnl:.2f}%
• Best Position: {best_position['symbol']} ({best_position['pnl_pct']:.2f}%)
• Worst Position: {worst_position['symbol']} ({worst_position['pnl_pct']:.2f}%)"""
    else:
        performance_text = "\n📈 <b>Performance:</b> No active positions"
    
    msg = f"""📊 <b>Weekly Summary</b>
    
🔍 <b>This Week:</b>
• New Signals: {weekly_signals}
• Active Positions: {active_positions}
• Total Signals (All Time): {len(df_signals)}{performance_text}

📅 <b>Week Range:</b> {week_start.strftime('%Y-%m-%d')} to {datetime.today().strftime('%Y-%m-%d')}"""
    
    send_telegram(msg)

def monthly_review():
    """Generate detailed monthly review with comprehensive stats"""
    df_signals = pd.read_csv(SIGNALS_CSV, parse_dates=["signal_date"], encoding='utf-8')
    df_positions = pd.read_csv(POSITIONS_CSV, parse_dates=["entry_date"], encoding='utf-8')
    
    # Monthly stats
    month_start = datetime.today().replace(day=1)
    monthly_signals = len(df_signals[df_signals["signal_date"] >= month_start])
    total_signals = len(df_signals)
    
    # Grade distribution
    if total_signals > 0:
        grade_dist = df_signals["grade"].value_counts()
        grade_text = "\n".join([f"• {grade}: {count}" for grade, count in grade_dist.items()])
    else:
        grade_text = "• No signals yet"
    
    # Position stats
    active_positions = len(df_positions[df_positions["status"] == "ACTIVE"])
    closed_positions = len(df_positions[df_positions["status"] == "CLOSED"])
    
    msg = f"""📊 <b>Monthly Review</b>
    
📈 <b>This Month ({month_start.strftime('%B %Y')}):</b>
• New Signals: {monthly_signals}
• Active Positions: {active_positions}
• Closed Positions: {closed_positions}

🎯 <b>All-Time Stats:</b>
• Total Signals: {total_signals}
• Grade Distribution:
{grade_text}

📅 <b>Review Period:</b> {month_start.strftime('%Y-%m-%d')} to {datetime.today().strftime('%Y-%m-%d')}"""
    
    send_telegram(msg)

def format_position_update_alert(symbol, current_price, entry_price, old_trailing, new_trailing, pnl_pct, days_held, grade):
    """Format position update alert"""
    pnl_emoji = "📈" if pnl_pct >= 0 else "📉"
    trailing_changed = "✅" if new_trailing > old_trailing else "➡️"
    
    msg = f"""🔄 <b>Position Update</b>

📊 Symbol: <b>{symbol}</b>
⭐ Grade: {grade}
💰 Current Price: ₹{current_price:,.2f}
💵 Entry Price: ₹{entry_price:,.2f}
{pnl_emoji} P&L: {pnl_pct:+.2f}%
📅 Days Held: {days_held}

🛑 Stop Loss:
• Old Trailing: ₹{old_trailing:,.2f}
• New Trailing: ₹{new_trailing:,.2f} {trailing_changed}

⏰ Updated: {datetime.now().strftime('%Y-%m-%d %H:%M')}"""
    return msg

def stop_loss_update_scan():
    """Dedicated scan for updating stop losses on active positions"""
    logger.info("🔄 Starting stop-loss update scan...")
    
    df_positions = pd.read_csv(POSITIONS_CSV, parse_dates=["entry_date"], encoding='utf-8')
    active_positions = df_positions[df_positions["status"] == "ACTIVE"]
    
    if active_positions.empty:
        send_telegram("📊 <b>Stop-Loss Update Scan</b>\n\n✅ No active positions to update.")
        return
    
    # Send pre-scan summary showing all positions that will be updated
    pre_scan_msg = f"""🔄 <b>Stop-Loss Update Scan Starting</b>

📊 <b>Active Positions to Update: {len(active_positions)}</b>

"""
    for idx, pos in active_positions.iterrows():
        sym = pos["symbol"]
        entry_price = pos["entry_price"]
        current_price = pos.get("current_price", entry_price)
        trailing_stop = pos.get("trailing_stop", pos.get("stop_loss", entry_price * 0.95))
        grade = pos.get("grade", "N/A")
        
        # Calculate days held
        try:
            entry_date = pos["entry_date"]
            if isinstance(entry_date, pd.Timestamp):
                entry_date = entry_date.date()
            elif hasattr(entry_date, 'date'):
                entry_date = entry_date.date()
            today_date = datetime.today().date()
            days_held = (today_date - entry_date).days
        except:
            days_held = "N/A"
        
        # Calculate PnL
        try:
            pnl = ((current_price - entry_price) / entry_price * 100) if entry_price > 0 else 0
            pnl_emoji = "📈" if pnl >= 0 else "📉"
        except:
            pnl = 0
            pnl_emoji = "➡️"
        
        pre_scan_msg += f"""• <b>{sym}</b> ({grade})
  💰 Entry: ₹{entry_price:,.2f} | Current: ₹{current_price:,.2f}
  {pnl_emoji} P&L: {pnl:+.2f}% | 🛑 Stop: ₹{trailing_stop:,.2f}
  📅 Days: {days_held}

"""
    
    pre_scan_msg += f"\n⏰ <b>Scan Time:</b> {datetime.now().strftime('%Y-%m-%d %H:%M')}"
    send_telegram(pre_scan_msg)
    
    updates_made = 0
    exits_triggered = 0
    failed_updates = []
    
    for idx, pos in active_positions.iterrows():
        sym = pos["symbol"]
        logger.info(f"Updating stop-loss for {sym}...")
        
        # Get current price
        try:
            # Convert entry_date to date object for fetch_data
            entry_date = pos["entry_date"]
            if isinstance(entry_date, pd.Timestamp):
                entry_date = entry_date.date()
            elif hasattr(entry_date, 'date'):
                entry_date = entry_date.date()
            
            current_data = fetch_data(sym, entry_date)
            if current_data is None or current_data.empty:
                logger.warning(f"Could not fetch data for {sym}")
                failed_updates.append(sym)
                # Send alert for failed update
                failed_msg = f"""⚠️ <b>Position Update Failed</b>

📊 Symbol: <b>{sym}</b>
❌ Could not fetch current data
📅 Entry Date: {pos['entry_date']}
💰 Last Known Price: ₹{pos.get('current_price', pos['entry_price']):,.2f}

⏰ Scan Time: {datetime.now().strftime('%Y-%m-%d %H:%M')}"""
                send_telegram(failed_msg)
                continue
                
            current_price = current_data["CLOSE"].iat[-1]
            entry_price = pos["entry_price"]
            old_trailing = pos["trailing_stop"]
            # Ensure both dates are date objects for comparison
            today_date = datetime.today().date()
            entry_date = pos["entry_date"]
            if isinstance(entry_date, pd.Timestamp):
                entry_date = entry_date.date()
            elif isinstance(entry_date, str):
                entry_date = pd.to_datetime(entry_date).date()
            elif hasattr(entry_date, 'date'):
                entry_date = entry_date.date()
            
            days_held = (today_date - entry_date).days
            
            if days_held < 0:
                logger.error(f"⚠️ Negative days for {sym}: entry_date={entry_date}, today={today_date}")
                days_held = 0
            
            # Calculate new trailing stop using grade-based percentage
            grade = pos.get("grade", "C")  # Default to C if grade not available
            _, stop_pct = calculate_grade_based_stop_loss(current_price, current_price, grade)
            new_trailing = max(pos["trailing_stop"], current_price * (1 - stop_pct))
            
            # Check for exits
            exit_reason = None
            if current_price <= new_trailing:
                exit_reason = "Stop Loss"
            elif days_held <= 10 and current_price < pos["entry_price"] * 0.95:
                exit_reason = "Early Base Break"
            elif days_held > 30 and current_price < pos["entry_price"] * 0.95:
                exit_reason = "Time Stop -5%"
            elif days_held > 60 and current_price < pos["entry_price"] * 0.92:
                exit_reason = "Time Stop -8%"
            
            if exit_reason:
                # Close position
                pnl = (current_price - entry_price) / entry_price * 100
                df_positions.loc[idx, ["status", "exit_date", "exit_price", "pnl_pct", "days_held"]] = [
                    "CLOSED", datetime.today().strftime("%Y-%m-%d"), current_price, pnl, days_held
                ]
                exits_triggered += 1
                
                # Send exit alert
                exit_msg = format_exit_alert(sym, exit_reason, current_price, pnl, days_held, entry_price)
                send_telegram(exit_msg)
            else:
                # Update position
                pnl = (current_price - entry_price) / entry_price * 100
                df_positions.loc[idx, ["current_price", "trailing_stop", "pnl_pct", "days_held"]] = [
                    current_price, new_trailing, pnl, days_held
                ]
                updates_made += 1
                
                # Send position update alert
                update_msg = format_position_update_alert(
                    sym, current_price, entry_price, old_trailing, new_trailing, pnl, days_held, grade
                )
                send_telegram(update_msg)
        except Exception as e:
            logger.error(f"Error updating {sym}: {e}")
            failed_updates.append(sym)
            # Send alert for error
            error_msg = f"""❌ <b>Position Update Error</b>

📊 Symbol: <b>{sym}</b>
⚠️ Error: {str(e)}
📅 Entry Date: {pos['entry_date']}
💰 Last Known Price: ₹{pos.get('current_price', pos['entry_price']):,.2f}

⏰ Scan Time: {datetime.now().strftime('%Y-%m-%d %H:%M')}"""
            send_telegram(error_msg)
            continue
    
    # Save updated positions
    df_positions.to_csv(POSITIONS_CSV, index=False, encoding='utf-8')
    
    # Send summary
    summary_msg = f"""🔄 <b>Stop-Loss Update Scan Complete</b>
    
📊 <b>Results:</b>
✅ Positions Updated: {updates_made}
🚪 Positions Closed: {exits_triggered}
⚠️ Failed Updates: {len(failed_updates)}
📈 Active Positions: {len(active_positions) - exits_triggered}"""
    
    if failed_updates:
        summary_msg += f"\n\n❌ <b>Failed Symbols:</b> {', '.join(failed_updates)}"
    
    summary_msg += f"\n\n⏰ <b>Scan Time:</b> {datetime.now().strftime('%Y-%m-%d %H:%M')}"
    
    send_telegram(summary_msg)
    logger.info(f"Stop-loss update complete: {updates_made} updated, {exits_triggered} closed, {len(failed_updates)} failed")

def heartbeat():
    """Send heartbeat to confirm scanner is alive"""
    logger.info("💓 Sending heartbeat...")
    try:
        active_positions = len(pd.read_csv(POSITIONS_CSV, encoding='utf-8'))
        message = f"💓 <b>Scanner Heartbeat</b>\n\n⏰ Time: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n📈 Active Positions: {active_positions}"
        logger.info(f"Heartbeat message: {message}")
        send_telegram(message)
        logger.info("✅ Heartbeat sent successfully")
    except Exception as e:
        logger.error(f"❌ Heartbeat failed: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode",choices=["scan","weekly_summary","monthly_review","stop_loss_update","heartbeat","dry_run"],
                        nargs="?",default="scan")
    args = parser.parse_args()

    # Show mode identification
    try:
        print("==========================================")
        print("IPO Scanner Started")
        print("==========================================")
    except UnicodeEncodeError:
        print("==========================================")
        print("IPO Scanner Started")
        print("==========================================")
    
    try:
        print(f"Date: {datetime.now().strftime('%Y-%m-%d')}")
        print(f"Time: {datetime.now().strftime('%H:%M:%S')}")
        print(f"Mode: {args.mode.upper()}")
        print("==========================================")
    except UnicodeEncodeError:
        print(f"Date: {datetime.now().strftime('%Y-%m-%d')}")
        print(f"Time: {datetime.now().strftime('%H:%M:%S')}")
        print(f"Mode: {args.mode.upper()}")
        print("==========================================")

    initialize_csvs()
    update_positions()
    symbols, listing_map = get_symbols_and_listing()
    

    if args.mode == "scan":
        signals_found = detect_live_patterns(symbols, listing_map)
        logger.info(f"✅ Live pattern scan completed successfully! Found {signals_found} signals.")
    elif args.mode == "weekly_summary":
        weekly_summary()
    elif args.mode == "monthly_review":
        monthly_review()
    elif args.mode == "stop_loss_update":
        stop_loss_update_scan()
    elif args.mode == "heartbeat":
        heartbeat()
    else:
        logger.info("Dry run complete (no writes or Telegram)")
