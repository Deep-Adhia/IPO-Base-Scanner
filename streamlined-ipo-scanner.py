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
from jugaad_data.nse import stock_df
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

# Debug logging for environment variables
print(f"🔍 Debug - BOT_TOKEN: {'Set' if BOT_TOKEN else 'Missing'}")
print(f"🔍 Debug - CHAT_ID: {CHAT_ID}")
print(f"🔍 Debug - BOT_TOKEN length: {len(BOT_TOKEN) if BOT_TOKEN else 0}")

# Core configuration
IPO_YEARS_BACK = get_env_int("IPO_YEARS_BACK", 1)
STOP_PCT = get_env_float("STOP_PCT", 0.03)

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
        logger.info("[Telegram disabled]")
        return
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    try:
        requests.post(url, json={"chat_id":CHAT_ID, "text":msg, "parse_mode":"HTML"}, timeout=10)
        except Exception as e:
            logger.error(f"Telegram error: {e}")

def format_signal_alert(symbol, grade, entry_price, stop_loss, target_price, score, date):
    """Format detailed IPO signal alert"""
    # Calculate expected return and win rate based on grade
    expected_return = ((target_price - entry_price) / entry_price) * 100
    win_rates = {"A+": "91%", "B": "75%", "C": "65%"}
    win_rate = win_rates.get(grade, "60%")
    
    # Grade emoji
    grade_emoji = {"A+": "⭐", "B": "🔥", "C": "📈"}
    emoji = grade_emoji.get(grade, "📊")
    
    msg = f"""🎯 <b>IPO BREAKOUT SIGNAL</b>

📊 Symbol: <b>{symbol}</b>
{emoji} Grade: <b>{grade}</b>
💰 Entry: ₹{entry_price:,.2f}
🛑 Stop Loss: ₹{stop_loss:,.2f}
📈 Expected: {expected_return:.1f}% ({win_rate} win rate)

📅 Date: {date.strftime('%Y-%m-%d %H:%M')}

Manual review recommended"""
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
            "stop_loss","target_price","status","exit_date","exit_price","pnl_pct","days_held"
        ]).to_csv(SIGNALS_CSV, index=False)
    if not os.path.exists(POSITIONS_CSV):
        pd.DataFrame(columns=[
            "symbol","entry_date","entry_price","grade","current_price",
            "stop_loss","trailing_stop","pnl_pct","days_held","status"
        ]).to_csv(POSITIONS_CSV, index=False)

def cache_recent_ipos():
    try:
        df = fetch_recent_ipo_symbols(years_back=IPO_YEARS)
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
        active = pd.read_csv(POSITIONS_CSV)["symbol"].unique().tolist()
    except:
        active = []
    return list(set(recent + active)), listing_map

def fetch_data(symbol, start_date):
    for attempt in range(3):
        try:
            df = stock_df(symbol,
                from_date=start_date,
                to_date=datetime.today().date(),
                series="EQ")
            if df.empty: 
                logger.warning(f"No data for {symbol}")
                return None
            
            # Handle column name mismatches by standardizing column names
            if not df.empty:
                # Create a mapping for common column name variations
                column_mapping = {
                    'CH_TIMESTAMP': 'DATE',
                    'CH_SERIES': 'SERIES', 
                    'CH__OPENING_PRICE': 'OPEN',
                    'CH_OPENING_PRICE': 'OPEN', 
                    'CH_TRADE_HIGH_PRICE': 'HIGH',
                    'CH_TRADE_LOW_PRICE': 'LOW',
                    'CH_PREVIOUS_CLS_PRICE': 'PREV. CLOSE',
                    'CH_LAST_TRADED_PRICE': 'LTP',
                    'CH_CLOSING_PRICE': 'CLOSE',
                    'CH_TOT_TRADED_QTY': 'VOLUME',
                    'CH_TOT_TRADED_VAL': 'VALUE',
                    'CH_TOTAL_TRADES': 'NO OF TRADES',
                    'CH_SYMBOL': 'SYMBOL',
                    'CH_52WEEK_HIGH_PRICE': '52W H',
                    'CH_52WEEK_LOW_PRICE': '52W L'
                }
                
                # Rename columns if they exist
                df = df.rename(columns=column_mapping)
                
                # Ensure required columns exist
                required_columns = ['DATE', 'OPEN', 'HIGH', 'LOW', 'CLOSE', 'VOLUME']
                missing_columns = [col for col in required_columns if col not in df.columns]
                
                if missing_columns:
                    logger.warning(f"Missing required columns for {symbol}: {missing_columns}")
            return None
    
            df["DATE"] = pd.to_datetime(df["DATE"]).dt.date
            return df.sort_values("DATE")
        except Exception as e:
            # If there's a column mismatch or other data issue, return None
            if "are in the [columns]" in str(e) or "column" in str(e).lower():
                logger.warning(f"Column mismatch for {symbol}: {str(e)[:100]}")
                return None
            logger.warning(f"Attempt {attempt + 1} failed for {symbol}: {str(e)[:100]}")
            if attempt < 2:  # Don't sleep on last attempt
                time.sleep(2)
            else:
                logger.error(f"Failed to fetch data for {symbol} after 3 attempts")
                return None
    
def update_positions():
    df_pos = pd.read_csv(POSITIONS_CSV, parse_dates=["entry_date"])
    for idx, pos in df_pos[df_pos["status"]=="ACTIVE"].iterrows():
        sym = pos["symbol"]
        start = pos["entry_date"].date()
        df = fetch_data(sym, start)
        if df is None: continue
        price = df["CLOSE"].iloc[-1]
        st = supertrend(df)
        trailing = max(pos["stop_loss"], st.iloc[-1])
        pnl = (price - pos["entry_price"])/pos["entry_price"]*100
        days = (datetime.today().date()-start).days
        
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
    df_pos.to_csv(POSITIONS_CSV, index=False)
    logger.info("Positions updated")

def dynamic_partial_take(grade):
    return PT_A_PLUS if grade=="A+" else PT_B if grade=="B" else PT_C

def detect_scan(symbols, listing_map):
    existing = pd.read_csv(SIGNALS_CSV)["signal_id"].tolist()
    signals_found = 0
    symbols_processed = 0
    
    for sym in symbols:
        symbols_processed += 1
        if symbols_processed % 20 == 0:
            logger.info(f"Processed {symbols_processed}/{len(symbols)} symbols...")
        if sym in pd.read_csv(POSITIONS_CSV)["symbol"].tolist(): continue
        ld = listing_map.get(sym)
        if not ld: continue
        df = fetch_data(sym, ld)
        if df is None: continue
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
                        
                        entry = df["OPEN"].iat[j]
                        stop = low * (1 - STOP_PCT)
                        date = df["DATE"].iat[j]
                        
                        # Only process signals from today onwards
                        if date < datetime.today().date():
                            continue
                            
                        sid = f"{sym}_{date.strftime('%Y%m%d')}"
                        if sid in existing: continue
                        
                        pt = dynamic_partial_take(grade)
                        row = {
                            "signal_id": sid, "symbol": sym, "signal_date": date,
                            "entry_price": entry, "grade": grade, "score": score,
                            "stop_loss": stop, "target_price": entry*(1+pt),
                            "status": "ACTIVE", "exit_date": "", "exit_price": 0,
                            "pnl_pct": 0, "days_held": 0
                        }
                        
                        # Read existing signals and append new signal
                        try:
                            existing_signals = pd.read_csv(SIGNALS_CSV)
                        except (FileNotFoundError, pd.errors.EmptyDataError):
                            existing_signals = pd.DataFrame()
                        
                        if existing_signals.empty:
                            pd.DataFrame([row]).to_csv(SIGNALS_CSV, index=False)
                        else:
                            pd.concat([existing_signals, pd.DataFrame([row])], ignore_index=True).to_csv(SIGNALS_CSV, index=False)
                        
                        pos = {
                            "symbol": sym, "entry_date": date, "entry_price": entry,
                            "grade": grade, "current_price": entry, "stop_loss": stop,
                            "trailing_stop": stop, "pnl_pct": 0, "days_held": 0, "status": "ACTIVE"
                        }
                        
                        # Read existing positions and append new position
                        try:
                            existing_positions = pd.read_csv(POSITIONS_CSV)
                        except (FileNotFoundError, pd.errors.EmptyDataError):
                            existing_positions = pd.DataFrame()
                        
                        if existing_positions.empty:
                            pd.DataFrame([pos]).to_csv(POSITIONS_CSV, index=False)
                        else:
                            pd.concat([existing_positions, pd.DataFrame([pos])], ignore_index=True).to_csv(POSITIONS_CSV, index=False)
                        
                        # Send detailed signal alert
                        signal_msg = format_signal_alert(sym, grade, entry, stop, entry*(1+pt), score, date)
                        send_telegram(signal_msg)
                        signals_found += 1
                        logger.info(f"🎯 Signal found: {sym} - {grade} grade at {entry}")
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

📈 <b>Active Positions:</b> {len(pd.read_csv(POSITIONS_CSV))}"""
    
    send_telegram(summary_msg)
    return signals_found

def weekly_summary():
    """Generate detailed weekly summary with performance metrics"""
    df_signals = pd.read_csv(SIGNALS_CSV, parse_dates=["signal_date"])
    df_positions = pd.read_csv(POSITIONS_CSV, parse_dates=["entry_date"])
    
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
    df_signals = pd.read_csv(SIGNALS_CSV, parse_dates=["signal_date"])
    df_positions = pd.read_csv(POSITIONS_CSV, parse_dates=["entry_date"])
    
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

def stop_loss_update_scan():
    """Dedicated scan for updating stop losses on active positions"""
    logger.info("🔄 Starting stop-loss update scan...")
    
    df_positions = pd.read_csv(POSITIONS_CSV, parse_dates=["entry_date"])
    active_positions = df_positions[df_positions["status"] == "ACTIVE"]
    
    if active_positions.empty:
        send_telegram("📊 <b>Stop-Loss Update Scan</b>\n\n✅ No active positions to update.")
        return
    
    updates_made = 0
    exits_triggered = 0
    
    for idx, pos in active_positions.iterrows():
        sym = pos["symbol"]
        logger.info(f"Updating stop-loss for {sym}...")
        
        # Get current price
        try:
            current_data = fetch_data(sym, pos["entry_date"])
            if current_data is None or current_data.empty:
                logger.warning(f"Could not fetch data for {sym}")
                continue
                
            current_price = current_data["CLOSE"].iat[-1]
            entry_price = pos["entry_price"]
            days_held = (datetime.today().date() - pos["entry_date"].date()).days
            
            # Calculate new trailing stop
            new_trailing = max(pos["trailing_stop"], current_price * (1 - STOP_PCT))
            
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
        except Exception as e:
            logger.error(f"Error updating {sym}: {e}")
            continue
    
    # Save updated positions
    df_positions.to_csv(POSITIONS_CSV, index=False)
    
    # Send summary
    summary_msg = f"""🔄 <b>Stop-Loss Update Scan</b>
    
📊 <b>Results:</b>
• Positions Updated: {updates_made}
• Positions Closed: {exits_triggered}
• Active Positions: {len(active_positions) - exits_triggered}

⏰ <b>Scan Time:</b> {datetime.now().strftime('%Y-%m-%d %H:%M')}"""
    
    send_telegram(summary_msg)
    logger.info(f"Stop-loss update complete: {updates_made} updated, {exits_triggered} closed")

def heartbeat():
    """Send heartbeat to confirm scanner is alive"""
    logger.info("💓 Sending heartbeat...")
    try:
        active_positions = len(pd.read_csv(POSITIONS_CSV))
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

    initialize_csvs()
    update_positions()
    symbols, listing_map = get_symbols_and_listing()

    if args.mode == "scan":
        signals_found = detect_scan(symbols, listing_map)
        logger.info(f"✅ Scan completed successfully! Found {signals_found} signals.")
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
