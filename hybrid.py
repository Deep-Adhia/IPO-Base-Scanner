import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from jugaad_data.nse import stock_df
from nsetools import Nse
import time
import threading
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
import os
from dotenv import load_dotenv
load_dotenv()
# HYBRID CONFIGURATION - Best of Both Worlds
CONSOLIDATION_WINDOWS = [5,10, 20, 40, 80, 120]
CORRECT_LOWER, CORRECT_UPPER = 0.08, 0.35
MAX_DAYS = 200
LOOKAHEAD = 80
ABS_VOL_MIN = float(os.getenv("ABS_VOL_MIN", 1e6))
VOL_MULT = float(os.getenv("VOL_MULT", 1.2))
PARTIAL_TAKE_A_PLUS = float(os.getenv("PT_A_PLUS", 0.15))
PARTIAL_TAKE_B = float(os.getenv("PT_B", 0.12))
PARTIAL_TAKE_C = float(os.getenv("PT_C", 0.10))
STOP_PCT = float(os.getenv("STOP_PCT", 0.05))
IPO_YEARS_BACK = int(os.getenv("IPO_YEARS_BACK", "1"))
# DYNAMIC IPO FETCHING CONFIGURATION
# IPO_YEARS_BACK will be loaded from environment variables

# TIMEOUT AND RATE LIMITING CONFIGURATION
STOCK_DATA_TIMEOUT = 30  # seconds
REQUEST_DELAY = 0.5  # seconds between requests
MAX_RETRIES = 3
CHECKPOINT_INTERVAL = 100  # Save progress every N IPOs

# =============================================================================
# CONFIGURATION - Change these values to adjust IPO date range
# =============================================================================
# IPO_YEARS_BACK = 1   # Last 1 year IPOs (default)
# IPO_YEARS_BACK = 2   # Last 2 years IPOs  
# IPO_YEARS_BACK = 3   # Last 3 years IPOs
# =============================================================================

def fetch_stock_data_with_timeout(symbol, from_date, to_date, max_retries=MAX_RETRIES):
    """Fetch stock data with timeout and retry logic (Windows compatible)"""
    def fetch_data():
        """Function to fetch data in a separate thread"""
        try:
            df = stock_df(symbol, from_date=from_date, to_date=to_date, series="EQ")
            
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
                    print(f"⚠️ Missing required columns for {symbol}: {missing_columns}")
                    return None
                    
            return df
            
        except Exception as e:
            # If there's a column mismatch or other data issue, return None
            if "are in the [columns]" in str(e) or "column" in str(e).lower():
                return None
            raise e
    
    for attempt in range(max_retries):
        try:
            # Use ThreadPoolExecutor for timeout on Windows
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(fetch_data)
                df = future.result(timeout=STOCK_DATA_TIMEOUT)
                return df
        except (FutureTimeoutError, Exception) as e:
            if attempt < max_retries - 1:
                print(f"⚠️ Attempt {attempt + 1} failed for {symbol}: {str(e)[:100]}... Retrying...")
                time.sleep(REQUEST_DELAY * (attempt + 1))  # Exponential backoff
            else:
                print(f"❌ Failed to fetch data for {symbol} after {max_retries} attempts: {str(e)[:100]}...")
                return None
    return None

def fetch_ipo_data_from_nse(years_back=1):
    """Dynamic IPO data fetching using working fetch.py function"""
    try:
        print(f"🔄 Dynamically fetching IPO data for last {years_back} year(s)...")
        
        # Import the working fetch function
        import sys
        import os
        sys.path.append(os.path.dirname(os.path.abspath(__file__)))
        from fetch import fetch_recent_ipo_symbols
        
        # Get recent IPO symbols
        symbols_df = fetch_recent_ipo_symbols(years_back=years_back)
        
        if symbols_df is None:
            print("❌ Failed to fetch IPO symbols")
            return None
        
        # Convert to the format expected by hybrid.py
        ipo_df = pd.DataFrame({
            'Symbol': symbols_df['symbol'],
            'COMPANY NAME': symbols_df['company'],
            'DATE OF LISTING': pd.to_datetime(symbols_df['listing_date']),
            'SECURITY TYPE': 'EQ'
        })
        
        print(f"✅ Found {len(ipo_df)} recent IPOs")
        print(f"📋 Sample symbols: {ipo_df['Symbol'].head(10).tolist()}")
        
        return ipo_df
        
    except Exception as e:
        print(f"❌ Error fetching IPO data: {e}")
        return None

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

def should_exit_early(df, entry_idx, current_idx, entry_price, base_low):
    """Check for early exit conditions to prevent big losses"""
    current_price = df['CLOSE'].iat[current_idx]
    days_held = current_idx - entry_idx
    
    # Early stop if drops below base low within first 10 days
    if days_held <= 10 and current_price < base_low:
        return True, "Base Break"
    
    # Progressive stop loss - tighter stops for longer holds without profit
    if days_held > 30 and current_price < entry_price * 0.95:  # -5% after 30 days
        return True, "Time Stop"
    
    if days_held > 60 and current_price < entry_price * 0.92:  # -8% after 60 days
        return True, "Extended Time Stop"
    
    return False, None

def get_dynamic_stop_loss(entry_price, base_low, days_held):
    """Calculate dynamic stop loss based on time held - LESS AGGRESSIVE"""
    base_stop = base_low * 0.98  # 2% below base low
    # Much less aggressive: only tighten after 100+ days
    time_stop = entry_price * (0.97 - (max(0, days_held - 100) * 0.0001))
    
    return max(base_stop, time_stop)

# Environment variables already loaded at the top

# Dynamically fetch IPO data from NSE API
ipo_df = fetch_ipo_data_from_nse(years_back=IPO_YEARS_BACK)
if ipo_df is None:
    print("❌ Failed to fetch IPO data from NSE. Exiting...")
    exit(1)

print("🎯 HYBRID 'BEST OF BOTH WORLDS' IPO DETECTION")
print("=" * 60)
print(f"📊 Processing {len(ipo_df)} IPOs from last {IPO_YEARS_BACK} year(s)")

results = []
processed_signals = set()
processed_ipos = set()  # Track which IPOs have been processed
stats = {'total_processed': 0, 'rejected_losers': 0, 'signals_generated': 0, 'failed_fetches': 0, 'empty_data': 0}

total_ipos = len(ipo_df)
print(f"🔄 Starting processing of {total_ipos} IPOs...")

# Checkpoint system - save progress periodically
def save_checkpoint(results, stats, processed_signals, processed_ipos):
    """Save current progress to a checkpoint file"""
    checkpoint_data = {
        'results': results,
        'stats': stats,
        'processed_signals': list(processed_signals),
        'processed_ipos': list(processed_ipos)
    }
    import pickle
    with open('hybrid_checkpoint.pkl', 'wb') as f:
        pickle.dump(checkpoint_data, f)
    print(f"💾 Checkpoint saved: {len(results)} results, {len(processed_ipos)} IPOs processed")

# Load checkpoint if it exists
try:
    import pickle
    with open('hybrid_checkpoint.pkl', 'rb') as f:
        checkpoint_data = pickle.load(f)
        results = checkpoint_data['results']
        stats = checkpoint_data['stats']
        processed_signals = set(checkpoint_data['processed_signals'])
        processed_ipos = set(checkpoint_data.get('processed_ipos', []))
        print(f"🔄 Resuming from checkpoint: {len(results)} results, {len(processed_ipos)} IPOs processed")
except FileNotFoundError:
    print("🆕 Starting fresh - no checkpoint found")
except KeyError:
    # Handle old checkpoint format
    processed_ipos = set()
    print("🔄 Resuming from old checkpoint format - will track IPOs from now on")

for idx, row in ipo_df.iterrows():
    sym, rowdate = row['Symbol'].strip(), row['DATE OF LISTING'].date()
    
    # Skip if already processed
    if sym in processed_ipos:
        continue
    
    stats['total_processed'] += 1
    processed_ipos.add(sym)
    
    # Progress indicator
    if stats['total_processed'] % 50 == 0 or stats['total_processed'] == 1:
        progress_pct = (len(processed_ipos) / total_ipos) * 100
        print(f"📊 Progress: {len(processed_ipos)}/{total_ipos} ({progress_pct:.1f}%) - Processing {sym}")
    
    # Save checkpoint periodically
    if stats['total_processed'] % CHECKPOINT_INTERVAL == 0:
        save_checkpoint(results, stats, processed_signals, processed_ipos)
    
    # Fetch stock data with timeout
    df = fetch_stock_data_with_timeout(sym, rowdate, datetime.today().date())
    
    if df is None:
        stats['failed_fetches'] += 1
        continue
    
    if df.empty:
        stats['empty_data'] += 1
        continue
    
    # Add delay between requests to avoid rate limiting
    time.sleep(REQUEST_DELAY)
    df = df.sort_values("DATE")
    df['DATE'] = pd.to_datetime(df['DATE']).dt.date
    lhigh = df['HIGH'].iat[0]
    st = supertrend(df)
    symbol_signals = []
    for w in CONSOLIDATION_WINDOWS[::-1]:
        if len(df) < w:
            continue
        for i in range(w, min(len(df), MAX_DAYS)):
            perf = (df['CLOSE'].iat[i] - lhigh) / lhigh
            if not (CORRECT_LOWER <= -perf <= CORRECT_UPPER):
                continue
            low, high2 = df['LOW'][:i+1].tail(w).min(), df['HIGH'][:i+1].tail(w).max()
            prng = (high2 - low) / low * 100
            if prng > 60:
                continue
            avgv = df['VOLUME'][:i+1].tail(w).mean()
            vol_ok = ((df['VOLUME'].iat[i] >= 2.5*avgv and df['VOLUME'].iloc[i-2:i+1].sum() >= 4*avgv) or
                     df['VOLUME'].iat[i]/avgv >= VOL_MULT or
                     (df['VOLUME'].iloc[i-2:i+1].sum() * df['CLOSE'].iat[i]) >= ABS_VOL_MIN)
            if not vol_ok:
                continue
            for j in range(i+1, min(i+1+LOOKAHEAD, len(df))):
                if df['HIGH'].iat[j] > max(high2, lhigh*0.97):
                    # Follow-through filter: next day close > base high and volume ≥110% of breakout day
                    if j + 1 < len(df):
                        breakout_close = df['CLOSE'].iat[j]
                        breakout_volume = df['VOLUME'].iat[j]
                        next_day_close = df['CLOSE'].iat[j + 1]
                        next_day_volume = df['VOLUME'].iat[j + 1]
                        base_high = df['HIGH'][j-w+1:j+1].max()

                        if next_day_close <= base_high:
                            continue
                        if next_day_volume < 1.1 * breakout_volume:
                            continue

                    if reject_quick_losers(df, j, w, avgv):
                        stats['rejected_losers'] += 1
                        continue

                    score = compute_grade_hybrid(df, j, w, avgv)
                    grade = assign_grade(score)

                    # Enhanced B-grade filters with RSI and MACD
                    if grade == 'B' and not smart_b_filters(df, j, avgv):
                        continue

                    if grade == 'C' and not smart_c_filters(df, j, df['OPEN'].iat[j], w, avgv):
                        continue

                    if grade == 'D':
                        continue

                    entry_date = df['DATE'].iat[j]
                    signal_key = (sym, entry_date)
                    symbol_signals.append({
                        'key': signal_key,
                        'score': score,
                        'grade': grade,
                        'window': w,
                        'entry_idx': j,
                        'entry_date': entry_date,
                        'entry_price': df['OPEN'].iat[j],
                        'low': low
                    })
                    break
            break
    if symbol_signals:
        from collections import defaultdict
        date_groups = defaultdict(list)
        for signal in symbol_signals:
            date_groups[signal['entry_date']].append(signal)
        for entry_date, signals in date_groups.items():
            best_signal = max(signals, key=lambda x: (x['score'], 
                                                      1 if x['grade']=='A+' else 
                                                      0.8 if x['grade']=='B' else 
                                                      0.6 if x['grade']=='C' else 0))
            signal_key = best_signal['key']
            if signal_key in processed_signals:
                continue
            processed_signals.add(signal_key)
            stats['signals_generated'] += 1
            # Execute trade (with full labeling logic)
            j = best_signal['entry_idx']
            entry, j_open = best_signal['entry_date'], best_signal['entry_price']
            grade, score = best_signal['grade'], best_signal['score']
            partial_take = PARTIAL_TAKE_A_PLUS if grade == 'A+' else PARTIAL_TAKE_B if grade == 'B' else PARTIAL_TAKE_C
            low = best_signal['low']
            pos_size = 8 if grade == 'A+' else 4 if grade == 'B' else 1
            stop = low * (1 - STOP_PCT)
            part = False
            profit = 0
            exit_idx = None
            first_sell_price = None
            exit_label = "Full Exit"
            days_to_first_profit = None
            days_trailed_after_partial = None
            
            for k in range(j+1, len(df)):
                c, lo = df['CLOSE'].iat[k], df['LOW'].iat[k]
                days_held = k - j
                
                # Early Base-Break Exit (0-10 days)
                if days_held <= 10 and c < low:
                    profit = (c - j_open) / j_open
                    exit_label = "Early Base Break"
                    exit_idx = k
                    break
                
                # Tiered Time-Based Stops
                if days_held > 30 and c < j_open * 0.95:
                    profit = (c - j_open) / j_open
                    exit_label = "Time Stop -5%"
                    exit_idx = k
                    break
                
                if days_held > 60 and c < j_open * 0.92:
                    profit = (c - j_open) / j_open
                    exit_label = "Time Stop -8%"
                    exit_idx = k
                    break
                
                if not part and c >= j_open * (1 + partial_take):
                    profit += 0.5 * partial_take
                    part = True
                    stop = j_open
                    first_sell_price = c
                    exit_label = "Partial+Stop"
                    days_to_first_profit = days_held  # Track days to first profit
                
                if part:
                    stop = max(stop, st.iat[k])
                
                if lo <= stop or c <= stop:
                    if not part:
                        profit = (stop - j_open) / j_open
                        exit_label = "Full Exit"
                        first_sell_price = None
                    else:
                        profit += 0.5 * ((stop - j_open) / j_open)
                        days_trailed_after_partial = days_held - days_to_first_profit  # Track days trailed after partial
                    exit_idx = k
                    break
            else:
                final = df['CLOSE'].iat[-1]
                days_held = len(df) - 1 - j
                if not part:
                    profit = (final - j_open) / j_open
                    exit_label = "Full Hold"
                    first_sell_price = None
                else:
                    profit += 0.5 * ((final - j_open) / j_open)
                    exit_label = "Partial+Hold"
                    days_trailed_after_partial = days_held - days_to_first_profit  # Track days trailed after partial
                exit_idx = len(df) - 1
            days_in_trade = (df['DATE'].iat[exit_idx] - entry).days if exit_idx is not None else (df['DATE'].iat[-1] - entry).days
            absolute_return_pct = (df['CLOSE'].iat[exit_idx] - j_open) / j_open * 100
            system_return_pct = profit * 100
            results.append((
                sym, entry, j_open, grade, score, pos_size, df['DATE'].iat[exit_idx],
                df['CLOSE'].iat[exit_idx], system_return_pct, absolute_return_pct,
                days_in_trade, first_sell_price, exit_label, days_to_first_profit, days_trailed_after_partial
            ))

# Save final checkpoint
save_checkpoint(results, stats, processed_signals, processed_ipos)

# Save results
dfout = pd.DataFrame(results, columns=[
    'symbol', 'entry_date', 'entry_price', 'grade', 'score', 'position_size',
    'exit_date', 'exit_price', 'system_return_pct', 'absolute_return_pct', 'days_in_trade',
    'first_sell_price', 'exit_label', 'days_to_first_profit', 'days_trailed_after_partial'
])
print("🎯 HYBRID STRATEGY RESULTS:")
print("=" * 35)
print(f"Total IPOs processed: {stats['total_processed']}")
print(f"Failed data fetches: {stats['failed_fetches']}")
print(f"Empty data sets: {stats['empty_data']}")
print(f"Quick losers rejected: {stats['rejected_losers']}")
print(f"Final trades: {len(dfout)}")
if len(dfout) > 0:
    print(f"System return (avg): {dfout['system_return_pct'].mean():.2f}%")
    print(f"Win rate: {(dfout['system_return_pct'] > 0).mean()*100:.1f}%")
    print(f"Best trade: {dfout['system_return_pct'].max():.2f}%")
else:
    print("No trades generated - check data fetch issues")

# Grade distribution
if len(dfout) > 0:
    print(f"\n📊 Grade Distribution:")
    grade_counts = dfout['grade'].value_counts()
    for grade, count in grade_counts.items():
        avg_return = dfout[dfout['grade']==grade]['system_return_pct'].mean()
        win_rate = (dfout[dfout['grade']==grade]['system_return_pct'] > 0).mean() * 100
        print(f"{grade} Grade: {count} trades, {avg_return:.2f}% avg, {win_rate:.1f}% win rate")
    
    # New analytical columns
    print(f"\n⏱️ Timing Analysis:")
    partial_trades = dfout[dfout['days_to_first_profit'].notna()]
    if len(partial_trades) > 0:
        avg_days_to_profit = partial_trades['days_to_first_profit'].mean()
        print(f"Average days to first profit: {avg_days_to_profit:.1f} days")
        
        trailed_trades = partial_trades[partial_trades['days_trailed_after_partial'].notna()]
        if len(trailed_trades) > 0:
            avg_days_trailed = trailed_trades['days_trailed_after_partial'].mean()
            print(f"Average days trailed after partial: {avg_days_trailed:.1f} days")
        else:
            print("No trades were trailed after partial profit")
    else:
        print("No trades reached partial profit target")
    
    # Enhanced Exit Strategy Analysis
    print(f"\n🚪 Exit Strategy Analysis:")
    exit_counts = dfout['exit_label'].value_counts()
    for exit_type, count in exit_counts.items():
        avg_return = dfout[dfout['exit_label']==exit_type]['system_return_pct'].mean()
        avg_days = dfout[dfout['exit_label']==exit_type]['days_in_trade'].mean()
        print(f"{exit_type}: {count} trades, {avg_return:.2f}% avg return, {avg_days:.1f} days avg")
    
    # Losing Trades Analysis
    losing_trades = dfout[dfout['system_return_pct'] < 0]
    if len(losing_trades) > 0:
        print(f"\n📉 Losing Trades Analysis:")
        print(f"Total losing trades: {len(losing_trades)}")
        print(f"Average loss: {losing_trades['system_return_pct'].mean():.2f}%")
        print(f"Average days held: {losing_trades['days_in_trade'].mean():.1f} days")
        
        # Check how many were "Full Hold" vs other exits
        full_hold_losses = losing_trades[losing_trades['exit_label'] == 'Full Hold']
        if len(full_hold_losses) > 0:
            print(f"Full Hold losses: {len(full_hold_losses)} ({len(full_hold_losses)/len(losing_trades)*100:.1f}%)")
            print(f"Full Hold avg loss: {full_hold_losses['system_return_pct'].mean():.2f}%")
            print(f"Full Hold avg days: {full_hold_losses['days_in_trade'].mean():.1f} days")

# Save with backtest year range filename
filename = f"hybrid_test_{IPO_YEARS_BACK}_years.csv"

# Delete existing file with same name if it exists
if os.path.exists(filename):
    try:
        os.remove(filename)
        print(f"🗑️ Deleted existing file: {filename}")
    except Exception as e:
        print(f"⚠️ Could not delete existing file {filename}: {e}")

dfout.to_csv(filename, index=False)
print(f"\n✅ Results saved to: {filename}")
print("🎉 Hybrid strategy combines strict quality with expanded coverage!")