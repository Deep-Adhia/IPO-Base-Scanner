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
import json
import pandas as pd
import numpy as np
import requests
import time
from datetime import datetime, timedelta
from dotenv import load_dotenv
import logging
from datetime import time as dt_time

try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except Exception:
    YFINANCE_AVAILABLE = False

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import from main scanner
import importlib.util
spec = importlib.util.spec_from_file_location("scanner", "streamlined-ipo-scanner.py")
scanner_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(scanner_module)

# Import version and logging utilities from main scanner
SCANNER_VERSION = getattr(scanner_module, 'SCANNER_VERSION', '2.1.0')
write_daily_log = getattr(scanner_module, 'write_daily_log', lambda *a, **k: None)

fetch_data = scanner_module.fetch_data
send_telegram = scanner_module.send_telegram
logger = scanner_module.logger
get_live_price = scanner_module.get_live_price

# Load environment
load_dotenv()


def _env_bool(key: str, default: bool) -> bool:
    v = os.getenv(key)
    if v is None:
        return default
    return str(v).strip().lower() in ("1", "true", "yes", "on")


def _env_float(key: str, default: float) -> float:
    try:
        return float(os.getenv(key, str(default)))
    except (TypeError, ValueError):
        return default


def _env_int(key: str, default: int) -> int:
    try:
        return int(os.getenv(key, str(default)))
    except (TypeError, ValueError):
        return default


# Quality mode: strict = only trades with full volume + freshness (default ON — "good trades only")
LISTING_STRICT_QUALITY = _env_bool("LISTING_STRICT_QUALITY", True)

# Configuration (stricter defaults when LISTING_STRICT_QUALITY is True)
LISTING_DATA_CSV = "ipo_listing_data.csv"
SIGNALS_CSV = "ipo_signals.csv"
POSITIONS_CSV = "ipo_positions.csv"
WATCHLIST_SIGNALS_CSV = "ipo_watchlist_signals.csv"
RECENT_IPO_CSV = "recent_ipo_symbols.csv"
PENDING_BREAKOUTS_FILE = "listing_pending_breakouts.json"
MIN_VOLUME_MULTIPLIER = _env_float(
    "LISTING_MIN_VOLUME_MULT", 1.8 if LISTING_STRICT_QUALITY else 1.5
)
MAX_ENTRY_ABOVE_HIGH_PCT = _env_float(
    "LISTING_MAX_ENTRY_ABOVE_HIGH_PCT", 3.5 if LISTING_STRICT_QUALITY else 5.0
)
MIN_RISK_REWARD = _env_float(
    "LISTING_MIN_RISK_REWARD", 1.25 if LISTING_STRICT_QUALITY else 1.0
)
STOP_LOSS_PCT = _env_float("LISTING_STOP_LOSS_PCT", 8.0)
MIN_VOLUME_VS_LISTING_DAY = _env_float(
    "LISTING_MIN_VOL_VS_LISTING", 1.0 if LISTING_STRICT_QUALITY else 1.2
)
# Reject listing-high breakouts when IPO is too old (strategy is "listing day" edge)
MAX_DAYS_SINCE_LISTING_FOR_BREAKOUT = _env_int(
    "LISTING_MAX_DAYS_SINCE_LISTING", 60 if LISTING_STRICT_QUALITY else 365
)
# If listing-day volume in CSV is 0 / missing, require this multiple of recent avg volume
MIN_VOL_MULT_WHEN_NO_LISTING_VOL = _env_float(
    "LISTING_MIN_VOL_MULT_WHEN_NO_LISTING_VOL", 2.0 if LISTING_STRICT_QUALITY else 1.5
)
LISTING_CONFIRMATION_MINUTES = _env_int(
    "LISTING_CONFIRMATION_MINUTES", 60
)
LISTING_MIN_LEADER_SCORE = _env_int(
    "LISTING_MIN_LEADER_SCORE", 5
)

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


def initialize_watchlist_data_csv():
    """Initialize dedicated watchlist CSV if it doesn't exist."""
    if not os.path.exists(WATCHLIST_SIGNALS_CSV):
        pd.DataFrame(columns=[
            "signal_id", "symbol", "signal_date", "signal_time", "status",
            "watch_level", "current_price", "distance_pct", "notes", "version", "scanner"
        ]).to_csv(WATCHLIST_SIGNALS_CSV, index=False, encoding='utf-8')


def load_pending_breakouts():
    """Load pending breakout confirmation state from disk."""
    if not os.path.exists(PENDING_BREAKOUTS_FILE):
        return {}
    try:
        with open(PENDING_BREAKOUTS_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            return data
    except Exception:
        pass
    return {}


def save_pending_breakouts(data):
    """Persist pending breakout confirmation state."""
    try:
        with open(PENDING_BREAKOUTS_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.warning(f"Could not save pending breakouts: {e}")


def _now_ist():
    # Keep one IST source for consistent timestamps
    return datetime.utcnow() + timedelta(hours=5, minutes=30)


def _market_is_open_ist():
    now = _now_ist()
    if now.weekday() >= 5:
        return False
    t = now.time()
    return dt_time(9, 15) <= t <= dt_time(15, 30)


def _get_intraday_bars(symbol, interval="5m", period="1d"):
    """Fetch intraday bars from yfinance for confirmation logic."""
    if not YFINANCE_AVAILABLE:
        return pd.DataFrame()
    try:
        df = yf.Ticker(f"{symbol}.NS").history(period=period, interval=interval)
        if df is None or df.empty:
            return pd.DataFrame()
        # Normalize columns to expected shape
        out = df.reset_index().rename(columns={
            "Datetime": "DATE", "Open": "OPEN", "High": "HIGH", "Low": "LOW", "Close": "CLOSE", "Volume": "VOLUME"
        })
        cols = [c for c in ["DATE", "OPEN", "HIGH", "LOW", "CLOSE", "VOLUME"] if c in out.columns]
        return out[cols]
    except Exception:
        return pd.DataFrame()


def _leader_score(entry_above_high_pct, volume_spike, risk_reward, current_price, listing_high, listing_range_pct):
    """
    Simple leader score (0-8 with extension penalty) for selection quality.
    """
    score = 0
    # Volume strength (0-2)
    if volume_spike >= 2.0:
        score += 2
    elif volume_spike >= 1.5:
        score += 1
    # Breakout strength (0-2)
    if current_price >= listing_high * 1.01:
        score += 2
    elif current_price >= listing_high:
        score += 1
    # Structure quality proxy (0-2) tighter range preferred
    if listing_range_pct <= 18:
        score += 2
    elif listing_range_pct <= 28:
        score += 1
    # Close/risk quality (0-2)
    if risk_reward >= 1.8:
        score += 2
    elif risk_reward >= 1.25:
        score += 1
    # Extension penalty (-2 to 0)
    if entry_above_high_pct > 4.0:
        score -= 2
    elif entry_above_high_pct > 2.5:
        score -= 1
    return score

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
            
            # Skip RE/SME right away to avoid noise
            if '-RE' in symbol or symbol.endswith('-SM') or 'RE1' in symbol:
                continue
                
            # Skip if listing date is today or in the future
            # NSE archives only update EOD, so we wait until tomorrow to fetch listing data
            if listing_date >= datetime.now().date():
                continue
            
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

def check_listing_day_breakout(symbol, listing_info, pending_breakouts=None):
    """Check if symbol has broken listing day high with volume"""
    try:
        listing_day_high = listing_info['listing_day_high']
        listing_day_low = listing_info['listing_day_low']
        listing_day_close = listing_info.get('listing_day_close', listing_day_high)  # Fallback to high if close not available
        listing_day_volume = float(listing_info.get('listing_day_volume', 0))  # CRITICAL: was missing, caused NameError
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
        rejection_reason = None
        volume_warnings = []  # Track volume-related warnings
        
        if current_high > listing_day_high:
            is_breakout = True
            signal_type = 'BREAKOUT'
            breakout_conditions.append(f"Price broke listing day high ({current_high:.2f} > {listing_day_high:.2f})")
        elif current_high >= listing_day_high * 0.95:
            # Watchlist condition: Within 5% of listing high
            is_breakout = True  # We set this to True to proceed with calculations, but mark type as WATCHLIST
            signal_type = 'WATCHLIST'
            breakout_conditions.append(f"Near Breakout: {current_high:.2f} is within 5% of {listing_day_high:.2f}")
            logger.info(f"👀 {symbol}: Detected as WATCHLIST candidate (High: {current_high:.2f}, Trigger: {listing_day_high:.2f})")
        else:
            rejection_reason = f"Price ({current_high:.2f}) below listing day high ({listing_day_high:.2f})"
        
        # Condition 2: Volume confirmation (now a warning, not a rejection)
        volume_spike = current_volume >= avg_volume * MIN_VOLUME_MULTIPLIER
        if volume_spike:
            breakout_conditions.append(f"Volume spike ({current_volume:,.0f} vs avg {avg_volume:,.0f})")
        elif is_breakout:
            # Price broke but volume insufficient - add warning instead of rejecting
            volume_warnings.append(f"Low volume spike: {current_volume:,.0f} vs avg {avg_volume:,.0f} (need {MIN_VOLUME_MULTIPLIER}x)")
        
        # Proceed if price broke listing day high OR is watchlist
        if is_breakout:
            # Calculate entry, stop loss, and target
            # For Watchlist, use Listing High as the hypothetical entry price
            if signal_type == 'WATCHLIST':
                entry_price = listing_day_high
            else:
                entry_price = current_price  # For confirmed breakout, use current price
            
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
            # Only apply for actual BREAKOUTs, not WATCHLIST
            if signal_type == 'BREAKOUT' and entry_above_high_pct > MAX_ENTRY_ABOVE_HIGH_PCT:
                rejection_reason = f"Entry ({entry_price:.2f}) is {entry_above_high_pct:.1f}% above listing high ({listing_day_high:.2f}) - too far from breakout level"
                logger.info(f"⏭️ Skipping {symbol}: {rejection_reason}")
                return None
            
            # Calculate days since listing (for display/information only - no filter)
            today_date = datetime.today().date()
            if isinstance(listing_date, str):
                listing_date_obj = pd.to_datetime(listing_date).date()
            elif hasattr(listing_date, 'date'):
                listing_date_obj = listing_date.date()
            else:
                listing_date_obj = listing_date
            
            days_since_listing = (today_date - listing_date_obj).days
            
            # Check volume vs listing day (warning in relaxed mode; strict mode enforces below)
            volume_vs_listing_day = current_volume / listing_day_volume if listing_day_volume > 0 else 0
            if volume_vs_listing_day < MIN_VOLUME_VS_LISTING_DAY:
                volume_warnings.append(f"Low volume vs listing day: {volume_vs_listing_day:.1f}x (need {MIN_VOLUME_VS_LISTING_DAY:.1f}x)")
                if signal_type == 'BREAKOUT' and not LISTING_STRICT_QUALITY:
                    logger.warning(f"⚠️ {symbol}: Low volume vs listing day ({volume_vs_listing_day:.1f}x, need {MIN_VOLUME_VS_LISTING_DAY:.1f}x) - sending signal with caution")

            # --- Strict quality gate (default): only persist / alert full-quality breakouts ---
            if LISTING_STRICT_QUALITY and signal_type == 'BREAKOUT':
                if days_since_listing > MAX_DAYS_SINCE_LISTING_FOR_BREAKOUT:
                    rejection_reason = (
                        f"Strict: {days_since_listing}d since listing (max {MAX_DAYS_SINCE_LISTING_FOR_BREAKOUT}d)"
                    )
                    logger.info(f"⏭️ Skipping {symbol}: {rejection_reason}")
                    return None
                if not volume_spike:
                    rejection_reason = (
                        f"Strict: volume spike required (current {current_volume:,.0f} vs avg {avg_volume:,.0f}, need {MIN_VOLUME_MULTIPLIER}x)"
                    )
                    logger.info(f"⏭️ Skipping {symbol}: {rejection_reason}")
                    return None
                if listing_day_volume > 0:
                    if volume_vs_listing_day < MIN_VOLUME_VS_LISTING_DAY:
                        rejection_reason = (
                            f"Strict: volume vs listing day {volume_vs_listing_day:.2f}x < {MIN_VOLUME_VS_LISTING_DAY}x"
                        )
                        logger.info(f"⏭️ Skipping {symbol}: {rejection_reason}")
                        return None
                else:
                    if current_volume < avg_volume * MIN_VOL_MULT_WHEN_NO_LISTING_VOL:
                        rejection_reason = (
                            f"Strict: listing day volume missing/0 — need current vol ≥ {MIN_VOL_MULT_WHEN_NO_LISTING_VOL}x avg ({avg_volume:,.0f})"
                        )
                        logger.info(f"⏭️ Skipping {symbol}: {rejection_reason}")
                        return None
                # Passed strict checks — treat as high-quality (no LOW_VOL grade)
                volume_warnings = []
            
            # Stop loss % below entry (configurable via LISTING_STOP_LOSS_PCT)
            stop_loss_pct = STOP_LOSS_PCT / 100.0
            
            # Calculate stop loss purely based on entry price percentage
            stop_loss = entry_price * (1 - stop_loss_pct)
            
            # Target calculation: Use entry price + percentage of listing range
            if entry_above_high_pct <= 2.0:
                target_multiplier = 1.0
            elif entry_above_high_pct <= 5.0:
                target_multiplier = 0.75
            else:
                target_multiplier = 0.5
            
            target_price = entry_price + (listing_range * target_multiplier)
            
            # Risk/Reward
            risk = entry_price - stop_loss
            reward = target_price - entry_price
            risk_reward = reward / risk if risk > 0 else 0
            
            # FILTER: Minimum risk/reward ratio (reward must be at least equal to risk)
            if risk_reward < MIN_RISK_REWARD:
                rejection_reason = f"Risk/Reward ratio ({risk_reward:.2f}) below minimum ({MIN_RISK_REWARD:.1f})"
                logger.info(f"⏭️ Skipping {symbol}: Risk/Reward ratio ({risk_reward:.2f}) is below minimum ({MIN_RISK_REWARD:.1f})")
                return None

            # Leader score gate (selection quality)
            leader_score = _leader_score(
                entry_above_high_pct=entry_above_high_pct,
                volume_spike=(current_volume / avg_volume) if avg_volume > 0 else 0,
                risk_reward=risk_reward,
                current_price=current_price,
                listing_high=listing_day_high,
                listing_range_pct=listing_range_pct
            )
            if signal_type == 'BREAKOUT' and leader_score < LISTING_MIN_LEADER_SCORE:
                logger.info(f"⏭️ Skipping {symbol}: Leader score {leader_score} < {LISTING_MIN_LEADER_SCORE}")
                return None

            # Intraday confirmation engine: PENDING -> CONFIRMED -> ENTER
            if signal_type == 'BREAKOUT' and LISTING_CONFIRMATION_MINUTES > 0 and _market_is_open_ist():
                if pending_breakouts is None:
                    pending_breakouts = {}
                state = pending_breakouts.get(symbol)
                now_ts = _now_ist()
                now_iso = now_ts.isoformat()
                if not state:
                    pending_breakouts[symbol] = {
                        "started_at": now_iso,
                        "breakout_level": float(listing_day_high),
                        "max_price_seen": float(current_price),
                        "last_price": float(current_price)
                    }
                    logger.info(f"⏳ {symbol}: breakout moved to PENDING for {LISTING_CONFIRMATION_MINUTES}m confirmation")
                    write_daily_log("listing_day", symbol, "PENDING_STARTED", {
                        "breakout_level": float(listing_day_high),
                        "confirm_minutes": LISTING_CONFIRMATION_MINUTES,
                        "price": round(float(current_price), 2),
                        "leader_score": int(leader_score),
                    })
                    return {
                        "symbol": symbol,
                        "type": "PENDING",
                        "current_price": round(current_price, 2),
                        "listing_day_high": listing_day_high,
                        "confirm_minutes": LISTING_CONFIRMATION_MINUTES,
                    }
                # update state
                started = datetime.fromisoformat(state["started_at"])
                state["max_price_seen"] = max(float(state.get("max_price_seen", current_price)), float(current_price))
                state["last_price"] = float(current_price)
                pending_breakouts[symbol] = state

                # rejection filter during observation
                max_seen = float(state["max_price_seen"])
                rejection_pct = ((max_seen - current_price) / max_seen * 100) if max_seen > 0 else 0
                if current_price < listing_day_high or rejection_pct > 2.5:
                    rej_reason = "below_breakout" if current_price < listing_day_high else "rejection_from_high"
                    pending_breakouts.pop(symbol, None)
                    logger.info(f"⏭️ {symbol}: pending confirmation rejected (price hold/rejection failed)")
                    write_daily_log("listing_day", symbol, "PENDING_REJECTED", {
                        "reason": rej_reason,
                        "current_price": round(float(current_price), 2),
                        "breakout_level": float(listing_day_high),
                        "rejection_pct": round(float(rejection_pct), 2),
                        "max_price_seen": round(float(max_seen), 2),
                        "elapsed_minutes": int((now_ts - started).total_seconds() // 60),
                    })
                    return None

                elapsed_min = int((now_ts - started).total_seconds() // 60)
                if elapsed_min < LISTING_CONFIRMATION_MINUTES:
                    logger.info(f"⏳ {symbol}: pending {elapsed_min}/{LISTING_CONFIRMATION_MINUTES}m confirmed hold")
                    return {
                        "symbol": symbol,
                        "type": "PENDING",
                        "current_price": round(current_price, 2),
                        "listing_day_high": listing_day_high,
                        "confirm_minutes": LISTING_CONFIRMATION_MINUTES,
                    }
                # Confirmed
                pending_breakouts.pop(symbol, None)
                breakout_conditions.append(f"Confirmed hold {LISTING_CONFIRMATION_MINUTES}m above breakout")
                write_daily_log("listing_day", symbol, "PENDING_CONFIRMED", {
                    "breakout_level": float(listing_day_high),
                    "confirm_minutes": LISTING_CONFIRMATION_MINUTES,
                    "entry_reference": round(float(current_price), 2),
                    "leader_score": int(leader_score),
                    "elapsed_minutes": elapsed_min,
                })
            
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
                'last_updated': last_updated,
                'volume_warnings': volume_warnings,
                'has_volume_caution': len(volume_warnings) > 0,
                'leader_score': int(leader_score),
                'type': signal_type  # 'BREAKOUT' or 'WATCHLIST'
            }
        
        # Log rejection reason if available
        if rejection_reason:
            logger.info(f"⏭️ {symbol}: Breakout rejected - {rejection_reason}")
        
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
    volume_warnings = breakout_data.get('volume_warnings', [])
    has_volume_caution = breakout_data.get('has_volume_caution', False)
    
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
• Stop Loss: ₹{stop:,.2f} ({STOP_LOSS_PCT:.0f}% below entry)
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
• Volume vs Listing Day: {volume_vs_listing_day:.1f}x {'✅' if not has_volume_caution else '⚠️'}
• Listing Day Range: {listing_range_pct:.1f}% (High-Low spread)
• {conditions}"""

    # Add volume caution section if warnings exist
    if has_volume_caution and volume_warnings:
        msg += f"""

⚠️ <b>VOLUME CAUTION:</b>
• {' | '.join(volume_warnings)}
• Signal sent for tracking - volume filters disabled for analysis
• Review performance after 1-2 months to validate volume filter effectiveness"""

    msg += f"""

⚠️ <b>Action Required:</b> Enter position - Listing day high broken!

🤖 Scanner v{SCANNER_VERSION} | {datetime.now().strftime('%Y-%m-%d %H:%M IST')}"""
    
    return msg

def format_watchlist_alert(breakout_data):
    """Format watchlist alert for near-breakout candidates"""
    symbol = breakout_data['symbol']
    current_price = breakout_data['current_price']
    listing_high = breakout_data['listing_day_high']
    listing_date = breakout_data['listing_date']
    price_source = breakout_data.get('price_source', 'Live')
    days_since = breakout_data.get('days_since_listing', 0)
    vol_spike = breakout_data.get('volume_spike', 0)
    
    # Calculate distance to breakout
    distance_amt = listing_high - current_price
    distance_pct = (distance_amt / listing_high * 100)
    
    # Format listing date
    if hasattr(listing_date, 'strftime'):
        listing_date_str = listing_date.strftime('%Y-%m-%d')
    else:
        listing_date_str = str(listing_date)
    
    # Volume trend assessment
    vol_status = "Building Up 🟢" if vol_spike > 1.0 else "Normal 🟡"
    if vol_spike > 2.0:
        vol_status = "Very High 💥"
    
    msg = f"""👀 <b>WATCHLIST ALERT: {symbol}</b>
    
🚀 <b>Approaching Breakout Level!</b>
The stock is within <b>{distance_pct:.1f}%</b> of its Listing Day High.

📊 <b>Status:</b>
• Current Price: ₹{current_price:,.2f} ({price_source})
• Breakout Level: ₹{listing_high:,.2f}
• Distance: {distance_pct:.1f}% away

📉 <b>Volume Trend:</b>
• Volume: {vol_spike:.1f}x avg ({vol_status})
• Pre-breakout buildup detected

📅 <b>Listing Context:</b>
• Listed on: {listing_date_str}
• Age: {days_since} days old

💡 <b>Actionable Advice:</b>
Keep {symbol} on your radar. A close above ₹{listing_high:.2f} with volume triggers a valid entry.

🤖 Scanner v{SCANNER_VERSION} | {datetime.now().strftime('%Y-%m-%d %H:%M IST')}
"""
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
            # Add notes column if it doesn't exist
            if 'notes' not in existing_signals.columns:
                existing_signals['notes'] = ''
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
        # Add note about volume caution if applicable
        volume_note = ""
        if breakout_data.get('has_volume_caution', False):
            volume_warnings = breakout_data.get('volume_warnings', [])
            volume_note = f"VOLUME_CAUTION: {'; '.join(volume_warnings)}"
        
        new_signal = {
            "signal_id": signal_id,
            "symbol": breakout_data['symbol'],
            "signal_date": today,
            "signal_time": datetime.now().strftime("%H:%M:%S"),
            "entry_price": breakout_data['entry_price'],
            "grade": "LISTING_BREAKOUT" + ("_LOW_VOL" if breakout_data.get('has_volume_caution', False) else ""),
            "score": 100 if not breakout_data.get('has_volume_caution', False) else 80,  # Lower score for low volume
            "stop_loss": breakout_data['stop_loss'],
            "target_price": breakout_data['target_price'],
            "status": "ACTIVE",
            "exit_date": "",
            "exit_price": 0,
            "pnl_pct": 0,
            "days_held": 0,
            "signal_type": "LISTING_DAY_BREAKOUT",
            "notes": volume_note,
            "version": SCANNER_VERSION,
            "scanner": "listing_day",
            "leader_score": int(breakout_data.get("leader_score", 0))
        }
        
        # Write to daily log
        write_daily_log("listing_day", breakout_data['symbol'], "BREAKOUT_SIGNAL", {
            "entry": breakout_data['entry_price'],
            "stop_loss": breakout_data['stop_loss'],
            "target": breakout_data['target_price'],
            "listing_high": breakout_data.get('listing_day_high', 0),
            "volume_caution": breakout_data.get('has_volume_caution', False)
        })
        
        # Append to CSV
        new_df = pd.DataFrame([new_signal])
        if existing_signals.empty:
            new_df.to_csv(SIGNALS_CSV, index=False, encoding='utf-8')
        else:
            # Ensure signal_type and notes columns exist in existing_signals
            if 'signal_type' not in existing_signals.columns:
                existing_signals['signal_type'] = 'UNKNOWN'
            if 'notes' not in existing_signals.columns:
                existing_signals['notes'] = ''
            pd.concat([existing_signals, new_df], ignore_index=True).to_csv(SIGNALS_CSV, index=False, encoding='utf-8')
        
        logger.info(f"✅ Saved breakout signal for {breakout_data['symbol']}")
        return True
    
    except Exception as e:
        logger.error(f"Error saving signal: {e}")
        return False

def save_watchlist_signal(breakout_data):
    """Save watchlist signal to prevent duplicate alerts"""
    try:
        today = datetime.now().date()
        # Use WATCHLIST prefix to distinguish from actual breakouts
        signal_id = f"WATCHLIST_{breakout_data['symbol']}_{today.strftime('%Y%m%d')}"
        
        # Check if signal already exists
        initialize_watchlist_data_csv()
        if os.path.exists(WATCHLIST_SIGNALS_CSV):
            existing_signals = pd.read_csv(WATCHLIST_SIGNALS_CSV, encoding='utf-8')
            if 'signal_id' in existing_signals.columns:
                # Check for same signal ID
                if signal_id in existing_signals['signal_id'].tolist():
                    logger.info(f"Watchlist alert already sent for {breakout_data['symbol']} today")
                    return False
        else:
            existing_signals = pd.DataFrame()
            
        distance_pct = ((breakout_data['listing_day_high'] - breakout_data['current_price']) / breakout_data['listing_day_high'] * 100) if breakout_data['listing_day_high'] > 0 else 0

        # Create new signal record (watchlist-only storage)
        new_signal = {
            "signal_id": signal_id,
            "symbol": breakout_data['symbol'],
            "signal_date": today,
            "signal_time": _now_ist().strftime("%H:%M:%S"),
            "status": "WATCH",
            "watch_level": breakout_data['listing_day_high'],
            "current_price": breakout_data['current_price'],
            "distance_pct": round(distance_pct, 2),
            "notes": "Within 5% of listing high",
            "version": SCANNER_VERSION,
            "scanner": "listing_day"
        }
        
        new_df = pd.DataFrame([new_signal])
        if existing_signals.empty:
            new_df.to_csv(WATCHLIST_SIGNALS_CSV, index=False, encoding='utf-8')
        else:
            pd.concat([existing_signals, new_df], ignore_index=True).to_csv(WATCHLIST_SIGNALS_CSV, index=False, encoding='utf-8')
            
        logger.info(f"✅ Saved watchlist signal for {breakout_data['symbol']}")
        return True
        
    except Exception as e:
        logger.error(f"Error saving watchlist signal: {e}")
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
    if LISTING_STRICT_QUALITY:
        logger.info(
            f"⚙️ Quality: STRICT — full volume + freshness, min R:R {MIN_RISK_REWARD}, "
            f"max {MAX_ENTRY_ABOVE_HIGH_PCT}% above listing high, max {MAX_DAYS_SINCE_LISTING_FOR_BREAKOUT}d since listing"
        )
    else:
        logger.info("⚙️ Quality: RELAXED — low-vol breakouts may be sent with caution")
    logger.info("=" * 60)
    
    # Initialize CSV
    initialize_listing_data_csv()
    initialize_watchlist_data_csv()
    
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
    pending_breakouts = load_pending_breakouts()
    
    for idx, listing_info in active_listings.iterrows():
        symbol = listing_info['symbol']
        logger.info(f"\n🔍 Checking {symbol}...")
        
        try:
            # Check for breakout
            breakout = check_listing_day_breakout(symbol, listing_info, pending_breakouts)
            
            if breakout:
                signal_type = breakout.get('type', 'BREAKOUT')
                
                if signal_type == 'BREAKOUT':
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
                        
                elif signal_type == 'WATCHLIST':
                    # Save watchlist signal (returns False if duplicate)
                    if save_watchlist_signal(breakout):
                        logger.info(f"👀 Sending WATCHLIST alert for {symbol}")
                        alert_msg = format_watchlist_alert(breakout)
                        send_telegram(alert_msg)
                elif signal_type == 'PENDING':
                    logger.info(f"⏳ {symbol}: pending confirmation in progress")
                
                # Small delay
                time.sleep(0.5)
            else:
                # Breakout was checked but not confirmed - rejection reason already logged in check_listing_day_breakout
                pass
            
            # Rate limiting
            time.sleep(0.3)
        
        except Exception as e:
            logger.error(f"Error scanning {symbol}: {e}")
            continue
    
    logger.info(f"\n{'='*60}")
    logger.info(f"✅ Scan complete: {breakouts_found} breakouts found")
    save_pending_breakouts(pending_breakouts)
    
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

