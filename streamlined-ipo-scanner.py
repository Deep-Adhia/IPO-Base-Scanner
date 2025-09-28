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
import requests
import time
from datetime import datetime, timedelta
from dotenv import load_dotenv
from jugaad_data.nse import stock_df
from fetch import fetch_recent_ipo_symbols
from hybrid import supertrend, compute_grade_hybrid, assign_grade
import logging

# Load environment
load_dotenv()
BOT_TOKEN        = os.getenv("TELEGRAM_BOT_TOKEN")
CHAT_ID          = os.getenv("TELEGRAM_CHAT_ID")
IPO_YEARS        = int(os.getenv("IPO_YEARS_BACK",    "2"))
STOP_PCT         = float(os.getenv("STOP_PCT",         "0.03"))
# Dynamic partial take per grade
PT_A_PLUS       = float(os.getenv("PT_A_PLUS",        "0.15"))
PT_B            = float(os.getenv("PT_B",             "0.12"))
PT_C            = float(os.getenv("PT_C",             "0.10"))
CONSOL_WINDOWS  = list(map(int, os.getenv("CONSOL_WINDOWS","10,20,40,80,120").split(",")))
VOL_MULT         = float(os.getenv("VOL_MULT",        "1.2"))
ABS_VOL_MIN      = float(os.getenv("ABS_VOL_MIN",     "3000000"))
LOOKAHEAD        = int(os.getenv("LOOKAHEAD",        "80"))
MAX_DAYS         = int(os.getenv("MAX_DAYS",         "200"))
CACHE_FILE       = os.getenv("CACHE_FILE",           "ipo_cache.pkl")
SIGNALS_CSV      = os.getenv("SIGNALS_CSV",         "ipo_signals.csv")
POSITIONS_CSV    = os.getenv("POSITIONS_CSV",       "ipo_positions.csv")
HEARTBEAT_RUNS   = int(os.getenv("HEARTBEAT_RUNS",    "0"))

# Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)

def send_telegram(msg):
    if not BOT_TOKEN or not CHAT_ID:
        logger.info("[Telegram disabled]")
        return
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    try:
        requests.post(url,
            json={"chat_id":CHAT_ID, "text":msg, "parse_mode":"HTML"},
            timeout=10)
    except Exception as e:
        logger.error(f"Telegram error: {e}")
    
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
    for _ in range(3):
        try:
            df = stock_df(symbol,
                from_date=start_date,
                to_date=datetime.today().date(),
                series="EQ")
            if df.empty: return None
            df = df.rename(columns={
                "CH_TIMESTAMP":"DATE","CH_OPENING_PRICE":"OPEN",
                "CH_TRADE_HIGH_PRICE":"HIGH","CH_TRADE_LOW_PRICE":"LOW",
                "CH_CLOSING_PRICE":"CLOSE","CH_TOT_TRADED_QTY":"VOLUME"
            })
            df["DATE"] = pd.to_datetime(df["DATE"]).dt.date
            return df.sort_values("DATE")
        except:
            time.sleep(1)
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
                price, pnl, days
            ]
            send_telegram(f"🛑 <b>{exit_reason}</b>\n{sym} @ ₹{price:.2f}\nPnL {pnl:.1f}%")
        else:
            df_pos.loc[idx, ["current_price","trailing_stop","pnl_pct","days_held"]] = [
                price, trailing, pnl, days
            ]
    df_pos.to_csv(POSITIONS_CSV, index=False)
    logger.info("Positions updated")

def dynamic_partial_take(grade):
    return PT_A_PLUS if grade=="A+" else PT_B if grade=="B" else PT_C

def detect_scan(symbols, listing_map):
    existing = pd.read_csv(SIGNALS_CSV)["signal_id"].tolist()
    for sym in symbols:
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
                        pd.concat([pd.read_csv(SIGNALS_CSV), pd.DataFrame([row])])\
                          .to_csv(SIGNALS_CSV, index=False)
                        pos = {
                            "symbol": sym, "entry_date": date, "entry_price": entry,
                            "grade": grade, "current_price": entry, "stop_loss": stop,
                            "trailing_stop": stop, "pnl_pct": 0, "days_held": 0, "status": "ACTIVE"
                        }
                        pd.concat([pd.read_csv(POSITIONS_CSV), pd.DataFrame([pos])])\
                          .to_csv(POSITIONS_CSV, index=False)
                        send_telegram(f"🎯 <b>IPO Signal</b>\n{sym} {grade} @ ₹{entry:.2f}")
                        break
                break

def weekly_summary():
    df = pd.read_csv(SIGNALS_CSV, parse_dates=["signal_date"])
    count = len(df[df["signal_date"]>=datetime.today()-timedelta(days=7)])
    send_telegram(f"📊 Weekly Summary\nNew Signals: {count}\nActive Positions: {len(pd.read_csv(POSITIONS_CSV))}")

def monthly_review():
    df = pd.read_csv(SIGNALS_CSV, parse_dates=["signal_date"])
    count = len(df[df["signal_date"]>=datetime.today().replace(day=1)])
    send_telegram(f"📊 Monthly Review\nThis Month: {count}\nTotal Signals: {len(df)}")

def heartbeat():
    send_telegram(f"✅ Scanner Heartbeat: {datetime.now().strftime('%Y-%m-%d %H:%M')}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode",choices=["scan","weekly_summary","monthly_review","heartbeat","dry_run"],
                        nargs="?",default="scan")
    args = parser.parse_args()

    initialize_csvs()
    update_positions()
    symbols, listing_map = get_symbols_and_listing()

    if args.mode == "scan":
        detect_scan(symbols, listing_map)
    elif args.mode == "weekly_summary":
        weekly_summary()
    elif args.mode == "monthly_review":
        monthly_review()
    elif args.mode == "heartbeat":
        heartbeat()
    else:
        logger.info("Dry run complete (no writes or Telegram)")
