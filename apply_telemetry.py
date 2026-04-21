import os

filepath = 'streamlined-ipo-scanner.py'
with open(filepath, 'r', encoding='utf-8') as f:
    content = f.read()

changes = 0

# 1. Add sanitize_metric and replace compute_grade_hybrid
old_grade_func = """def compute_grade_hybrid(df, idx, w, avg_vol):
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
    return score"""

new_grade_func = """import pandas as pd
import numpy as np

def sanitize_metric(val):
    \"\"\"Cast pandas/numpy variables to python natives for clean JSON logging\"\"\"
    if pd.isna(val) or val is None:
        return None
    try:
        if hasattr(val, 'item'):
            return val.item()
        if isinstance(val, (int, float, bool, str)):
            return val
        return float(val)
    except:
        return str(val)

def compute_grade_hybrid(df, idx, w, avg_vol):
    score=0
    low, high = df['LOW'].tail(w).min(), df['HIGH'].tail(w).max()
    prng = (high-low)/low*100
    if prng<=18: score+=1
    
    vol_ratio = df['VOLUME'].iat[idx]/avg_vol if avg_vol>0 else 0
    if df['VOLUME'].iat[idx]>=2.5*avg_vol and df['VOLUME'].iloc[idx-2:idx+1].sum()>=4*avg_vol: score+=1
    
    ret20 = (df['CLOSE'].iat[idx]/df['CLOSE'].iat[max(0,idx-20)]-1)
    percentile=np.percentile((df['CLOSE']-df['CLOSE'].shift(20))/df['CLOSE'].shift(20).fillna(0),85)
    rs_percentile_met = bool(ret20>=percentile)
    if rs_percentile_met: score+=1
    
    ema20,ema50 = df['CLOSE'].ewm(20).mean().iat[idx], df['CLOSE'].ewm(50).mean().iat[idx]
    macd = df['CLOSE'].ewm(12).mean().iat[idx] - df['CLOSE'].ewm(26).mean().iat[idx]
    sig = pd.Series(df['CLOSE'].ewm(12).mean()-df['CLOSE'].ewm(26).mean()).ewm(9).mean().iat[idx]
    rsi = 100-100/(1+(df['CLOSE'].diff().clip(lower=0).rolling(14).mean()/
                     df['CLOSE'].diff().clip(upper=0).abs().rolling(14).mean())).iat[idx]
    
    trend_alignment = bool(macd>sig and rsi>65 and ema20>ema50)
    if trend_alignment: score+=1
    if idx+1<len(df) and (df['OPEN'].iat[idx+1]/df['CLOSE'].iat[idx]-1)>=0.04: score+=1
    
    metrics_dict = {
        "metric_prng": sanitize_metric(prng),
        "metric_vol_ratio": sanitize_metric(vol_ratio),
        "metric_rsi": sanitize_metric(rsi),
        "metric_base_width": sanitize_metric(w),
        "metric_rs_percentile_met": sanitize_metric(rs_percentile_met),
        "metric_trend_alignment": sanitize_metric(trend_alignment)
    }

    return score, metrics_dict"""

if old_grade_func in content:
    content = content.replace(old_grade_func, new_grade_func)
    changes += 1
else:
    print("Could not find compute_grade_hybrid to replace")


# Update detect_live_patterns rejections
old_live_rejections = """                score = compute_grade_hybrid(df, j, w, avgv)
                grade = assign_grade(score)

                # Enforce minimum grade for LIVE signals
                if not is_live_grade_allowed(grade):
                    logger.info(f"⏭️ Skipping {sym} - grade {grade} below live threshold {MIN_LIVE_GRADE}")
                    _log_consolidation_reject_once({"reason": "low_grade", "grade": grade, "min_required": MIN_LIVE_GRADE})
                    continue

                # Enhanced B-grade filters with RSI and MACD
                if grade == 'B' and not smart_b_filters(df, j, avgv):
                    _log_consolidation_reject_once({"reason": "failed_b_filters", "grade": grade})
                    continue

                if grade == 'C' and not smart_c_filters(df, j, df["OPEN"].iat[j], w, avgv):
                    _log_consolidation_reject_once({"reason": "failed_c_filters", "grade": grade})
                    continue

                if grade == 'D':
                    _log_consolidation_reject_once({"reason": "grade_d", "grade": grade})
                    continue"""

new_live_rejections = """                score, metrics = compute_grade_hybrid(df, j, w, avgv)
                grade = assign_grade(score)

                # Enforce minimum grade for LIVE signals
                if not is_live_grade_allowed(grade):
                    logger.info(f"⏭️ Skipping {sym} - grade {grade} below live threshold {MIN_LIVE_GRADE}")
                    _log_consolidation_reject_once({"reason": "low_grade", "grade": grade, "min_required": MIN_LIVE_GRADE, **metrics})
                    continue

                # Enhanced B-grade filters with RSI and MACD
                if grade == 'B' and not smart_b_filters(df, j, avgv):
                    _log_consolidation_reject_once({"reason": "failed_b_filters", "grade": grade, **metrics})
                    continue

                if grade == 'C' and not smart_c_filters(df, j, df["OPEN"].iat[j], w, avgv):
                    _log_consolidation_reject_once({"reason": "failed_c_filters", "grade": grade, **metrics})
                    continue

                if grade == 'D':
                    _log_consolidation_reject_once({"reason": "grade_d", "grade": grade, **metrics})
                    continue"""

if old_live_rejections in content:
    content = content.replace(old_live_rejections, new_live_rejections)
    changes += 1
else:
    print("Could not find detect_live_patterns rejections to replace")


# Replace distance checks with metrics unpacking
content = content.replace(
    'write_daily_log("consolidation", sym, "REJECTED_BREAKOUT", {"reason": "poor_risk_reward", "ratio": round(risk_reward_ratio, 2), "min_required": MIN_RISK_REWARD})',
    'write_daily_log("consolidation", sym, "REJECTED_BREAKOUT", {"reason": "poor_risk_reward", "ratio": round(risk_reward_ratio, 2), "min_required": MIN_RISK_REWARD, **metrics})'
)
content = content.replace(
    'write_daily_log("consolidation", sym, "REJECTED_BREAKOUT", {"reason": "too_extended", "distance_pct": round(distance_above, 2), "max_allowed": MAX_ENTRY_ABOVE_BREAKOUT_PCT})',
    'write_daily_log("consolidation", sym, "REJECTED_BREAKOUT", {"reason": "too_extended", "distance_pct": round(distance_above, 2), "max_allowed": MAX_ENTRY_ABOVE_BREAKOUT_PCT, **metrics})'
)

# Insert the risk_pct hard stop for Live Patterns
old_live_stop = """                # Grade-based stop loss: More appropriate for IPO volatility
                stop, stop_pct = calculate_grade_based_stop_loss(entry, low, grade)
                date = entry_date  # Use actual entry date from dataframe"""

new_live_stop = """                # Grade-based stop loss: More appropriate for IPO volatility
                stop, stop_pct = calculate_grade_based_stop_loss(entry, low, grade)
                risk_pct = (entry - stop) / entry * 100
                if risk_pct > 10.0:
                    logger.info(f"⏭️ Skipping {sym} - Stop risk {risk_pct:.2f}% exceeds hard 10% limit")
                    _log_consolidation_reject_once({"reason": "excessive_stop_risk", "risk_pct": round(risk_pct, 2), "max_allowed": 10.0, **metrics})
                    continue
                date = entry_date  # Use actual entry date from dataframe"""

if old_live_stop in content:
    content = content.replace(old_live_stop, new_live_stop)
    changes += 1

# Change live accepted
old_live_accept = """                write_daily_log("consolidation", sym, "SIGNAL_GENERATED", {"""
new_live_accept = """                metrics["metric_ipo_age"] = sanitize_metric(ipo_age) if 'ipo_age' in locals() else None
                write_daily_log("consolidation", sym, "ACCEPTED_BREAKOUT", {**metrics, """

if old_live_accept in content:
    content = content.replace(old_live_accept, new_live_accept)
    changes += 1

# detect_scan updates
old_scan_rejections = """                score = compute_grade_hybrid(df, j, w, avgv)
                grade = assign_grade(score)
                if grade == "D":
                    _log_scan_reject_once({"reason": "grade_d", "grade": grade, "mode": "scan"})
                    continue"""
                    
new_scan_rejections = """                score, metrics = compute_grade_hybrid(df, j, w, avgv)
                grade = assign_grade(score)
                if grade == "D":
                    _log_scan_reject_once({"reason": "grade_d", "grade": grade, "mode": "scan", **metrics})
                    continue"""

if old_scan_rejections in content:
    content = content.replace(old_scan_rejections, new_scan_rejections)
    changes += 1

# detect_scan 10% risk limit
old_scan_stop = """                # Grade-based stop loss: More appropriate for IPO volatility
                stop, stop_pct = calculate_grade_based_stop_loss(entry, low, grade)
                # Use actual entry date from dataframe
                date = entry_date"""

new_scan_stop = """                # Grade-based stop loss: More appropriate for IPO volatility
                stop, stop_pct = calculate_grade_based_stop_loss(entry, low, grade)
                risk_pct = (entry - stop) / entry * 100
                if risk_pct > 10.0:
                    logger.info(f"⏭️ Skipping {sym} - Stop risk {risk_pct:.2f}% exceeds hard 10% limit")
                    _log_scan_reject_once({"reason": "excessive_stop_risk", "risk_pct": round(risk_pct, 2), "max_allowed": 10.0, **metrics})
                    continue
                # Use actual entry date from dataframe
                date = entry_date"""

if old_scan_stop in content:
    content = content.replace(old_scan_stop, new_scan_stop)
    changes += 1


with open(filepath, 'w', encoding='utf-8') as f:
    f.write(content)

print(f"Applied {changes} changes to streamlined-ipo-scanner.py")
