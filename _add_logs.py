import os

filepath = 'streamlined-ipo-scanner.py'
with open(filepath, 'r', encoding='utf-8') as f:
    content = f.read()

replacements = [
    (
        '''                    if not close_holds and not volume_confirms:\n                        continue''',
        '''                    if not close_holds and not volume_confirms:\n                        write_daily_log("consolidation", sym, "REJECTED_BREAKOUT", {"reason": "failed_follow_through", "close_holds": bool(close_holds), "volume_confirms": bool(volume_confirms)})\n                        continue'''
    ),
    (
        '''                if not is_live_grade_allowed(grade):\n                    logger.info(f"⏭️ Skipping {sym} - grade {grade} below live threshold {MIN_LIVE_GRADE}")\n                    continue''',
        '''                if not is_live_grade_allowed(grade):\n                    logger.info(f"⏭️ Skipping {sym} - grade {grade} below live threshold {MIN_LIVE_GRADE}")\n                    write_daily_log("consolidation", sym, "REJECTED_BREAKOUT", {"reason": "low_grade", "grade": grade, "min_required": MIN_LIVE_GRADE})\n                    continue'''
    ),
    (
        '''                if grade == 'B' and not smart_b_filters(df, j, avgv):\n                    continue''',
        '''                if grade == 'B' and not smart_b_filters(df, j, avgv):\n                    write_daily_log("consolidation", sym, "REJECTED_BREAKOUT", {"reason": "failed_b_filters", "grade": grade})\n                    continue'''
    ),
    (
        '''                if grade == 'C' and not smart_c_filters(df, j, df["OPEN"].iat[j], w, avgv):\n                    continue''',
        '''                if grade == 'C' and not smart_c_filters(df, j, df["OPEN"].iat[j], w, avgv):\n                    write_daily_log("consolidation", sym, "REJECTED_BREAKOUT", {"reason": "failed_c_filters", "grade": grade})\n                    continue'''
    ),
    (
        '''                if grade == 'D':\n                    continue''',
        '''                if grade == 'D':\n                    write_daily_log("consolidation", sym, "REJECTED_BREAKOUT", {"reason": "grade_d", "grade": grade})\n                    continue'''
    ),
    (
        '''                if risk_amount <= 0 or reward_amount <= 0:\n                    logger.info(f"⏭️ Skipping {sym} - invalid risk/reward (risk={risk_amount:.2f}, reward={reward_amount:.2f})")\n                    continue''',
        '''                if risk_amount <= 0 or reward_amount <= 0:\n                    logger.info(f"⏭️ Skipping {sym} - invalid risk/reward (risk={risk_amount:.2f}, reward={reward_amount:.2f})")\n                    write_daily_log("consolidation", sym, "REJECTED_BREAKOUT", {"reason": "invalid_risk_reward", "risk": round(risk_amount, 2), "reward": round(reward_amount, 2)})\n                    continue'''
    ),
    (
        '''                if risk_reward_ratio < MIN_RISK_REWARD:\n                    logger.info(f"⏭️ Skipping {sym} - poor risk/reward 1:{risk_reward_ratio:.2f} (< {MIN_RISK_REWARD})")\n                    continue''',
        '''                if risk_reward_ratio < MIN_RISK_REWARD:\n                    logger.info(f"⏭️ Skipping {sym} - poor risk/reward 1:{risk_reward_ratio:.2f} (< {MIN_RISK_REWARD})")\n                    write_daily_log("consolidation", sym, "REJECTED_BREAKOUT", {"reason": "poor_risk_reward", "ratio": round(risk_reward_ratio, 2), "min_required": MIN_RISK_REWARD})\n                    continue'''
    ),
    (
        '''                    if distance_above > MAX_ENTRY_ABOVE_BREAKOUT_PCT:\n                        logger.info(\n                            f"⏭️ Skipping {sym} - entry {distance_above:.2f}% above breakout "\n                            f"(max allowed {MAX_ENTRY_ABOVE_BREAKOUT_PCT}%)\"\n                        )\n                        continue''',
        '''                    if distance_above > MAX_ENTRY_ABOVE_BREAKOUT_PCT:\n                        logger.info(\n                            f"⏭️ Skipping {sym} - entry {distance_above:.2f}% above breakout "\n                            f"(max allowed {MAX_ENTRY_ABOVE_BREAKOUT_PCT}%)\"\n                        )\n                        write_daily_log("consolidation", sym, "REJECTED_BREAKOUT", {"reason": "too_extended", "distance_pct": round(distance_above, 2), "max_allowed": MAX_ENTRY_ABOVE_BREAKOUT_PCT})\n                        continue'''
    ),
    (
        '''                if days_since_breakout > 10:\n                    logger.info(f"⏭️ Skipping {sym} - breakout is {days_since_breakout} days old (>10 days, too stale)")\n                    continue''',
        '''                if days_since_breakout > 10:\n                    logger.info(f"⏭️ Skipping {sym} - breakout is {days_since_breakout} days old (>10 days, too stale)")\n                    write_daily_log("consolidation", sym, "REJECTED_BREAKOUT", {"reason": "stale_breakout", "days_old": days_since_breakout})\n                    continue'''
    ),
    (
        '''                    if entry < high2:\n                        logger.info(f"⏭️ Skipping {sym} - breakout {days_since_breakout} days old and price ₹{entry:.2f} has fallen below breakout level ₹{high2:.2f}")\n                        continue''',
        '''                    if entry < high2:\n                        logger.info(f"⏭️ Skipping {sym} - breakout {days_since_breakout} days old and price ₹{entry:.2f} has fallen below breakout level ₹{high2:.2f}")\n                        write_daily_log("consolidation", sym, "REJECTED_BREAKOUT", {"reason": "stale_and_fallen", "days_old": days_since_breakout, "entry": round(entry, 2), "breakout_level": round(high2, 2)})\n                        continue'''
    ),
]

changes_made = 0
for old, new in replacements:
    if old in content:
        content = content.replace(old, new)
        changes_made += 1
    else:
        # Try finding a normalized version to debug
        old_norm = " ".join(old.split())
        content_norm = " ".join(content.split())
        if old_norm in content_norm:
            print(f"Whitespace mismatch for:\n{old[:50]}...")
        else:
            print(f"Not found completely:\n{old[:50]}...")

with open(filepath, 'w', encoding='utf-8') as f:
    f.write(content)

print(f"Successfully applied {changes_made} replacements.")
