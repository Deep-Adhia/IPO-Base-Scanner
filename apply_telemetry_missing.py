import os

filepath = 'streamlined-ipo-scanner.py'
with open(filepath, 'r', encoding='utf-8') as f:
    content = f.read()

replacements = [
    (
        '_log_consolidation_reject_once({"reason": "invalid_risk_reward", "risk": round(risk_amount, 2), "reward": round(reward_amount, 2)})',
        '_log_consolidation_reject_once({"reason": "invalid_risk_reward", "risk": round(risk_amount, 2), "reward": round(reward_amount, 2), **metrics})'
    ),
    (
        '_log_consolidation_reject_once({"reason": "poor_risk_reward", "ratio": round(risk_reward_ratio, 2), "min_required": MIN_RISK_REWARD})',
        '_log_consolidation_reject_once({"reason": "poor_risk_reward", "ratio": round(risk_reward_ratio, 2), "min_required": MIN_RISK_REWARD, **metrics})'
    ),
    (
        '_log_consolidation_reject_once({"reason": "too_extended", "distance_pct": round(distance_above, 2), "max_allowed": MAX_ENTRY_ABOVE_BREAKOUT_PCT})',
        '_log_consolidation_reject_once({"reason": "too_extended", "distance_pct": round(distance_above, 2), "max_allowed": MAX_ENTRY_ABOVE_BREAKOUT_PCT, **metrics})'
    ),
    (
        '_log_consolidation_reject_once({"reason": "stale_breakout", "days_old": days_since_breakout})',
        '_log_consolidation_reject_once({"reason": "stale_breakout", "days_old": days_since_breakout, **metrics})'
    ),
    (
        '_log_consolidation_reject_once({"reason": "stale_and_fallen", "days_old": days_since_breakout, "entry": round(entry, 2), "breakout_level": round(high2, 2)})',
        '_log_consolidation_reject_once({"reason": "stale_and_fallen", "days_old": days_since_breakout, "entry": round(entry, 2), "breakout_level": round(high2, 2), **metrics})'
    ),
    (
        '_log_scan_reject_once({"reason": "active_position", "mode": "scan"})',
        '_log_scan_reject_once({"reason": "active_position", "mode": "scan", **metrics})'
    )
]

for old, new in replacements:
    content = content.replace(old, new)

with open(filepath, 'w', encoding='utf-8') as f:
    f.write(content)
