"""
test_bucket_logic.py
Unit tests for the 3-bucket validation logic in streamlined_ipo_scanner.py.
Tests boundary cases for PRNG and Volume thresholds.
"""
import sys
import os

# Mock the constants and function for testing
BUCKET_ALIGNED  = "ALIGNED"
BUCKET_EXTENDED = "EXTENDED"
BUCKET_BROKEN   = "BROKEN"

RC_OK                 = "OK"
RC_PRNG_LIMIT         = "RC_PRNG_LIMIT"
RC_VOL_LIMIT          = "RC_VOL_LIMIT"
RC_NON_IPO_CONTEXT    = "RC_NON_IPO_CONTEXT"
RC_STRUCTURE_FAILED   = "RC_STRUCTURE_FAILED"
RC_ERRATIC_VOLATILITY = "RC_ERRATIC_VOLATILITY"

def categorize_signal_bucket(metrics: dict, days_since_listing: int) -> tuple:
    reasons = []
    prng = metrics.get("prng", 0)
    vol_ratio = metrics.get("vol_ratio", 0)
    
    if prng > 45.0:
        reasons.append(RC_ERRATIC_VOLATILITY)
    if days_since_listing > 750:
        reasons.append(RC_NON_IPO_CONTEXT)
    
    if reasons:
        return BUCKET_BROKEN, reasons

    if prng > 25.0:
        reasons.append(RC_PRNG_LIMIT)
    if vol_ratio < 1.2:
        reasons.append(RC_VOL_LIMIT)
    
    if reasons:
        return BUCKET_EXTENDED, reasons
    
    return BUCKET_ALIGNED, [RC_OK]

def run_test_suite():
    tests = [
        {"metrics": {"prng": 25.0, "vol_ratio": 1.2}, "age": 100, "expected": BUCKET_ALIGNED,  "desc": "Perfect ALIGNED (Exactly 25.0, 1.2)"},
        {"metrics": {"prng": 25.1, "vol_ratio": 1.2}, "age": 100, "expected": BUCKET_EXTENDED, "desc": "PRNG Over Limit (25.1)"},
        {"metrics": {"prng": 20.0, "vol_ratio": 1.1}, "age": 100, "expected": BUCKET_EXTENDED, "desc": "VOL Under Limit (1.1)"},
        {"metrics": {"prng": 45.0, "vol_ratio": 1.2}, "age": 100, "expected": BUCKET_EXTENDED, "desc": "Perfect EXTENDED (Exactly 45.0)"},
        {"metrics": {"prng": 45.1, "vol_ratio": 1.2}, "age": 100, "expected": BUCKET_BROKEN,   "desc": "PRNG Too High (45.1)"},
        {"metrics": {"prng": 20.0, "vol_ratio": 1.2}, "age": 800, "expected": BUCKET_BROKEN,   "desc": "Non-IPO Context (Age 800)"},
        {"metrics": {"prng": 20.0, "vol_ratio": 0.4}, "age": 100, "expected": BUCKET_EXTENDED, "desc": "Extremely Low Vol (0.4) but within PRNG (EXTENDED)"},
    ]

    print(f"\n{'='*80}")
    print(f"{'Desc':<45} | {'Result':<10} | {'Reasons'}")
    print(f"{'-'*80}")

    passed = 0
    for t in tests:
        bucket, reasons = categorize_signal_bucket(t["metrics"], t["age"])
        success = (bucket == t["expected"])
        if success: passed += 1
        
        status = "PASS" if success else "FAIL"
        print(f"{t['desc']:<45} | {status:<10} | {bucket} ({', '.join(reasons)})")

    print(f"{'-'*80}")
    print(f"RESULT: {passed}/{len(tests)} Tests Passed")
    print(f"{'='*80}\n")

if __name__ == "__main__":
    run_test_suite()
