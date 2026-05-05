import pandas as pd
import numpy as np

def compute_breakout_fingerprint(candle: pd.Series, history: pd.DataFrame) -> dict:
    """
    Derived features for the breakout candle character.
    """
    try:
        h = float(candle['HIGH'])
        l = float(candle['LOW'])
        o = float(candle['OPEN'])
        c = float(candle['CLOSE'])
        v = float(candle['VOLUME'])
        
        day_range = h - l if h > l else 1e-9
        body = abs(c - o)
        
        # 1. Body to Range (Conviction)
        body_to_range = body / day_range
        
        # 2. Upper Wick % (Rejection)
        upper_wick = h - max(o, c)
        upper_wick_pct = upper_wick / day_range
        
        # 3. Close Position % (Where it closed in the day's range)
        close_position_pct = (c - l) / day_range
        
        # 4. Volume Z-Score (Relative spike)
        # Use last 20 days for baseline
        if len(history) >= 5:
            recent_vols = history['VOLUME'].tail(20).astype(float)
            v_mean = recent_vols.mean()
            v_std = recent_vols.std()
            volume_zscore = (v - v_mean) / v_std if v_std > 0 else 0
        else:
            volume_zscore = 0
            
        # 5. Range Expansion
        if len(history) >= 10:
            avg_range_10d = (history['HIGH'].tail(10) - history['LOW'].tail(10)).mean()
            range_expansion = day_range / avg_range_10d if avg_range_10d > 0 else 1.0
        else:
            range_expansion = 1.0

        # 6. Rejection Ratio (Close vs High)
        # 0.0 = Closed at High, 1.0 = Closed at Low
        close_vs_high_pct = (h - c) / day_range if day_range > 0 else 0.0
            
        return {
            "body_to_range": round(body_to_range, 3),
            "upper_wick_pct": round(upper_wick_pct, 3),
            "close_position_pct": round(close_position_pct, 3),
            "volume_zscore": round(volume_zscore, 2),
            "range_expansion": round(range_expansion, 2),
            "close_vs_high_pct": round(close_vs_high_pct, 3)
        }
    except Exception as e:
        return {"error": str(e)}
