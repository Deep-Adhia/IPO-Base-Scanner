import pandas as pd
import numpy as np

def compute_base_quality(base_candles: pd.DataFrame) -> dict:
    """
    Continuous metrics for the consolidation base structure.
    """
    if base_candles is None or len(base_candles) < 3:
        return {"error": "Insufficient base data"}
        
    try:
        # Convert to numeric to be safe
        closes = base_candles['CLOSE'].astype(float)
        highs = base_candles['HIGH'].astype(float)
        lows = base_candles['LOW'].astype(float)
        vols = base_candles['VOLUME'].astype(float)
        
        # 1. Tightness Index (Avg Daily Range %)
        daily_ranges_pct = (highs - lows) / closes * 100.0
        tightness_index = daily_ranges_pct.mean()
        
        # 2. Volume Dry-up Ratio
        # Last 5 days vs prior 15 days in the base
        if len(vols) >= 10:
            recent_vol = vols.tail(5).mean()
            prior_vol = vols.iloc[:-5].tail(15).mean()
            vol_dryup_ratio = recent_vol / prior_vol if prior_vol > 0 else 1.0
        else:
            vol_dryup_ratio = 1.0
            
        # 3. Base Depth %
        base_high = highs.max()
        base_low = lows.min()
        base_depth = (base_high - base_low) / base_low * 100.0
        
        # 4. Base Symmetry (Price position in base)
        # 0.5 = middle, 1.0 = top, 0.0 = bottom
        current_close = closes.iloc[-1]
        symmetry = (current_close - base_low) / (base_high - base_low) if base_high > base_low else 0.5
        
        # 5. Breakout Tension (Standard deviation of closes / range)
        # Lower = more compressed/coiled
        tension = closes.std() / (base_high - base_low) if base_high > base_low else 0
        
        return {
            "tightness_index": round(float(tightness_index), 2),
            "vol_dryup_ratio": round(float(vol_dryup_ratio), 3),
            "base_depth": round(float(base_depth), 2),
            "base_symmetry": round(float(symmetry), 3),
            "breakout_tension": round(float(tension), 3),
            "base_duration_days": len(base_candles)
        }
    except Exception as e:
        return {"error": str(e)}
