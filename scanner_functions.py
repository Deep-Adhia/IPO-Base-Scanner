# scanner_functions.py - Core functions for IPO scanner
import pandas as pd
import numpy as np

def supertrend(df, period=10, multiplier=2):
    """Calculate SuperTrend indicator"""
    high = df['HIGH']
    low = df['LOW']
    close = df['CLOSE']
    
    # Calculate ATR
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(period).mean()
    
    # Calculate SuperTrend
    hl2 = (high + low) / 2
    upper = hl2 + (multiplier * atr)
    lower = hl2 - (multiplier * atr)
    
    # Initialize arrays
    supertrend = pd.Series(index=df.index, dtype=float)
    direction = pd.Series(index=df.index, dtype=int)
    
    for i in range(1, len(df)):
        if close.iloc[i] <= lower.iloc[i-1]:
            supertrend.iloc[i] = lower.iloc[i]
            direction.iloc[i] = 1
        elif close.iloc[i] >= upper.iloc[i-1]:
            supertrend.iloc[i] = upper.iloc[i]
            direction.iloc[i] = -1
        else:
            supertrend.iloc[i] = supertrend.iloc[i-1]
            direction.iloc[i] = direction.iloc[i-1]
    
    return supertrend

def compute_grade_hybrid(df, entry_idx, window, avg_volume):
    """Compute hybrid grade based on technical indicators - COMPLEX VERSION"""
    close = df['CLOSE']
    volume = df['VOLUME']
    high = df['HIGH']
    low = df['LOW']
    
    score = 0
    
    # RSI calculation
    delta = close.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    
    # RSI check - more restrictive
    if 40 <= rsi.iloc[entry_idx] <= 75:
        score += 1
    
    # MACD calculation
    exp1 = close.ewm(span=12).mean()
    exp2 = close.ewm(span=26).mean()
    macd = exp1 - exp2
    signal = macd.ewm(span=9).mean()
    
    # MACD check
    if macd.iloc[entry_idx] > signal.iloc[entry_idx]:
        score += 1
    
    # Volume check - STRICT
    vol_ratio = volume.iloc[entry_idx] / avg_volume
    if vol_ratio >= 2.5:  # Must be 2.5x average volume
        score += 1
    
    # Price momentum - STRICT
    momentum = (close.iloc[entry_idx] - close.iloc[entry_idx-window]) / close.iloc[entry_idx-window]
    if momentum > 0.05:
        score += 1
    
    # Breakout confirmation - STRICT
    recent_high = high.iloc[entry_idx-window:entry_idx].max()
    if close.iloc[entry_idx] > recent_high:
        score += 1
    
    # Additional volume confirmation - 3-day volume spike
    if entry_idx >= 2:
        three_day_vol = volume.iloc[entry_idx-2:entry_idx+1].sum()
        if three_day_vol >= 4 * avg_volume:
            score += 1
    
    # Price range check - consolidation
    price_range = (high.iloc[entry_idx-window:entry_idx+1].max() - 
                   low.iloc[entry_idx-window:entry_idx+1].min()) / low.iloc[entry_idx-window:entry_idx+1].min() * 100
    if price_range <= 60:  # Must be consolidated (range < 60%)
        score += 1
    
    return score

def assign_grade(score):
    """Assign grade based on score"""
    if score >= 5:
        return "A+"
    elif score >= 4:
        return "A"
    elif score >= 3:
        return "B"
    elif score >= 2:
        return "C"
    else:
        return "D"
