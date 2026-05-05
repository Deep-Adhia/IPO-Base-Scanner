import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

def compute_market_context(market_data: pd.DataFrame = None) -> dict:
    """
    Fetch and analyze Nifty 50 context.
    If market_data is not provided, it fetches the last 60 days.
    """
    try:
        if market_data is None:
            # Fetch Nifty 50 (^NSEI)
            nifty = yf.Ticker("^NSEI")
            market_data = nifty.history(period="60d")
            
        if market_data is None or market_data.empty:
            return {"error": "No market data available"}
            
        # Ensure column names are standard
        market_data.columns = [c.upper() for c in market_data.columns]
        
        # 1. Nifty Daily Return
        latest_close = market_data['CLOSE'].iloc[-1]
        prev_close = market_data['CLOSE'].iloc[-2]
        nifty_return = (latest_close / prev_close - 1) * 100.0
        
        # 2. Distance from 20DMA
        ma20 = market_data['CLOSE'].rolling(window=20).mean()
        dist_20ma = (latest_close / ma20.iloc[-1] - 1) * 100.0
        
        # 3. Trend Slope (Last 5 days)
        # Simple linear regression slope proxy
        y = market_data['CLOSE'].tail(5).values
        x = range(5)
        slope = (len(x) * sum(x*y) - sum(x) * sum(y)) / (len(x) * sum(x**2) - sum(x)**2)
        trend_slope = (slope / latest_close) * 100.0 # Normalized as %
        
        # 4. Market State Label (Derived)
        if latest_close > ma20.iloc[-1] and trend_slope > 0:
            state = "BULL_CONFIRMED"
        elif latest_close < ma20.iloc[-1] and trend_slope < 0:
            state = "BEAR_CONFIRMED"
        elif latest_close > ma20.iloc[-1] and trend_slope < 0:
            state = "DISTRIBUTION"
        else:
            state = "ACCUMULATION"
            
        return {
            "nifty_return": round(float(nifty_return), 2),
            "nifty_20ma_dist": round(float(dist_20ma), 2),
            "nifty_trend_slope": round(float(trend_slope), 3),
            "market_state": state,
            "nifty_close": round(float(latest_close), 2),
            "market_data_date": market_data.index[-1].strftime("%Y-%m-%d") if hasattr(market_data.index[-1], 'strftime') else str(market_data.index[-1]),
            "market_cache_timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
    except Exception as e:
        logger.error(f"Error computing market context: {e}")
        return {"error": str(e)}
