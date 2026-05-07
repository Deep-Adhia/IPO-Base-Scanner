import pandas as pd
from datetime import datetime
from .breakout import compute_breakout_fingerprint
from .base import compute_base_quality
from .market import compute_market_context

class EnrichmentEngine:
    """
    Coordinator for signal enrichment.
    Takes raw data and returns structured features.
    """
    def __init__(self):
        # We could cache market data here for a single run
        self._market_cache = None
        self._cache_date = None

    def get_market_data(self):
        """Fetch market data once per run."""
        from datetime import datetime
        today = datetime.now().date()
        
        if self._market_cache is None or self._cache_date != today:
            context = compute_market_context()
            self._market_cache = context
            self._cache_date = today
            
        return self._market_cache

    def enrich_signal(self, candle: pd.Series, history: pd.DataFrame, base_candles: pd.DataFrame, reference_date: datetime = None) -> dict:
        """
        Enrich a raw signal with institutional features.
        """
        breakout_features = compute_breakout_fingerprint(candle, history)
        base_features = compute_base_quality(base_candles)
        
        # If backfilling, we need historical market data, not cached current data
        if reference_date:
            market_features = compute_market_context(end_date=reference_date)
        else:
            market_features = self.get_market_data()

        return {
            "breakout": breakout_features,
            "base": base_features,
            "market": market_features
        }
