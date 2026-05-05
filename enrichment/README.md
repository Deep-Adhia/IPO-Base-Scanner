# 🧠 Enrichment Layer (Feature Store)

The Enrichment Layer is responsible for extracting "Decision Context." It turns raw price data into high-predictive features that enable causal analysis.

## Enrichment Modules

### 🎯 `breakout.py`
Analyzes the **fingerprint** of the breakout candle.
- **Body-to-Range**: Measures conviction.
- **Upper Wick %**: Measures supply/rejection.
- **Close vs High %**: Measures price holding power.
- **Volume Z-Score**: Measures institutional participation.

### 🧱 `base.py`
Analyzes the quality of the **Consolidation Base** preceding the breakout.
- **Tightness Index**: Standard deviation of price within the base.
- **Vol Dry-up**: Volume reduction near the end of the base.
- **Base Depth & Symmetry**: Structural soundness.

### 📈 `market.py`
Captures the **Market Regime** at the moment of the signal.
- **Nifty Trend Slope**: Long-term direction.
- **20MA Distance**: Measures extension/exhaustion.
- **Regime Labeling**: BULL / SIDEWAYS / DISTRIBUTION.

### ⚙️ `engine.py`
The coordinator that runs all enrichment modules and packages them into a single context object.
