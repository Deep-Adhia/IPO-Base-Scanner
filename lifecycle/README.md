# 🔄 Lifecycle & Evaluation Layer

This layer tracks the **evolution of truth** from the moment a signal is generated until the position is closed.

## Components

### 📈 `tracker.py`
The **Lifecycle Tracker** records append-only daily updates for every active signal. 
- Records `runup_pct` and `drawdown_pct`.
- Tracks `days_since_signal`.
- Enables time-series reconstruction of trade performance.

### 🏆 `evaluator.py`
The **Outcome Evaluator** produces the final "Report Card" for a signal.
- **Horizon Analysis**: Max runup at 5, 10, 20 days.
- **Efficiency Metrics**: Profit-to-Drawdown ratios.
- **Outcome Labeling**: WINNER / LOSER / CHOP based on objective price paths.
