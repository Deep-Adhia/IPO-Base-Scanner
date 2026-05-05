import logging
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

def evaluate_signal_outcome(repo, signal_id: str) -> dict:
    """
    Analyzes the full lifecycle of a signal to produce the final outcome summary.
    """
    try:
        # Fetch all updates for this signal
        updates = list(repo.signal_updates.find({"signal_id": signal_id}).sort("date", 1))
        
        if not updates:
            return {"error": "No updates found for signal"}

        # Extract time-series for metrics
        runups = [u["runup_pct"] for u in updates]
        drawdowns = [u["drawdown_pct"] for u in updates]
        
        max_runup = max(runups) if runups else 0.0
        max_drawdown = min(drawdowns) if drawdowns else 0.0
        
        # Horizon Snapshots (1d, 3d, 5d)
        def get_horizon_metrics(days: int):
            if len(updates) >= days:
                upd = updates[days-1]
                return {
                    f"runup_{days}d": upd["runup_pct"],
                    f"close_{days}d": upd["close"]
                }
            return {f"runup_{days}d": None, f"close_{days}d": None}

        h1 = get_horizon_metrics(1)
        h3 = get_horizon_metrics(3)
        h5 = get_horizon_metrics(5)

        # Holding Efficiency: How much of the max rally did we actually keep?
        latest_runup = updates[-1]["runup_pct"]
        holding_efficiency = latest_runup / max_runup if max_runup > 0 else 0.0

        # Outcome Labeling (Logical, not just PnL)
        if max_runup >= 0.15:
            label = "WINNER"
        elif max_runup >= 0.05 and latest_runup > 0:
            label = "MODERATE_WIN"
        elif max_drawdown <= -0.08:
            label = "LOSER"
        else:
            label = "CHOP"

        outcome_doc = {
            "signal_id": signal_id,
            "max_runup": round(max_runup, 4),
            "max_drawdown": round(max_drawdown, 4),
            "holding_efficiency": round(holding_efficiency, 3),
            "label": label,
            **h1, **h3, **h5,
            "processed_at": datetime.now(timezone.utc)
        }

        repo.save_outcome(outcome_doc)
        return outcome_doc

    except Exception as e:
        logger.error(f"❌ [Evaluator] Failed to evaluate signal {signal_id}: {e}")
        return {"error": str(e)}
