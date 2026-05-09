from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Dict, Any

@dataclass(frozen=True)
class Signal:
    """
    The Immutable Snapshot: Ground truth at decision time.
    """
    signal_id: str
    symbol: str
    signal_date: datetime
    breakout_date: datetime
    candle_timestamp: datetime

    entry_price: float
    stop_price: float
    target_price: float

    # Raw inputs from the scanner
    features: Dict[str, Any]
    
    # Enrichment Layers
    breakout_fingerprint: Dict[str, Any]
    base_quality: Dict[str, Any]
    market_context: Dict[str, Any]

    # Reconciliation with Execution Layer (v1)
    source_log_id: str
    v1_entry_price: float
    entry_price_delta_pct: float
    is_complete_snapshot: bool
    # Score breakdown
    score_components: Dict[str, Any]
    
    # Metadata
    scanner: str
    scanner_version: str
    sector: str = "Unknown"
    industry: str = "Unknown"

    # Research Metadata (Phase 2)
    pattern_type: str = "UNKNOWN"
    market_regime: str = "UNKNOWN"
    lifecycle_state: str = "POSITION_ACTIVE"
    source_type: str = "UNKNOWN"
    data_quality: str = "UNKNOWN"
    decision_snapshot: Dict[str, Any] = field(default_factory=dict)

    # Fields with Defaults
    incomplete_reasons: list[str] = field(default_factory=list)
    data_version: str = "v2"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict for MongoDB storage."""
        return asdict(self)

@dataclass(frozen=True)
class SignalUpdate:
    """
    The Lifecycle Event: Append-only PnL and state evolution.
    """
    signal_id: str
    date: datetime
    days_since_signal: int
    
    current_price: float
    runup_pct: float
    drawdown_pct: float
    
    status: str # ACTIVE / CLOSED
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

@dataclass(frozen=True)
class SignalOutcome:
    """
    The Final Grade: Post-trade performance summary.
    """
    signal_id: str
    symbol: str
    
    exit_date: datetime
    exit_price: float
    final_pnl_pct: float
    
    # Analysis metrics
    max_runup_pct: float
    max_drawdown_pct: float
    efficiency_ratio: float # PnL / Max Drawdown
    
    outcome_label: str # WINNER / LOSER / CHOP
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
