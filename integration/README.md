# 🌉 Integration Layer (The Bridge)

The Integration Layer acts as the glue between the high-speed scanner and the structured analytics engine.

## Components

### 🏗️ `signal_builder.py`
The **Signal Builder** is a factory class that aggregates data from multiple sources:
1. **Scanner Metrics**: Raw scores, tier weights, and rejection reasons.
2. **Feature Store**: Enriched breakout and base features.
3. **Reconciliation**: Links v2 snapshots to original v1 log IDs and tracks execution price drift (`entry_price_delta_pct`).

It is responsible for generating the **Deterministic Signal ID** and ensuring the `is_complete_snapshot` flag accurately reflects data integrity.
