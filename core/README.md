# 🏛️ Core Data Layer

This directory contains the foundational data models and the primary repository layer for the institutional analytics engine.

## Components

### 📦 `models.py`
Defines the strictly-typed `Signal`, `SignalUpdate`, and `SignalOutcome` dataclasses. These models act as the "Data Contract" for the entire system, ensuring consistent field names across snapshots, time-series, and outcomes.

### 🗄️ `repository.py`
The MongoDB interaction layer. It enforces:
- **Deterministic IDs**: `symbol_YYYYMMDD_hash` for snapshots.
- **Unique Indexes**: Prevents duplicate lifecycle updates on the same signal/day.
- **Schema Protection**: Centralizes all database writes to ensure data versioning (`v2`) is strictly followed.
