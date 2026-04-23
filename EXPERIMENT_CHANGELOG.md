# Experiment Changelog

This file tracks analysis and logging cutovers so experiment windows stay comparable.

## Current Active Baseline

- `scanner_version`: `2.1.0`
- `log_schema_version`: `2026-04-23.v1`
- recommended clean analysis start: `2026-04-24`

## Why this exists

- Strategy and logging logic evolve over time.
- Comparing pre-change and post-change rows in one bucket can pollute results.
- This changelog provides explicit cut points for analysis filters.

## Analysis command (clean cohort)

```bash
python analyze_30d_data.py --start-date 2026-04-24 --version 2.1.0 --clean-cohort
```

`--clean-cohort` excludes:
- `signal_type == WATCHLIST`
- grades containing `LOW_VOL`

## Notable recent milestones

- `2026-04-15`: lifecycle logging additions in positions pipeline
- `2026-04-21`: granular telemetry integration for consolidation
- `2026-04-23`: analysis filters and schema-aligned docs updates
