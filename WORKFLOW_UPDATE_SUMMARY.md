# Workflow Update Summary

## Changes Made to `.github/workflows/ipo-scanner.yml`:

✅ **Removed TA-Lib Installation Step** (lines 43-66 in old version)
- The TA-Lib installation step has been removed
- TA-Lib is not used in the project (all indicators calculated manually)

## Changes Made to `requirements.txt`:

✅ **Removed TA-Lib dependency**
- Removed: `TA-Lib>=0.4.25`
- Removed: `beautifulsoup4` (let jugaad-data install it automatically)

## Phase 2.2: Structured Telemetry (v2.2.0)
(Applied 2026-04-25)

✅ **High-Fidelity Research Telemetry**
- Upgraded scanners to capture "Near Miss" data (setups within 10-15% of threshold).
- Standardized `log_type: "REJECTED"` for instant dataset segmentation in Atlas.
- Added `failing_metric` and `threshold` mapping for unambiguous audit trails.

✅ **Data Intelligence Layer**
- Integrated `analyze_telemetry.py` for distribution analysis (Accepted vs Rejected).
- Added `manage_db.py analyze` command for rapid strategy health checks.

## Current Workflow Structure:

1.  **Checkout code**
2.  **Set up Python 3.10**
3.  **Clean up caches**
4.  **Install dependencies** (`requirements.txt` forced to UTF-8)
5.  **Check MongoDB Connection** (`test_db_connection.py`)
6.  **Run Scanners** (v2.2.0 Telemetry Enabled)
7.  **Commit/Push CSVs** (Legacy validation layer)

---

## Roadmap:
- **Phase 3**: 3-Day Live Validation (Monitoring CSV vs DB parity).
- **Phase 4**: Extracting edge from rejection telemetry (Threshold optimization).

