import argparse
from collections import Counter
from datetime import datetime, timedelta, timezone

from db import logs_col


REQUIRED_REJECTION_FIELDS = [
    "rejection_reason",
    "failing_metric",
    "failing_value",
    "threshold",
    "metrics",
]


def _safe_pct(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return (numerator / denominator) * 100.0


def analyze_log_quality(days: int = 30, version: str = None):
    if logs_col is None:
        print("Error: MongoDB logs collection is unavailable. Check MONGO_URI.")
        return

    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    query = {"timestamp": {"$gte": cutoff}}
    if version:
        query["version"] = str(version)

    projection = {
        "_id": 0,
        "scanner": 1,
        "symbol": 1,
        "action": 1,
        "log_type": 1,
        "version": 1,
        "details": 1,
        "timestamp": 1,
    }

    docs = list(logs_col.find(query, projection))
    total_logs = len(docs)

    print("=" * 58)
    print("DB Log Quality Report (Telemetry + Filters)")
    print("=" * 58)
    print(f"Lookback days      : {days}")
    print(f"Version filter     : {version if version else 'ALL'}")
    print(f"Total logs scanned : {total_logs}")

    if total_logs == 0:
        print("No logs found in the selected window.")
        return

    by_scanner = Counter()
    by_action = Counter()
    rejected_reasons = Counter()

    rejected_total = 0
    accepted_total = 0
    system_total = 0

    missing_required = Counter()
    details_empty = 0
    malformed_metrics = 0

    for doc in docs:
        scanner = doc.get("scanner", "unknown")
        action = doc.get("action", "unknown")
        log_type = doc.get("log_type", "unknown")
        details = doc.get("details") or {}

        by_scanner[scanner] += 1
        by_action[action] += 1

        if not details:
            details_empty += 1

        if action == "SCAN_COMPLETED":
            system_total += 1

        if log_type == "REJECTED" or action in ("REJECTED_BREAKOUT", "PENDING_REJECTED"):
            rejected_total += 1
            reason = details.get("rejection_reason", details.get("reason", "unknown"))
            rejected_reasons[reason] += 1

            for field in REQUIRED_REJECTION_FIELDS:
                if field not in details or details.get(field) in (None, "", {}):
                    missing_required[field] += 1

            if "metrics" in details and not isinstance(details.get("metrics"), dict):
                malformed_metrics += 1

        if action in ("ACCEPTED_BREAKOUT", "BREAKOUT_SIGNAL", "SIGNAL_GENERATED"):
            accepted_total += 1

    print("\nVolume by scanner:")
    for name, count in by_scanner.most_common():
        print(f"  - {name}: {count}")

    print("\nTop actions:")
    for name, count in by_action.most_common(8):
        print(f"  - {name}: {count}")

    print("\nFunnel snapshot:")
    print(f"  - Accepted events : {accepted_total}")
    print(f"  - Rejected events : {rejected_total}")
    print(f"  - System events   : {system_total}")
    if accepted_total + rejected_total > 0:
        acceptance_rate = _safe_pct(accepted_total, accepted_total + rejected_total)
        print(f"  - Acceptance rate : {acceptance_rate:.2f}%")

    print("\nRejection reason ranking (top 10):")
    if rejected_reasons:
        for reason, count in rejected_reasons.most_common(10):
            pct = _safe_pct(count, rejected_total)
            print(f"  - {reason}: {count} ({pct:.2f}%)")
    else:
        print("  - No rejection records found.")

    print("\nTelemetry quality checks:")
    print(f"  - Empty details payloads: {details_empty} ({_safe_pct(details_empty, total_logs):.2f}%)")
    print(f"  - Malformed metrics map : {malformed_metrics}")

    if rejected_total > 0:
        print("  - Rejection field completeness:")
        for field in REQUIRED_REJECTION_FIELDS:
            missing = missing_required[field]
            present = rejected_total - missing
            print(
                f"    * {field}: present={present} "
                f"({_safe_pct(present, rejected_total):.2f}%), missing={missing}"
            )

    print("\nInterpretation:")
    if rejected_total == 0:
        print("  - No rejection data means filter diagnostics are insufficient for optimization.")
    else:
        primary_reason, primary_count = rejected_reasons.most_common(1)[0]
        print(
            f"  - Most dominant filter failure is '{primary_reason}' "
            f"({primary_count}/{rejected_total})."
        )
        if missing_required["rejection_reason"] > 0 or missing_required["metrics"] > 0:
            print("  - Some rejection logs are incomplete; fix these before monthly optimization decisions.")
        else:
            print("  - Rejection telemetry is structurally healthy for monthly filter analysis.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze MongoDB log quality and filter outcomes.")
    parser.add_argument("--days", type=int, default=30, help="Lookback window in days (default: 30).")
    parser.add_argument("--version", type=str, default=None, help="Optional scanner version filter, e.g. 2.2.0.")
    args = parser.parse_args()
    analyze_log_quality(days=args.days, version=args.version)
