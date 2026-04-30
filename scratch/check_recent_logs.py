import argparse
from datetime import datetime, timedelta, timezone
from db import logs_col

def check_recent(days: int = 1, limit: int = 20):
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    query = {"timestamp": {"$gte": cutoff}}
    
    logs = list(logs_col.find(query, {"scanner": 1, "action": 1, "timestamp": 1, "symbol": 1, "_id": 0})
                .sort("timestamp", -1)
                .limit(limit))
    
    print(f"--- Recent Logs (Last {days} day(s), max {limit}) ---")
    print(f"Current UTC time: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')}")
    print("-" * 60)
    
    if not logs:
        print(f"No logs found in the last {days} day(s).")
        return

    for l in logs:
        ts = l['timestamp'].strftime('%Y-%m-%d %H:%M:%S')
        symbol = l.get('symbol', 'N/A')
        print(f"{ts} | {l['scanner']:<15} | {symbol:<12} | {l['action']}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Check recent logs from MongoDB.")
    parser.add_argument("--days", type=int, default=1, help="Lookback window in days (default: 1).")
    parser.add_argument("--limit", type=int, default=20, help="Max number of logs to display (default: 20).")
    args = parser.parse_args()
    
    check_recent(days=args.days, limit=args.limit)
