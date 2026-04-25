"""
manage_db.py
Unified entrypoint for MongoDB infrastructure tasks.
"""
import sys
import argparse
import subprocess

def run_script(script_name, args=None):
    cmd = [sys.executable, script_name]
    if args:
        cmd.extend(args)
    print(f"\nRunning: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

def main():
    parser = argparse.ArgumentParser(description="IPO Scanner MongoDB Management Tool")
    parser.add_argument("task", choices=["test", "backfill-all", "validate", "backup"], 
                        help="Task to perform")
    parser.add_argument("--today", action="store_true", help="For validation: logs only for today")

    args = parser.parse_args()

    if args.task == "test":
        run_script("test_db_connection.py")
    
    elif args.task == "backfill-all":
        print("Starting full backfill sequence...")
        run_script("backfill_instrument_keys.py")
        run_script("backfill_metadata.py")
        run_script("mongodb_backfill.py")
        print("\n✅ All backfills complete.")

    elif args.task == "validate":
        v_args = ["--today-logs-only"] if args.today else []
        run_script("compare_csv_vs_db.py", v_args)

    elif args.task == "backup":
        run_script("mongodb_backup.py")

if __name__ == "__main__":
    main()
