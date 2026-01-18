
import sys
import logging
import os

# Set up logging to console
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

# Ensure the current directory is in the path so imports work
sys.path.append(os.getcwd())

try:
    from listing_day_breakout_scanner import update_listing_data_for_new_ipos
    print("Starting IPO listing data update...")
    update_listing_data_for_new_ipos()
    print("Update finished.")
except Exception as e:
    print(f"Error: {e}")
