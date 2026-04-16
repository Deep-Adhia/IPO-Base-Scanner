import json
import os
from datetime import datetime

# Define the log file path based on the current date
log_directory = os.path.join('logs', datetime.utcnow().strftime('%Y-%m-%d'))
if not os.path.exists(log_directory):
    os.makedirs(log_directory)
log_file_path = os.path.join(log_directory, 'positions.jsonl')

# Function to log position snapshot

def log_position_snapshot(position):
    log_entry = {
        'event': 'position_snapshot',
        'timestamp': datetime.utcnow().isoformat(),
        'position': position
    }
    with open(log_file_path, 'a') as log_file:
        log_file.write(json.dumps(log_entry) + '\n')

# Function to log when stop loss is updated

def log_stop_loss_updated(position, new_stop_loss):
    log_entry = {
        'event': 'stop_loss_updated',
        'timestamp': datetime.utcnow().isoformat(),
        'position': position,
        'new_stop_loss': new_stop_loss
    }
    with open(log_file_path, 'a') as log_file:
        log_file.write(json.dumps(log_entry) + '\n')

# Function to log position scan completed

def log_position_scan_completed(scanned_positions):
    log_entry = {
        'event': 'position_scan_completed',
        'timestamp': datetime.utcnow().isoformat(),
        'scanned_positions': scanned_positions
    }
    with open(log_file_path, 'a') as log_file:
        log_file.write(json.dumps(log_entry) + '\n')
