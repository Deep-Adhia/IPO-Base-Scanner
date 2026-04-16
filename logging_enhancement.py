def log_position_snapshot(data):
    """
    Log the POSITION_SNAPSHOT event with the provided data.
    """
    print(f"[POSITION_SNAPSHOT] - {data}")


def log_stop_loss_updated(data):
    """
    Log the STOP_LOSS_UPDATED event with the provided data.
    """
    print(f"[STOP_LOSS_UPDATED] - {data}")


def log_position_scan_completed(data):
    """
    Log the POSITION_SCAN_COMPLETED event with the provided data.
    """
    print(f"[POSITION_SCAN_COMPLETED] - {data}")


# Example usage:
# log_position_snapshot({'position': 'AAPL', 'value': 150})
# log_stop_loss_updated({'position': 'AAPL', 'new_stop_loss': 145})
# log_position_scan_completed({'batch': 1, 'total_positions': 100})
