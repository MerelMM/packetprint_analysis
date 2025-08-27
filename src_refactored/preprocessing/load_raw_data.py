"""File for loading in the raw data."""

from collections import defaultdict
import os
from helper.helper import get_app_key


def get_frame_sizes_per_app(path="capture_data_train"):
    """
    Combines packet sizes from all sessions into per-app lists.
    """
    app_packet_sizes = defaultdict(list)

    for session_dir in os.listdir(path):
        full_path = os.path.join(path, session_dir)
        if not os.path.isdir(full_path):
            continue

        app_key = get_app_key(session_dir)
        packet_file = os.path.join(full_path, "packet_sizes.txt")
        if not os.path.exists(packet_file):
            print(f"Warning: packet_sizes.txt missing in {full_path}")
            continue

        with open(packet_file, "r") as f:
            sizes = [int(line.strip()) for line in f if line.strip()]
            app_packet_sizes[app_key].extend(sizes)

    return dict(app_packet_sizes)


def get_capture_durations_per_app(path="data_path_train"):
    """
    Combines durations from all sessions into total durations per app.
    """
    app_durations = defaultdict(lambda: {"duration": 0.0})

    for session_dir in os.listdir(path):
        full_path = os.path.join(path, session_dir)
        if not os.path.isdir(full_path):
            continue

        app_key = get_app_key(session_dir)
        capture_file = os.path.join(full_path, "capture_time_range.txt")
        if not os.path.exists(capture_file):
            print(f"Warning: capture_time_range.txt missing in {full_path}")
            continue

        with open(capture_file, "r") as f:
            line = f.readline().strip()
            try:
                start, end, duration = map(float, line.split())
                app_durations[app_key]["duration"] += duration
            except ValueError:
                print(f"Warning: malformed line in {capture_file}: {line}")

    return dict(app_durations)
