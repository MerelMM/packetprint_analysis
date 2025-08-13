import os
import pickle


def _combine_captured_data(path="capture_data"):
    """
    Reads directional packet size sequences for each app directory in the given path.

    Assumes each app has a subdirectory under `path` and contains a file named 'packet_sizes.txt'.

    Returns:
        dict: Mapping from app name (folder) to list or array of packet sizes.
    """
    packet_sizes = {}

    for app_dir in os.listdir(path):
        app_path = os.path.join(path, app_dir)
        if not os.path.isdir(app_path):
            continue  # Skip non-directories (e.g., .DS_Store)

        file_path = os.path.join(app_path, "packet_sizes.txt")
        if not os.path.exists(file_path):
            print(f"Warning: File missing in {app_path}")
            continue

        with open(file_path, "r") as f:
            # Assuming one packet size per line
            sizes = [int(line.strip()) for line in f if line.strip()]
            packet_sizes[app_dir] = sizes

    return packet_sizes


def get_capture_durations(path):
    capture_durations = {}

    for app_dir in os.listdir(path):
        app_path = os.path.join(path, app_dir)
        if not os.path.isdir(app_path):
            continue  # Skip non-directories

        file_path = os.path.join(app_path, "capture_time_range.txt")
        if not os.path.exists(file_path):
            print(f"Warning: capture_time_range.txt missing in {app_path}")
            continue

        with open(file_path, "r") as f:
            line = f.readline().strip()
            if not line:
                print(f"Warning: empty capture_time_range.txt in {app_path}")
                continue

            try:
                start, end, duration = map(float, line.split())
                capture_durations[app_dir] = {
                    "start": start,
                    "end": end,
                    "duration": duration,
                }
            except ValueError:
                print(f"Warning: Malformed line in {file_path}: {line}")

    return capture_durations


def get_all_packages(path):
    cache_path = "data/packet_splits.pkl"

    if os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            return pickle.load(f)
    else:
        return _combine_captured_data(path)
