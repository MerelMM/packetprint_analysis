# This script was developed with assistance from ChatGPT (OpenAI) and Github Copilot
# Final implementation and adaptation by Merel Haenaets.
from .packet_size_filtering import filter_packet_size_for_app
import os
import pickle
from helper.helper import get_app_key


def run_preprocessing(
    app_key: str,
    load_existing: bool,
    load_existing_filtering: bool = True,
    data_path: str = "capture_data_train",
    save_path: str = "data/split_filtered_data.pkl",
):
    """
    Filters all sessions using the packet size filter of a selected app.

    Args:
        app (str): The app whose packet size filter will be used across all sessions.
        load_existing (bool): Whether to load previously saved filtered data.
        load_existing_filtering (bool): Whether to load a cached filtering dictionary.
        data_path (str): Directory containing capture sessions.
        save_path (str): Base path to save filtered data (will be suffixed with _{app}.pkl).

    Returns:
        dict: Mapping from session name to filtered (timestamp, packet_size) tuples.
    """

    # Adjust final save path per app
    app_save_path = f"{os.path.splitext(save_path)[0]}_{app_key}.pkl"

    if load_existing and os.path.exists(app_save_path):
        with open(app_save_path, "rb") as f:
            print(f"Loading existing filtered data for app: {app_key}")
            return pickle.load(f)

    filtered_sizes_per_apps = filter_packet_size_for_app(
        app_key=app_key, path=data_path, load_existing=load_existing_filtering
    )

    keep_set = set(filtered_sizes_per_apps)
    if not keep_set:
        print(f"Warning: No packet sizes found for app: {app_key}")
        return {}

    filtered_packets = {}

    for session_dir in os.listdir(data_path):
        session_path = os.path.join(data_path, session_dir)
        if not os.path.isdir(session_path):
            continue

        size_file = os.path.join(session_path, "packet_sizes.txt")
        time_file = os.path.join(session_path, "packet_timings.txt")
        if not (os.path.exists(size_file) and os.path.exists(time_file)):
            continue

        with open(size_file) as sf, open(time_file) as tf:
            sizes = [int(s.strip()) for s in sf if s.strip()]
            times = [float(t.strip()) for t in tf if t.strip()]

        if len(sizes) != len(times):
            print(f"Warning: size/timing mismatch in {session_dir}")
            continue

        # Apply the filtering
        filtered = [(t, s) for t, s in zip(times, sizes) if s in keep_set]
        filtered_packets[session_dir] = filtered

    os.makedirs(os.path.dirname(app_save_path), exist_ok=True)
    with open(app_save_path, "wb") as f:
        pickle.dump(filtered_packets, f)

    print(f"Saved filtered data for app '{app_key}' to {app_save_path}")
    return filtered_packets
