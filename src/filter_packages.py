from typing import List, Tuple, Dict, Set
from load_data import (
    get_all_packages,
    get_capture_durations,
)
from collections import defaultdict
import json
from pathlib import Path
import os
import pickle
from s_similarity import extract_app_name


def _construct_SpA_greedy(
    Sp: List,  # S_p = {si} the set of all observed directional packet sizes
    r_pos,  # the outbound packets
    r_neg,  # the inbound packets
    R_min: float = 2.0,  # the expected minimum packet arrival rate from app A
    Ms_min: int = 200,  # the minimum number of different directional packet sizes to retain
) -> List[Tuple[str, int]]:
    """
    Construct app-specific directional packet size list SpA using greedy optimization (Algorithm 1).

    Args:
        Sp: List of all observed directional packet sizes as (direction, size) tuples.
        r_pos: Dict mapping directional packet size to its arrival rate from app A (r+).
        r_neg: Dict mapping directional packretainet size to its arrival rate from other apps (r-).
        R_min: Minimum total arrival rate to  from app A.
        Ms_min: Minimum number of packet sizes to retain.

    Returns:
        List of selected directional packet sizes (SpA)
    """

    SpA = set(Sp)
    Sp_prime = set(Sp)

    # Minimum r+ requirement (either sum of all or R_min)
    total_r_pos = sum(r_pos.get(si, 0.0) for si in Sp)
    rmin = min(total_r_pos, R_min)

    while len(SpA) > Ms_min and len(Sp_prime) > 0:
        # Select si with highest r-/r+ ratio
        def noise_ratio(si):
            r_plus = r_pos.get(si, 1e-6)  # prevent division by 0
            r_minus = r_neg.get(si, 0.0)
            return r_minus / r_plus

        si_star = max(Sp_prime, key=noise_ratio)

        # Check if we can safely remove it
        SpA_candidate = SpA - {si_star}
        new_r_pos_sum = sum(r_pos.get(s, 0.0) for s in SpA_candidate)

        if new_r_pos_sum >= rmin:
            SpA.remove(si_star)

        Sp_prime.remove(si_star)

    return sorted(SpA)


def filter_packages(path="capture_data"):
    filtered_path = Path("data/filtered_packet_sizes.json")

    all_packets = get_all_packages(path)  # {app_key: [size1, size2, ...]}
    durations = get_capture_durations(path)  # {app_key: {'duration': float}}

    Sp = set()
    R_min = 2.0
    Ms_min = 200

    # 1. Compute global r_neg_all = combined rate of all directional sizes (all apps)
    r_neg_all = defaultdict(float)

    for app, packet_list in all_packets.items():
        app_duration = durations[app]["duration"]
        counts = defaultdict(int)
        Sp.update(set(packet_list))
        for size in packet_list:
            counts[size] += 1
        for size in counts:
            r_neg_all[size] += counts[size] / app_duration

    # 2. Construct SpA for each app
    Sp_app = {}

    for app_key, packet_sizes in all_packets.items():
        total_duration = durations[app_key]["duration"]

        # r_pos for app
        r_pos = defaultdict(float)
        for size in packet_sizes:
            r_pos[size] += 1
        for size in r_pos:
            r_pos[size] /= total_duration

        # r_neg = r_neg_all - this app's contribution
        app_r_neg = defaultdict(float)
        for size in packet_sizes:
            app_r_neg[size] += 1
        for size in app_r_neg:
            app_r_neg[size] /= total_duration

        r_neg = r_neg_all.copy()
        for size in app_r_neg:
            r_neg[size] -= app_r_neg[size]
            if r_neg[size] < 0:
                r_neg[size] = 0.0

        # Greedy filter
        Sp_app[app_key] = _construct_SpA_greedy(list(Sp), r_pos, r_neg, R_min, Ms_min)

    # 3. Save to JSON
    with open(filtered_path, "w") as f:
        json.dump({app: sorted(sizes) for app, sizes in Sp_app.items()}, f, indent=2)

    return Sp_app


def get_filtered_data(
    filter_path: str = "data/filtered_packet_sizes.json",
    data_path: str = "capture_data",
    train_ratio: float = 0.8,
    save_path: str = "data/split_filtered_data.pkl",
    load_existing=False,
):
    """
    Combines all session captures for the same app (timings + sizes),
    applies packet size filtering, and splits sequentially into train/test sets.
    """
    filter_file = Path(filter_path)
    save_file = Path(save_path)
    if load_existing:  # do not calculate everything again
        if save_file.exists():
            with open(save_file, "rb") as f:
                return pickle.load(f)

    # else if not exists, have to compute
    # but check if filtering has been done -> don't have to redo this then
    if load_existing and filter_file.exists():
        with open(filter_file) as f:
            filtered_sizes = json.load(f)
    else:
        filtered_sizes = filter_packages()

    # Convert size strings to int
    filtered_sizes = {
        app: set(int(size) for size in sizes) for app, sizes in filtered_sizes.items()
    }

    # Combine packets per app across all session dirs
    combined_packets = defaultdict(list)
    for session_dir in os.listdir(data_path):
        session_path = os.path.join(data_path, session_dir)
        if not os.path.isdir(session_path):
            continue

        # parts = session_dir.split("_")
        # app_name = parts[1] if len(parts) > 1 else session_dir

        size_file = os.path.join(session_path, "packet_sizes.txt")
        time_file = os.path.join(session_path, "packet_timings.txt")
        if not (os.path.exists(size_file) and os.path.exists(time_file)):
            continue

        with open(size_file) as sf, open(time_file) as tf:
            sizes = [int(s.strip()) for s in sf if s.strip()]
            times = [float(t.strip()) for t in tf if t.strip()]

        if len(sizes) != len(times):
            print(f"Warning: mismatch in {session_dir}")
            continue

        # Combine (timestamp, size)
        combined_packets[session_dir].extend(zip(times, sizes))

    # Sort per app by timestamp -- is actually already done due to the capturing, but just to be sure
    for app in combined_packets:
        combined_packets[app].sort(key=lambda x: x[0])

    # Apply filtering and split
    split_filtered_data = {}
    for app, packet_list in combined_packets.items():
        keep_set = filtered_sizes.get(app, set())
        # Filter by size, keep timestamps
        filtered = [(t, s) for (t, s) in packet_list if s in keep_set]

        split_idx = int(len(filtered) * train_ratio)
        split_filtered_data[extract_app_name(app)] = {
            "train": filtered[:split_idx],
            "test": filtered[split_idx:],
        }
    # Save to file
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "wb") as f:
        pickle.dump(split_filtered_data, f)

    print(f"Saved filtered train/test data to {save_path}")

    return split_filtered_data


def get_filtered_data_without_timestamps(
    filter_path: str = "data/filtered_packet_sizes.json",
    data_path: str = "capture_data",
    train_ratio: float = 0.8,
    save_path: str = "data/split_filtered_data_untimed.pkl",
    load_existing=False,
    train=True,  # otherwise test data
):
    """
    Loads filtered training data but drops timestamps, keeping only packet sizes for S-XGBoost training.
    Saves processed dataset to a pickle file for faster reloads.
    """
    data_phase = "train" if train else "test"
    save_path = save_path.replace(".pkl", f"_{data_phase}.pkl")
    save_file = Path(save_path)

    # Load cached untimed data if available
    if load_existing and save_file.exists():
        with open(save_file, "rb") as f:
            print(f"Loaded cached untimed {data_phase} data from {save_file}")
            return pickle.load(f)

    # Get full filtered data with timestamps
    filtered_data_with_timestamps = get_filtered_data(
        filter_path=filter_path,
        data_path=data_path,
        train_ratio=train_ratio,
        save_path="data/split_filtered_data.pkl",  # keep separate cache
        load_existing=load_existing,
    )

    # Strip timestamps, keep only packet sizes
    filtered_data_without_timestamps = defaultdict(list)
    for key, data in filtered_data_with_timestamps.items():
        for time_and_packet in data[data_phase]:
            filtered_data_without_timestamps[key].append(time_and_packet[1])

    # Save untimed data
    save_file.parent.mkdir(parents=True, exist_ok=True)
    with open(save_file, "wb") as f:
        pickle.dump(filtered_data_without_timestamps, f)

    print(f"Saved untimed {data_phase} data to {save_file}")
    return filtered_data_without_timestamps
