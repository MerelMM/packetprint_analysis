import os
import random
import shutil
from collections import defaultdict
from helper.helper import get_app_key


def train_test_split_sessions(
    data_path="capture_data",
    train_path="capture_data_train",
    test_path="capture_data_test",
    train_ratio=0.8,
    seed=42,
):
    """
    Moves session folders from data_path into train/test folders (balanced per app).

    Args:
        data_path (str): Path containing all session folders.
        train_path (str): Destination folder for training sessions.
        test_path (str): Destination folder for testing sessions.
        train_ratio (float): Fraction of sessions per app to use for training.
        seed (int): Random seed for reproducibility.
    """
    random.seed(seed)
    remove_empty_dirs(data_path)
    os.makedirs(train_path, exist_ok=True)
    os.makedirs(test_path, exist_ok=True)

    # Group sessions by app
    app_sessions = defaultdict(list)
    for session_dir in os.listdir(data_path):
        session_path = os.path.join(data_path, session_dir)
        if not os.path.isdir(session_path):
            continue
        app_name = get_app_key(session_dir)
        app_sessions[app_name].append(session_dir)

    for app, sessions in app_sessions.items():
        sessions_sorted = sorted(sessions)  # consistent order
        random.shuffle(sessions_sorted)
        split_idx = int(len(sessions_sorted) * train_ratio)
        train_set = sessions_sorted[:split_idx]
        test_set = sessions_sorted[split_idx:]

        # Move the folders
        for session in train_set:
            src = os.path.join(data_path, session)
            dst = os.path.join(train_path, session)
            shutil.move(src, dst)

        for session in test_set:
            src = os.path.join(data_path, session)
            dst = os.path.join(test_path, session)
            shutil.move(src, dst)

    print(
        f"Moved sessions to:\n  {train_path}: {len(os.listdir(train_path))} sessions\n  {test_path}: {len(os.listdir(test_path))} sessions"
    )


"""
    Initial length of the open world traces captured were 5000.
    To follow PP, we will also work with same size traces.
    And then, which is absent in PP, we will test how well the whole architecture generalizes
    to traces of different length.
"""


def remove_empty_dirs(path="capture_data"):
    """
    Recursively remove all empty directories under the given path.
    """
    removed = 0
    for root, dirs, files in os.walk(path, topdown=False):
        for dir_name in dirs:
            full_path = os.path.join(root, dir_name)
            if not os.listdir(full_path):
                os.rmdir(full_path)
                print(f"Removed empty directory: {full_path}")
                removed += 1
    print(f"Done. Removed {removed} empty directories.")


def add_open_world_traces_to_train_test(
    data_path="capture_data_many_apps",
    train_path="capture_data_train",
    test_path="capture_data_test",
    train_ratio=0.5,
    seed=42,
):
    """
    Adds one session per open-world app (apps not seen in original train/test) to either
    train or test set, by copying and cropping them to half-length.
    The original data remains untouched.
    """
    random.seed(seed)
    remove_empty_dirs(data_path)
    os.makedirs(train_path, exist_ok=True)
    os.makedirs(test_path, exist_ok=True)

    # Step 1: Identify known apps already in train/test
    known_apps = set()
    for split_path in [train_path, test_path]:
        for session_dir in os.listdir(split_path):
            if os.path.isdir(os.path.join(split_path, session_dir)):
                known_apps.add(get_app_key(session_dir))
    print(f"Found {len(known_apps)} known apps in train/test.")

    # Step 2: Collect one session per open-world app
    open_world_app_sessions = {}
    for session_dir in os.listdir(data_path):
        session_path = os.path.join(data_path, session_dir)
        if not os.path.isdir(session_path):
            continue
        app_name = get_app_key(session_dir)
        if app_name in known_apps:
            continue
        if app_name not in open_world_app_sessions:
            open_world_app_sessions[app_name] = session_dir

    print(f"Identified {len(open_world_app_sessions)} open-world apps.")

    # Step 3: Split apps into train/test
    all_open_apps = list(open_world_app_sessions.keys())
    random.shuffle(all_open_apps)
    split_idx = int(len(all_open_apps) * train_ratio)
    train_apps = all_open_apps[:split_idx]
    test_apps = all_open_apps[split_idx:]

    # Step 4: Copy + crop each session
    for app in train_apps:
        session = open_world_app_sessions[app]
        src = os.path.join(data_path, session)
        dst = os.path.join(train_path, session)
        shutil.copytree(src, dst)
        crop_session_half(dst)

    for app in test_apps:
        session = open_world_app_sessions[app]
        src = os.path.join(data_path, session)
        dst = os.path.join(test_path, session)
        shutil.copytree(src, dst)
        crop_session_half(dst)

    print(
        f"Copied and cropped open-world sessions:\n"
        f"  To {train_path}: +{len(train_apps)} sessions\n"
        f"  To {test_path}: +{len(test_apps)} sessions"
    )


def crop_session_half(session_path):
    """
    Modifies the files in session_path to keep only the first half of the trace.
    """
    # Crop timings
    timings_file = os.path.join(session_path, "packet_timings.txt")
    sizes_file = os.path.join(session_path, "packet_sizes.txt")
    range_file = os.path.join(session_path, "capture_time_range.txt")

    with open(timings_file) as f:
        timings = [line.strip() for line in f if line.strip()]
    with open(sizes_file) as f:
        sizes = [line.strip() for line in f if line.strip()]

    # Keep only the first half
    half_len = min(len(timings), len(sizes)) // 2
    timings = timings[:half_len]
    sizes = sizes[:half_len]

    with open(timings_file, "w") as f:
        f.write("\n".join(timings) + "\n")
    with open(sizes_file, "w") as f:
        f.write("\n".join(sizes) + "\n")

    # Update time range
    with open(range_file) as f:
        line = next(f).strip()  # Read the single line
        start_str, end_str, duration_str = line.split()

    start = float(start_str)
    duration = float(duration_str)
    new_duration = duration / 2
    new_end = start + new_duration

    with open(range_file, "w") as f:
        f.write(f"{start:.9f} {new_end:.9f} {new_duration:.9f}\n")


# helper function, just did something wrong earlier
# def remove_sessions_by_date(
#     data_path="capture_data_test", dates_to_remove=("20250721", "20250725")
# ):
#     """
#     Remove session folders from the given path whose names contain any of the specified date strings (yyyymmdd).
#     """
#     removed = 0
#     for folder in os.listdir(data_path):
#         folder_path = os.path.join(data_path, folder)
#         if not os.path.isdir(folder_path):
#             continue
#         if any(date in folder for date in dates_to_remove):
#             shutil.rmtree(folder_path)
#             print(f"Removed: {folder_path}")
#             removed += 1
#     print(f"Done. Removed {removed} folders.")
