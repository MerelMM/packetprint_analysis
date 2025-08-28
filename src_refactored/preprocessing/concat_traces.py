import os
import random
import pickle
from preprocessing.packet_size_filtering import filter_packet_size_for_app
from helper.helper import get_app_key


def concat_traces(
    app_key,
    data_path="capture_data_train",
    save_path="data/concatenated_train_trace.pkl",
    load_existing=True,
    load_existing_filter=True,
    seed=42,
):
    """
    Concatenates session traces from capture_data_train into one continuous trace,
    filters using the size filter for a specific app, and returns packet sizes, timings, and labels.

    Args:
        app_key (str): The app whose filter should be used.
        data_path (str): Path to training session folders.
        save_path (str): Base path to save the result (.pkl will be appended with app_key).
        load_existing (bool): Whether to load from file if it exists.
        load_existing_filter (bool): Whether to reuse existing filters.
        seed (int): Random seed for reproducibility.

    Returns:
        Tuple:
            - concatenated_sizes (List[int])
            - concatenated_timings (List[float])
            - concatenated_labels (List[str])
    """
    # Include app_key in file name
    save_path = save_path.replace(".pkl", f"_{app_key}.pkl")

    if load_existing and os.path.exists(save_path):
        with open(save_path, "rb") as f:
            print(f"Loaded concatenated trace from {save_path}")
            return pickle.load(f)

    random.seed(seed)

    # Load filter once
    keep_set = set(
        filter_packet_size_for_app(
            app_key=app_key, path=data_path, load_existing=load_existing_filter
        )
    )

    if not keep_set:
        raise ValueError(f"No filtered packet sizes found for app '{app_key}'")

    # Shuffle session dirs
    session_dirs = [
        d for d in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, d))
    ]
    random.shuffle(session_dirs)

    # Start accumulation
    concatenated_sizes = []
    concatenated_timings = []
    concatenated_labels = []
    offset = 0.0

    for session_dir in session_dirs:
        session_path = os.path.join(data_path, session_dir)
        size_file = os.path.join(session_path, "packet_sizes.txt")
        time_file = os.path.join(session_path, "packet_timings.txt")

        if not (os.path.exists(size_file) and os.path.exists(time_file)):
            continue

        with open(size_file) as sf, open(time_file) as tf:
            sizes = [int(line.strip()) for line in sf if line.strip()]
            timings = [float(line.strip()) for line in tf if line.strip()]

        if len(sizes) != len(timings):
            print(f"Skipping {session_dir} due to mismatch")
            continue

        session_app = get_app_key(session_dir)

        # Filter using app_key's filter
        filtered = [(s, t) for s, t in zip(sizes, timings) if s in keep_set]
        if not filtered:
            continue

        filtered_sizes, filtered_timings = zip(*filtered)

        # Offset timings
        min_time = filtered_timings[0]
        adjusted_timings = [t - min_time + offset for t in filtered_timings]
        offset = adjusted_timings[-1]

        # Store
        concatenated_sizes.extend(filtered_sizes)
        concatenated_timings.extend(adjusted_timings)
        concatenated_labels.extend([session_app] * len(filtered_sizes))

    result = (concatenated_sizes, concatenated_timings, concatenated_labels)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "wb") as f:
        pickle.dump(result, f)
        print(f"Saved concatenated trace to {save_path}")

    return result
