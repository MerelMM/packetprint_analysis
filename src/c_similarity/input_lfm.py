from typing import List, Tuple, Dict, Any, Set
import numpy as np
from sklearn.tree import DecisionTreeClassifier
import math
import pickle
import numpy as np
from collections import defaultdict
import json
import os
import pandas as pd


def create_N_grams(packet_sizes, N):
    """
    Creates N-gram representations from a list of packet sizes.

    Args:
        packet_sizes (list[int]): List of packet sizes.
        N (int): Half-width of the N-gram window (total width = 2N+1).

    Returns:
        np.ndarray: Array of shape (len(packet_sizes) - 2N, 2N+1) with N-gram features.
    """
    if len(packet_sizes) < 2 * N + 1:
        return np.empty((0, 2 * N + 1))

    return np.array(
        [packet_sizes[i - N : i + N + 1] for i in range(N, len(packet_sizes) - N)]
    )


def construct_input_lfm_training(
    app_name: str,
    segment: List[int],
    segment_times: List[float],
    segment_labels: List[int],
    N: int = 1,
) -> List[Dict[str, Any]]:
    """
    Constructs hierarchical BoW-style features from a single segment.

    Hierarchy:
      - Packet-level (N-grams)
      - Burst-level (1-second windows)
      - Behavior-level (5-second windows)

    Args:
        segment: List of packet sizes.
        segment_times: Corresponding packet timestamps.
        segment_labels: 0/1 labels per packet (1 = App A).
        N: Half-width of the N-gram window.

    Returns:
    Tuple of:
        - t0: List[np.ndarray] of packet-level N-gram features
        - t1: List[Dict] of burst-level features and labels
        - t5: List[Dict] of behavior-level features and labels
    """
    t0 = []
    t1 = []
    t5 = []
    window_1s = 1.0
    window_5s = 5.0
    step = 1.0

    if not segment or not segment_times or len(segment) < 2 * N + 1:
        return None
    # something fishy seems to be going on with the pf -> stored??? should labels not be per packet zodat weet of 1gram label1
    # s = 1: packet-level features + ANY-in-window labels
    packet_features = create_N_grams(segment, N)  # shape (L-2N, 2N+1)
    packet_labels = make_ngram_labels(segment_labels, app_name, N)  # shape (L-2N,)
    t0 = {"features": packet_features, "labels": packet_labels}

    if N > 0 and packet_features.shape[0] > 0:
        padding = np.tile(packet_features[0], (N, 1))
        packet_features = np.vstack([padding, packet_features])

    t_start = segment_times[0]
    t_end = segment_times[-1]
    t = t_start
    offset_burst = 0
    offset_behavior = 0
    while t + window_1s <= t_end:
        # also appends empty burst and behavior windows,
        # this is for the next phase, but could potentially be implemented more elegant (by not saving empty dicts but instead a behavior windowidentifier)
        # possible to clean out and only keep empty bursts in non empty behavior windows but alasalas not implemented
        offset_behavior = offset_burst
        (offset_burst, burst_ixs) = get_time_ixs(
            segment_times, t, window_1s, offset_burst
        )

        if t + window_5s <= t_end:
            (_, behavior_ixs) = get_time_ixs(
                segment_times,
                t,
                window_5s,
                offset_behavior,  # old offset_burst is also offset here, since windows both slide 1 second, which also is the length of the burst
            )
            behavior_features = get_features_from_ixs(
                packet_features=packet_features, ixs=behavior_ixs, N=N
            )

            label_behavior = get_label(segment_labels, behavior_ixs)
            # if len(behavior_features) == 0:
            #     print("geflaggerd but exactly what i want")
            t5.append(
                {
                    "behavior_features": behavior_features,
                    "label": label_behavior,
                }
            )

        burst_features = get_features_from_ixs(
            packet_features=packet_features, ixs=burst_ixs, N=N
        )
        label_burst = get_label(segment_labels, burst_ixs)
        t1.append(
            {
                "burst_features": burst_features,
                "label": label_burst,
            }
        )

        t += step
    return t0, t1, t5


def make_ngram_labels(segment_labels, app_name, N):
    labels = []
    for i in range(1, len(segment_labels) - N):
        if app_name in segment_labels[i - N : i + N + 1]:
            labels.append(1)
        else:
            labels.append(0)
    return labels


def get_label(segment_labels, ixs):
    for i in ixs:
        if segment_labels[i] == 1:
            return 1
    return 0


def get_time_ixs(segment_times, t, window, offset_ix):
    end = t + window
    ixs = []

    for i in range(offset_ix, len(segment_times)):

        time = segment_times[i]
        assert time >= t, f"Timestamp {time} < window start {t} — offset logic broken"

        if time >= end:  # if current time is bigger then windowend -> break
            break
        ixs.append(i)

    offset_ix = ixs[-1] + 1 if ixs else offset_ix
    return offset_ix, ixs


def get_features_from_ixs(packet_features, ixs, N):
    if len(ixs) > 2 * N:
        return packet_features[ixs[N:-N]]
    else:
        return np.empty(0)


def get_lfm_features_per_app(
    app: str, input_dir: str = "data/lfm_features"
) -> List[Dict[str, Any]]:
    """Load LFM features for a single app from <input_dir>/<app>.pkl."""
    if len(app) >= 3 and ".pkl" in app:
        path = os.path.join(input_dir, app)
    else:
        path = os.path.join(input_dir, f"{app}.pkl")
    if not os.path.exists(path):
        raise FileNotFoundError(f"No LFM features found for app '{app}' at {path}")
    with open(path, "rb") as f:
        return pickle.load(f)


def get_lfm_features_per_app_segmented(app):
    try:
        with open(f"data/lfm_features_segmented/{app}", "rb") as f:
            return pickle.load(f)
    except:
        Exception(f"App didn't have lfm input ready at data/lfm_features/{app}.pkl")


def construct_input_lfm_per_app_segmented(
    app: str,
    segments: List[Dict[str, Any]] = None,
    N: int = 1,
    output_dir: str = "data/lfm_features_segmented",
    load_features: bool = False,
) -> List[Dict[str, Any]]:
    """
    Build hierarchical features per segment for a single app and save as a list:
      [
        {
          "packet_lvl":   {"packets": np.ndarray, "labels": np.ndarray} | None,
          "burst_lvl":    [ {"burst_features": np.ndarray, "label": int}, ... ],
          "behavior_lvl": [ {"behavior_features": np.ndarray, "label": int}, ... ]
        },
        ...
      ]
    """
    os.makedirs(output_dir, exist_ok=True)

    if load_features:
        return get_lfm_features_per_app_segmented(app, output_dir)

    all_features: List[Dict[str, Any]] = []

    for seg in segments:
        print(f"Processing segment for {app}")

        res = construct_input_lfm_training(
            app, seg["packet_sizes"], seg["timestamps"], seg["labels"], N=N
        )
        if res is None:
            continue

        packets, bursts, behavior = res

        # packet-level features
        X = packets["features"]  # shape (n, 2N+1)
        y = packets["labels"]  # shape (n,)
        if isinstance(X, np.ndarray) and X.ndim == 2 and X.size > 0:
            packet_data = {"packets": X, "labels": y}
        else:
            packet_data = None

        segment_features = {
            "packet_lvl": packet_data,
            "burst_lvl": bursts if bursts else [],
            "behavior_lvl": behavior if behavior else [],
        }

        all_features.append(segment_features)

    out_path = os.path.join(output_dir, f"{app}")
    with open(out_path, "wb") as f:
        pickle.dump(all_features, f)
    print(f"Saved per-segment features for {app} -> {out_path}")

    return all_features


def construct_input_lfm_per_app(
    app: str,
    segments: List[Dict[str, Any]] = None,
    N: int = 1,
    output_dir: str = "data/lfm_features",
    load_features: bool = False,
) -> Dict[str, Any]:
    """
    Build hierarchical features for a single app and save as one dict:
      {
        "packet_lvl":   np.ndarray | None,   # vstack of all packet arrays, shape (sum_rows, 2N+1)
        "burst_lvl":    [ {"burst_features": np.ndarray, "label": int}, ... ],
        "behavior_lvl": [ {"behavior_features": np.ndarray, "label": int}, ... ]
      }

    Assumes construct_input_lfm_training returns: (packet_array, bursts_list, behavior_list),
    where packet_array is a 2D ndarray for that segment (not wrapped in a list).
    """
    os.makedirs(output_dir, exist_ok=True)

    if load_features:
        return get_lfm_features_per_app(app, output_dir)

    pkts: List[np.ndarray] = []
    pkts_labels: List[np.ndarray] = []
    bursts_all: List[Dict[str, Any]] = []
    behavior_all: List[Dict[str, Any]] = []

    for seg in segments:
        print(f"Processing {app}")
        res = construct_input_lfm_training(
            seg["packet_sizes"], seg["timestamps"], seg["labels"], N=N
        )
        if res is None:
            continue

        packets, bursts, behavior = res

        # use correct keys from packet dict
        X = packets["features"]  # 2D
        y = packets["labels"]  # 1D

        if isinstance(X, np.ndarray) and X.ndim == 2 and X.size > 0:
            pkts.append(X)
            pkts_labels.append(y)

        # bursts / behavior: just extend flat lists
        if bursts:
            bursts_all.extend(bursts)
        if behavior:
            behavior_all.extend(behavior)

    packet_all = {
        "packets": np.vstack(pkts) if pkts else None,
        "labels": np.concatenate(pkts_labels) if pkts_labels else None,
    }

    agg = {
        "packet_lvl": packet_all,
        "burst_lvl": bursts_all,
        "behavior_lvl": behavior_all,
    }

    out_path = os.path.join(output_dir, f"{app}.pkl")
    with open(out_path, "wb") as f:
        pickle.dump(agg, f)
    print(f"Saved aggregated features for {app} -> {out_path}")

    return agg


def construct_input_lfm_all_apps(
    segments_all_apps: Dict[str, List[Dict[str, Any]]],
    N: int = 1,
    output_dir: str = "data/lfm_features",
    load_features: bool = False,
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Per-app wrapper: iterates apps and uses construct_input_lfm_per_app for each.
    Returns {app: features}.
    """
    os.makedirs(output_dir, exist_ok=True)
    out: Dict[str, List[Dict[str, Any]]] = {}

    for app, segments in segments_all_apps.items():
        out[app] = construct_input_lfm_per_app(
            app=app,
            segments=segments,
            N=N,
            output_dir=output_dir,
            load_features=load_features,
        )

    return out


def construct_input_lfm_all_apps_segmented(
    segments_all_apps: Dict[str, List[Dict[str, Any]]],
    N: int = 1,
    output_dir: str = "data/lfm_features_segmented",
    load_features: bool = False,
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Per-app wrapper: iterates apps and uses construct_input_lfm_per_app for each.
    Returns {app: features}.
    """
    os.makedirs(output_dir, exist_ok=True)
    out: Dict[str, List[Dict[str, Any]]] = {}

    for app, segments in segments_all_apps.items():
        out[app] = construct_input_lfm_per_app_segmented(
            app=app,
            segments=segments,
            N=N,
            output_dir=output_dir,
            load_features=load_features,
        )

    return out


# def load_lf_features_app(app, input_dir="data/lfm_features") -> Dict[str, list]:
#     """
#     Loads app-level LFM features from disk for a specific app.

#     Returns:
#         dict: {app_name: list of feature dicts}
#     """
#     features = {}
#     filename = app + ".pkl"
#     app = filename.replace(".pkl", "")
#     with open(os.path.join(input_dir, filename), "rb") as f:
#         features[app] = pickle.load(f)
#     return features


# def load_lfm_features(input_dir="data/lfm_features") -> Dict[str, list]:
#     """
#     Loads all saved app-level LFM features from disk.

#     Returns:
#         dict: {app_name: list of feature dicts}
#     """
#     features = {}
#     for filename in os.listdir(input_dir):
#         if filename.endswith(".pkl"):
#             app = filename.replace(".pkl", "")
#             with open(os.path.join(input_dir, filename), "rb") as f:
#                 features[app] = pickle.load(f)
#     return features
