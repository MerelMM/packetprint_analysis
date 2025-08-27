from typing import List, Tuple
from .load_raw_data import (
    get_frame_sizes_per_app,
    get_capture_durations_per_app,
)
from collections import defaultdict
import pickle
import os


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


def filter_packet_size_for_app(
    app_key: str,
    path: str = "capture_data_train",
    save_path: str = None,
    load_existing: bool = False,
    R_min: float = 2.0,
    Ms_min: int = 200,
):
    """
    Computes and returns the representative packet sizes for a specific app using greedy selection.

    Args:
        app_key (str): App identifier (e.g. 'com.instagram.android')
        path (str): Directory containing capture sessions
        save_path (str): Optional path to save results (e.g., f"data/{app_key}_filtered_sizes.pkl")
        load_existing (bool): If True, loads result from file if available
        R_min (float): Minimum total arrival rate for retained packets
        Ms_min (int): Minimum number of different directional sizes to retain
    """
    if save_path is None:
        save_path = f"data/filtered_sizes_{app_key}.pkl"

    if load_existing and os.path.exists(save_path):
        with open(save_path, "rb") as f:
            print(f"Loading filtered packet sizes for {app_key} from {save_path}")
            return pickle.load(f)

    all_packets = get_frame_sizes_per_app(path=path)
    durations = get_capture_durations_per_app(path=path)

    if app_key not in all_packets or app_key not in durations:
        raise ValueError(f"App '{app_key}' not found in capture data.")

    # Build Sp = all observed packet sizes across all apps
    Sp = set()
    for packet_list in all_packets.values():
        Sp.update(packet_list)

    # r_pos: arrival rates for this app only
    packet_sizes = all_packets[app_key]
    duration = durations[app_key]["duration"]
    r_pos = defaultdict(float)
    for size in packet_sizes:
        r_pos[size] += 1
    for size in r_pos:
        r_pos[size] /= duration

    # r_neg: average arrival rate of each packet size over all other apps
    r_neg_sum = defaultdict(float)

    for other_app, other_sizes in all_packets.items():
        if other_app == app_key:
            continue
        other_duration = durations[other_app]["duration"]
        counts = defaultdict(int)
        for s in other_sizes:
            counts[s] += 1
        for s, c in counts.items():
            r_neg_sum[s] += c / other_duration

    num_other_apps = len(all_packets) - 1
    r_neg = {s: r_neg_sum[s] / num_other_apps for s in r_neg_sum}

    # Greedy selection
    filtered_sizes = _construct_SpA_greedy(
        Sp=list(Sp),
        r_pos=r_pos,
        r_neg=r_neg,
        R_min=R_min,
        Ms_min=Ms_min,
    )

    # Save if requested
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "wb") as f:
        pickle.dump(filtered_sizes, f)
        print(f"Saved filtered packet sizes for {app_key} to {save_path}")

    return filtered_sizes


"""
test
app A: [1,1,1,1,1]
app B: [1,1,2,2]
app C: [3, 2]
app D: [4,4,4,4,4]
r
"""
