# This script was developed with assistance from ChatGPT (OpenAI) and Github Copilot
# Final implementation and adaptation by Merel Haenaets.
import os
import json
from typing import List, Dict, Any


def get_segments(
    app: str,
    clusters: List[Dict[str, float]] = None,  # for if precomputed
    serialized_data: tuple = None,
    save_dir: str = "data/segments_per_app",
    load_precomputed: bool = False,
) -> List[Dict[str, list]]:
    """
    Extract packet times and sizes that fall within the clustered target segments
    and save them per app. Can also load precomputed segments from disk.

    Args:
        app (str): App name
        clusters: List of segments [{"start_time": float, "end_time": float, ...}, ...]
        serialized_data: Tuple of (packet_sizes, timestamps, labels)
        save_dir: Directory where segments will be saved
        load_precomputed: If True, will load the saved segments if they exist

    Returns:
        List of segment dicts:
            [{ "timestamps": [...], "labels": [...], "packet_sizes": [...] }, ...]
    """
    os.makedirs(save_dir, exist_ok=True)
    app_file = os.path.join(save_dir, f"_{app}.json")

    if load_precomputed and os.path.exists(app_file):
        print(f"Loaded precomputed segments for {app} from {app_file}")
        with open(app_file, "r") as f:
            return json.load(f)

    sizes, timings, labels = serialized_data
    segments = []

    for cluster_info in clusters:
        start_time = cluster_info["start_time"]
        end_time = cluster_info["end_time"]

        current_segment_time = []
        current_segment_label = []
        current_segment_packet = []

        for ix, timestamp in enumerate(timings):
            if start_time <= timestamp <= end_time:
                current_segment_time.append(timestamp)
                current_segment_label.append(labels[ix])
                current_segment_packet.append(sizes[ix])
            elif timestamp > end_time:
                break  # timestamps are assumed sorted

        segments.append(
            {
                "timestamps": current_segment_time,
                "labels": current_segment_label,
                "packet_sizes": current_segment_packet,
            }
        )

    with open(app_file, "w") as f:
        json.dump(segments, f, indent=2)
    print(f"Saved {len(segments)} segments for {app} to {app_file}")

    return segments
