# This script was developed with assistance from ChatGPT (OpenAI) and Github Copilot
# Final implementation and adaptation by Merel Haenaets.
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import random
import matplotlib.pyplot as plt
from collections import defaultdict
from preprocessing.packet_size_filtering import filter_packet_size
from helper.helper import get_app_key


def plot_packet_size_histogram(data_path="capture_data", NUM_APPS=3):
    filtered_sizes_per_apps = filter_packet_size(path=data_path)

    app_sessions = defaultdict(list)
    for session_dir in os.listdir(data_path):
        session_path = os.path.join(data_path, session_dir)
        if not os.path.isdir(session_path):
            continue
        app_name = get_app_key(session_dir)
        app_sessions[app_name].append(session_path)

    sampled_apps = random.sample(
        list(app_sessions.keys()), min(NUM_APPS, len(app_sessions))
    )

    os.makedirs("plots/preprocessing", exist_ok=True)

    for app in sampled_apps:
        all_sizes = []
        all_labels = []

        for session_path in app_sessions[app]:
            size_file = os.path.join(session_path, "packet_sizes.txt")
            if not os.path.exists(size_file):
                continue

            with open(size_file) as f:
                sizes = [int(line.strip()) for line in f if line.strip()]
                labels = [
                    "keep" if s in filtered_sizes_per_apps[app] else "drop"
                    for s in sizes
                ]
                all_sizes.extend(sizes)
                all_labels.extend(labels)

        if not all_sizes:
            continue

        kept = [s for s, label in zip(all_sizes, all_labels) if label == "keep"]
        dropped = [s for s, label in zip(all_sizes, all_labels) if label == "drop"]

        plt.figure(figsize=(12, 5))
        plt.hist(
            [dropped, kept],
            bins=100,  # Higher number of bins for more detail
            stacked=True,
            label=["Dropped", "Kept"],
            color=["gray", "blue"],
            alpha=0.8,
        )

        # Optionally add rug lines (tick marks on x-axis)
        for s in kept:
            plt.axvline(x=s, ymin=0, ymax=0.02, color="blue", linewidth=0.5, alpha=0.3)
        for s in dropped:
            plt.axvline(x=s, ymin=0, ymax=0.02, color="gray", linewidth=0.5, alpha=0.3)

        plt.title(f"Packet Size Distribution for {app}")
        plt.xlabel("Packet Size (bytes)")
        plt.ylabel("Frequency")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(
            f"plots/preprocessing/packet_size_histogram_{app}.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.show()
