# This script was developed with assistance from ChatGPT (OpenAI) and Github Copilot
# Final implementation and adaptation by Merel Haenaets.
import sys
import os
import matplotlib.pyplot as plt
from collections import defaultdict
from preprocessing.packet_size_filtering import filter_packet_size
from helper.helper import get_app_key


def plot_filtering_process(data_path="capture_data_train"):
    # Load per-app packet sizes to keep
    filtered_sizes_per_apps = filter_packet_size(path=data_path)

    # For each app, accumulate total packets and total kept
    total_sizes_per_app = defaultdict(int)
    total_kept_per_app = defaultdict(int)

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
            print(f"Skipping {session_dir}: size/timing mismatch")
            continue

        app_name = get_app_key(session_dir)
        keep_set = filtered_sizes_per_apps.get(app_name, set())

        total_sizes_per_app[app_name] += len(sizes)
        total_kept_per_app[app_name] += sum(1 for s in sizes if s in keep_set)

    # Calculate percentage kept per app
    app_names = []
    percent_kept = []
    for app_name in sorted(total_sizes_per_app):
        total = total_sizes_per_app[app_name]
        kept = total_kept_per_app[app_name]
        percentage = 100 * kept / total if total > 0 else 0
        app_names.append(app_name)
        percent_kept.append(percentage)

    app_name_map = {
        "ai.character.app": "Character AI",
        "bbc.mobile.news.ww": "BBC News",
        "com.adobe.lrmobile": "Adobe Lightroom",
        "com.alltrails.alltrails": "AllTrails",
        "com.booking": "Booking.com",
        "com.decathlon.app": "Decathlon",
        "com.deepl.mobiletranslator": "DeepL",
        "com.facebook.orca": "Facebook Messenger",
        "com.fun.lastwar.gp": "Last War",
        "com.getsomeheadspace.android": "Headspace",
        "com.google.android.apps.maps": "Google Maps",
        "com.google.android.youtube": "YouTube",
        "com.groundspeak.geocaching.intro": "Geocaching",
        "com.instagram.android": "Instagram",
        "com.instagram.basel": "Instagram Basel",
        "com.kiloo.subwaysurf": "Subway Surfers",
        "com.microsoft.office.outlook": "Outlook",
        "com.pineapplestudio.codedelaroutebelge": "Belgian Road Code",
        "com.pinterest": "Pinterest",
        "com.spotify.music": "Spotify",
        "com.substack.app": "Substack",
        "com.supportware.Buienradar": "Buienradar",
        "com.themobilecompany.delijn": "De Lijn",
        "com.touchtype.swiftkey": "SwiftKey",
        "com.trainingboard.moon": "Moon Training",
        "com.wondershare.filmorago": "FilmoraGo",
        "de.wetteronline.wetterapp": "WetterOnline",
        "wp.wattpad": "Wattpad",
    }
    # Filter to only apps you want to include and map to pretty names
    filtered_app_names = []
    filtered_percent_kept = []

    for app_name, pct in zip(app_names, percent_kept):
        if app_name in app_name_map:
            filtered_app_names.append(app_name_map[app_name])
            filtered_percent_kept.append(pct)

    app_names = filtered_app_names
    percent_kept = filtered_percent_kept

    # Plot
    plt.figure(figsize=(12, 6))
    plt.bar(app_names, percent_kept, color="#d1b9e3")
    plt.ylabel("Packets Kept (%)", fontsize=22)
    plt.xlabel("App", fontsize=22)
    plt.xticks(rotation=45, ha="right", fontsize=22)
    plt.yticks(fontsize=22)
    plt.ylim(0, 100)
    plt.grid(axis="y")
    plt.tight_layout()
    plt.savefig(
        "plots/preprocessing/percentage_filtered_per_app.png",
        dpi=300,
        bbox_inches="tight",
    )
