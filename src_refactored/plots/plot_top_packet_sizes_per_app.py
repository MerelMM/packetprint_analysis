import os
from collections import defaultdict, Counter

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib import cm
import pandas as pd

from helper.helper import get_app_key

import os
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict, Counter
import matplotlib.colors as mcolors
from helper.helper import get_app_key


def plot_top_packet_sizes_table_youtube_colored_only(
    data_path="capture_data_train", top_k=10, save_path=None
):
    packet_sizes_per_app = defaultdict(list)

    for session_dir in os.listdir(data_path):
        session_path = os.path.join(data_path, session_dir)
        if not os.path.isdir(session_path):
            continue

        size_file = os.path.join(session_path, "packet_sizes.txt")
        if not os.path.exists(size_file):
            continue

        try:
            with open(size_file) as f:
                sizes = [int(line.strip()) for line in f if line.strip()]
        except Exception as e:
            print(f"Skipping {session_dir} due to error: {e}")
            continue

        app_name = get_app_key(session_dir)
        packet_sizes_per_app[app_name].extend(sizes)

    if not packet_sizes_per_app:
        print("No valid packet size data found.")
        return

    rows = []
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

    top_5_youtube = []
    for app, sizes in sorted(packet_sizes_per_app.items()):
        if app == "com.google.android.youtube":
            counter = Counter(sizes)
            top_5_youtube = [s for s, _ in counter.most_common(5)]
            break

    if not top_5_youtube:
        print("No data found for YouTube.")
        return

    for app, sizes in sorted(packet_sizes_per_app.items()):
        if app not in app_name_map:
            continue
        counter = Counter(sizes)
        top = counter.most_common(top_k)
        row = [app] + [f"{s} ({c})" for s, c in top]
        row += [""] * (top_k + 1 - len(row))  # pad
        rows.append(row)

    col_names = ["App"] + [f"Top {i+1}" for i in range(top_k)]
    df = pd.DataFrame(rows, columns=col_names)
    df["App"] = df["App"].replace(app_name_map)
    df = df[df["App"].isin(app_name_map.values())]

    pastel_colors = ["#fff2ae", "#f6c5c0", "#f0e6f6", "#d0e6a5", "#daeaf6"]
    top_color_map = {size: pastel_colors[i] for i, size in enumerate(top_5_youtube)}

    # Plot
    fig, ax = plt.subplots(figsize=(26, 0.6 * len(df)))
    ax.axis("off")

    table = ax.table(
        cellText=df.values, colLabels=df.columns, loc="center", cellLoc="center"
    )
    table.auto_set_font_size(False)
    table.set_fontsize(24)
    table.scale(1, 3)

    for i in range(len(df)):
        for j in range(1, top_k + 1):
            cell = table[i + 1, j]
            text = df.iloc[i, j]
            if text and "(" in text:
                size_str = text.split("(", 1)[0].strip()
                try:
                    size = int(size_str)
                except ValueError:
                    continue

                if size in top_color_map:
                    cell.set_facecolor(top_color_map[size])
                else:
                    cell.set_facecolor("white")

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
        print(f"Saved figure to {save_path}")


def plot_top_packet_sizes_table(
    data_path="capture_data_train", top_k=10, save_path=None
):
    packet_sizes_per_app = defaultdict(list)

    for session_dir in os.listdir(data_path):
        session_path = os.path.join(data_path, session_dir)
        if not os.path.isdir(session_path):
            continue

        size_file = os.path.join(session_path, "packet_sizes.txt")
        if not os.path.exists(size_file):
            continue

        try:
            with open(size_file) as f:
                sizes = [int(line.strip()) for line in f if line.strip()]
        except Exception as e:
            print(f"Skipping {session_dir} due to error: {e}")
            continue

        app_name = get_app_key(session_dir)
        packet_sizes_per_app[app_name].extend(sizes)

    if not packet_sizes_per_app:
        print("No valid packet size data found.")
        return

    # Count across all apps
    size_global_counts = Counter()
    rows = []
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

    for app, sizes in sorted(packet_sizes_per_app.items()):
        if app not in app_name_map:
            continue
        counter = Counter(sizes)
        top = counter.most_common(top_k)
        for s, _ in top:
            size_global_counts[s] += 1
        row = [app] + [f"{s} ({c})" for s, c in top]
        row += [""] * (top_k + 1 - len(row))  # pad
        rows.append(row)

    col_names = ["App"] + [f"Top {i+1}" for i in range(top_k)]
    df = pd.DataFrame(rows, columns=col_names)
    df["App"] = df["App"].replace(app_name_map)
    df = df[df["App"].isin(app_name_map.values())]

    # Determine coloring
    top_5_most_common = [size for size, _ in size_global_counts.most_common(5)]
    other_duplicates = {
        size
        for size, count in size_global_counts.items()
        if count > 1 and size not in top_5_most_common
    }

    pastel_colors = ["#fff2ae", "#f6c5c0", "#f0e6f6", "#d0e6a5", "#daeaf6"]
    top_color_map = {size: pastel_colors[i] for i, size in enumerate(top_5_most_common)}

    # Plot
    fig, ax = plt.subplots(figsize=(26, 0.6 * len(df)))
    ax.axis("off")

    table = ax.table(
        cellText=df.values, colLabels=df.columns, loc="center", cellLoc="center"
    )
    table.auto_set_font_size(False)
    table.set_fontsize(24)
    table.scale(1, 3)

    for i in range(len(df)):
        for j in range(1, top_k + 1):
            cell = table[i + 1, j]
            text = df.iloc[i, j]
            if text and "(" in text:
                size_str = text.split("(", 1)[0].strip()
                try:
                    size = int(size_str)
                except ValueError:
                    continue

                if size in top_color_map:
                    cell.set_facecolor(top_color_map[size])
                elif size in other_duplicates:
                    cell.set_facecolor("#f0f0f0")  # light gray
                else:
                    cell.set_facecolor("white")

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
        print(f"Saved figure to {save_path}")


# def get_top_packet_sizes_per_app(data_path="capture_data", top_k=10):
#     packet_sizes_per_app = defaultdict(list)

#     for session_dir in os.listdir(data_path):
#         session_path = os.path.join(data_path, session_dir)
#         if not os.path.isdir(session_path):
#             continue

#         size_file = os.path.join(session_path, "packet_sizes.txt")
#         if not os.path.exists(size_file):
#             continue

#         app_name = get_app_key(session_dir)

#         with open(size_file) as f:
#             sizes = [int(line.strip()) for line in f if line.strip()]
#             packet_sizes_per_app[app_name].extend(sizes)

#     # Count how many apps each size appears in
#     size_app_count = Counter()
#     for app, sizes in packet_sizes_per_app.items():
#         unique_sizes = set(sizes)
#         for s in unique_sizes:
#             size_app_count[s] += 1

#     # Build dataframe
#     rows = []
#     for app, sizes in sorted(packet_sizes_per_app.items()):
#         counter = Counter(sizes)
#         top = counter.most_common(top_k)
#         row = [app] + [f"{s} ({c})" for s, c in top]
#         row += [""] * (top_k + 1 - len(row))  # pad if needed
#         rows.append(row)

#     col_names = ["App"] + [f"Top {i+1}" for i in range(top_k)]
#     df = pd.DataFrame(rows, columns=col_names)

#     return df, size_app_count


# def plot_colored_packet_size_table(
#     df,
#     size_app_count,
#     save_path="plots/preprocessing/packet_size_table_shared_colored.png",
# ):
#     fig, ax = plt.subplots(figsize=(20, 0.6 * len(df)))

#     ax.axis("off")
#     table = ax.table(
#         cellText=df.values, colLabels=df.columns, loc="center", cellLoc="center"
#     )

#     # Format table style
#     table.auto_set_font_size(False)
#     table.set_fontsize(10)
#     table.scale(1, 1.5)

#     # Get shared sizes only (appear in >1 app)
#     shared_sizes = {size for size, count in size_app_count.items() if count > 1}
#     shared_sizes = sorted(shared_sizes)
#     cmap = cm.get_cmap("Pastel1", len(shared_sizes))
#     color_map = {
#         str(size): mcolors.to_hex(cmap(i)) for i, size in enumerate(shared_sizes)
#     }

#     # Color the cells if their size is shared
#     for i in range(len(df)):
#         for j in range(1, 11):  # Skip app name
#             cell = table[i + 1, j]
#             text = df.iloc[i, j]
#             if text and "(" in text:
#                 size_str = text.split("(", 1)[0].strip()
#                 if size_str in color_map:
#                     cell.set_facecolor(color_map[size_str])
#                 else:
#                     cell.set_facecolor("white")

#     plt.tight_layout()
#     os.makedirs(os.path.dirname(save_path), exist_ok=True)
#     plt.savefig(save_path, dpi=300, bbox_inches="tight")
#     print(f"Saved to {save_path}")
#     plt.show()


def plot():
    plot_top_packet_sizes_table_youtube_colored_only(
        top_k=5, save_path="plots/top_packet_sizes_table.png"
    )
    # plot_top_packet_sizes_table(top_k=5, save_path="plots/top_packet_sizes_table.png")
    # df, size_app_count = get_top_packet_sizes_per_app(
    #     data_path="capture_data", top_k=10
    # )
    # plot_colored_packet_size_table(df, size_app_count)
