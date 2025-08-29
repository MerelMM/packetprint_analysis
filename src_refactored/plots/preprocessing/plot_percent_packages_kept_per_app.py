# This script was developed with assistance from ChatGPT (OpenAI) and Github Copilot
# Final implementation and adaptation by Merel Haenaets.
import matplotlib.pyplot as plt
import os
import pickle
from collections import defaultdict
from helper.helper import get_app_key  # make sure this is accessible


def plot_percent_packets_kept_per_app(
    app_key: str,
    data_path: str = "capture_data_test",
    save_path: str = "data/filtered_sizes.pkl",
):
    """
    Plots how many percent of the packets are kept per app
    when filtering with the packet size filter of one specific app.
    """
    save_path: str = "data/filtered_sizes.pkl"
    app_save_path = f"{os.path.splitext(save_path)[0]}_{app_key}.pkl"

    if not os.path.exists(app_save_path):
        raise FileNotFoundError(
            f"No filtered data found at {app_save_path}. Run run_preprocessing first."
        )

    # Load filtered data
    with open(app_save_path, "rb") as f:
        filtered_packet_sizes = pickle.load(f)
    filtered_packet_sizes = set(filtered_packet_sizes)
    app_total_sizes = defaultdict(int)
    app_kept_sizes = defaultdict(int)

    for session_dir in os.listdir(data_path):
        session_path = os.path.join(data_path, session_dir)
        if not os.path.isdir(session_path):
            continue

        size_file = os.path.join(session_path, "packet_sizes.txt")
        if not os.path.exists(size_file):
            continue

        app = get_app_key(session_dir)

        with open(size_file) as sf:
            sizes = [int(s.strip()) for s in sf if s.strip()]
            app_total_sizes[app] += len(sizes)
            sizes_after_filtering = [1 for s in sizes if s in filtered_packet_sizes]
            app_kept_sizes[app] += len(sizes_after_filtering)

    # Calculate per-app percentages
    apps = sorted(app_total_sizes.keys())
    percent_kept = [
        (
            100 * app_kept_sizes[app] / app_total_sizes[app]
            if app_total_sizes[app] > 0
            else 0
        )
        for app in apps
    ]

    APP_DISPLAY_NAMES = {
        "ai.character.app": "Character AI",
        "app.nl.socialdeal": "Social Deal",
        "bbc.mobile.news.ww": "BBC News",
        "com.adobe.lrmobile": "Adobe Lightroom",
        "com.alltrails.alltrails": "AllTrails",
        "com.app.tgtg": "Too Good To Go",
        "com.applicaster.babytv.vod": "BabyTV",
        "com.block.juggle": "Juggle Game",
        "com.bol.shop": "Bol.com",
        "com.booking": "Booking.com",
        "com.decathlon.app": "Decathlon",
        "com.deepl.mobiletranslator": "DeepL",
        "com.facebook.orca": "Messenger",
        "com.finch.finch": "Finch",
        "com.fun.lastwar.gp": "Last War",
        "com.getsomeheadspace.android": "Headspace",
        "com.google.android.apps.maps": "Google Maps",
        "com.google.android.apps.youtube.music": "YouTube Music",
        "com.google.android.youtube": "YouTube",
        "com.groundspeak.geocaching.intro": "Geocaching",
        "com.hiya.star": "Hiya",
        "com.instagram.android": "Instagram",
        "com.instagram.basel": "Instagram Basel",
        "com.kiloo.subwaysurf": "Subway Surfers",
        "com.meetup": "Meetup",
        "com.microsoft.office.outlook": "Outlook",
        "com.naver.linewebtoon": "LINE Webtoon",
        "com.openai.chatgpt": "ChatGPT",
        "com.pineapplestudio.codedelaroutebelge": "Code de la Route (BE)",
        "com.pinterest": "Pinterest",
        "com.roblox.client": "Roblox",
        "com.spotify.music": "Spotify",
        "com.strava": "Strava",
        "com.substack.app": "Substack",
        "com.supportware.Buienradar": "Buienradar",
        "com.tellmewow.focus": "Tellmewow Focus",
        "com.themobilecompany.delijn": "De Lijn",
        "com.touchtype.swiftkey": "SwiftKey",
        "com.trainingboard.moon": "MoonBoard",
        "com.vitastudio.mahjong": "Mahjong",
        "com.wondershare.filmorago": "FilmoraGo",
        "com.zhiliaoapp.musically": "TikTok",
        "de.wetteronline.wetterapp": "WetterOnline",
        "de.zalando.mobile": "Zalando",
        "id.highfivestudio.stitchbluekoala": "Stitch Blue Koala",
        "vidma.video.editor.videomaker": "Vidma Video Editor",
        "wp.wattpad": "Wattpad",
    }

    labels = [APP_DISPLAY_NAMES.get(app, app) for app in apps]
    # Plot
    plt.figure(figsize=(12, 6))
    plt.bar(labels, percent_kept, color="#d1b9e3")

    plt.ylabel("Packets Kept (%)", fontsize=20)
    plt.xlabel("App", fontsize=24)
    # plt.title(f"Filtering Effectiveness per App (Filter from: {app_key})", fontsize=24)

    plt.xticks(rotation=45, ha="right", fontsize=20)
    plt.yticks(fontsize=20)

    plt.ylim(0, 100)
    plt.grid(axis="y")
    plt.tight_layout()

    plt.savefig(
        f"plots/preprocessing/percent_filtered_per_app_{app_key}.png",
        dpi=300,
        bbox_inches="tight",
    )
