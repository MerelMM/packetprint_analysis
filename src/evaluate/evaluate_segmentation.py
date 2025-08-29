# This script was developed with assistance from ChatGPT (OpenAI) and Github Copilot
# Final implementation and adaptation by Merel Haenaets.
import matplotlib.pyplot as plt
import os
from preprocessing.concat_traces import concat_traces
from segmentation.run_segmentation import run_segmentation

app_list = [
    "bbc.mobile.news.ww",
    "com.alltrails.alltrails",
    "com.decathlon.app",
    "com.google.android.youtube",
    "com.instagram.basel",
    "com.kiloo.subwaysurf",
    "com.pinterest",
    "com.spotify.music",
    "com.trainingboard.moon",
]


def evaluate_segmentation(app_key):
    phi_min = 0.75
    epsilon = 30

    concatenated_test_data = concat_traces(
        app_key,
        data_path="capture_data_test",
        save_path="data/concatenated_test_trace.pkl",
        load_existing=True,
        load_existing_filter=True,
    )
    _, _, concatenated_labels = concatenated_test_data

    total_positive = sum(1 for l in concatenated_labels if l == app_key)
    total_negative = len(concatenated_labels) - total_positive

    segments_test = run_segmentation(
        app_key,
        concatenated_data=concatenated_test_data,
        load_precomputed=False,
        epsilon=epsilon,
        anchor_threshold=phi_min,
        load_pretrained=True,
    )

    remaining_positive = 0
    remaining_negative = 0
    for seg in segments_test:
        for l in seg["labels"]:
            if l == app_key:
                remaining_positive += 1
            else:
                remaining_negative += 1

    return total_positive, total_negative, remaining_positive, remaining_negative


def compute_stats_per_app(app_list):
    ratios = {
        "positive_ratio_before": [],
        "positive_ratio_after": [],
        "negative_remaining_percentage": [],
        "positive_retained_percentage": [],
    }

    app_labels = []

    for app in app_list:
        tp, tn, rp, rn = evaluate_segmentation(app)

        total = tp + tn
        remaining_total = rp + rn

        if total > 0:
            ratios["positive_ratio_before"].append(tp / (total + 1e-8))
        else:
            ratios["positive_ratio_before"].append(0.0)

        if remaining_total > 0:
            ratios["positive_ratio_after"].append(rp / (remaining_total + 1e-8))
        else:
            ratios["positive_ratio_after"].append(0.0)

        if tn > 0:
            ratios["negative_remaining_percentage"].append(rn / (tn + 1e-8))
        else:
            ratios["negative_remaining_percentage"].append(0.0)

        if tp > 0:
            ratios["positive_retained_percentage"].append(rp / (tp + 1e-8))
        else:
            ratios["positive_retained_percentage"].append(0.0)

        app_labels.append(app)

    return ratios, app_labels


def plot_overall_average(stats):
    # plt.figure(figsize=(10, 6))
    plt.figure(figsize=(8, 6))
    plt.rcParams.update({"font.size": 22})

    categories = [
        "Pos. Before",
        "Pos. After",
        "Neg. Left",
        "Pos. Kept",
    ]
    values = [
        stats["positive_ratio_before"],
        stats["positive_ratio_after"],
        stats["negative_remaining_percentage"],
        stats["positive_retained_percentage"],
    ]

    pastel_color = "#daeaf6"
    plt.bar(categories, values, color=pastel_color, edgecolor="black")

    plt.ylim(0, 1.1)
    plt.ylabel("Proportion", fontsize=20)
    # plt.title("Overall Segmentation Filtering Effect (All Apps)", fontsize=20)
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()

    os.makedirs("plots", exist_ok=True)
    plt.savefig("plots/segmentation_filtering_overall.png", dpi=300)
    plt.savefig("plots/segmentation_filtering_overall.pdf")
    plt.close()


def plot_per_app_bars(ratios, app_labels):
    plt.figure(figsize=(14, 7))
    plt.rcParams.update({"font.size": 22})

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

    metric_names = [
        "positive_ratio_before",
        "positive_ratio_after",
        "negative_remaining_percentage",
        "positive_retained_percentage",
    ]
    metric_labels = [
        "Pos. Before",
        "Pos. After",
        "Neg. Left",
        "Pos. Kept",
    ]
    pastel_colors = ["#fff2ae", "#f6c5c0", "#f0e6f6", "#d0e6a5"]

    x = range(len(app_labels))
    width = 0.2

    for i, metric in enumerate(metric_names):
        values = ratios[metric]
        plt.bar(
            [p + i * width for p in x],
            values,
            width=width,
            label=metric_labels[i],
            edgecolor="black",
            color=pastel_colors[i],  # <-- use a different color for each metric
        )

    # X ticks in the center
    mid_x = [p + 1.5 * width for p in x]
    pretty_labels = [app_name_map.get(app, app) for app in app_labels]
    plt.xticks(mid_x, pretty_labels, rotation=45, ha="right")
    # plt.xticks(mid_x, app_labels, rotation=45, ha="right")
    plt.ylim(0, 1.1)
    plt.ylabel("Proportion", fontsize=22)
    # plt.title("Per-App Segmentation Statistics", fontsize=18)
    plt.legend()
    plt.tight_layout()

    os.makedirs("plots", exist_ok=True)
    plt.savefig("plots/segmentation_filtering_per_app.png", dpi=300)
    plt.savefig("plots/segmentation_filtering_per_app.pdf")
    plt.close()


def run_plotting_segmentation():
    ratios, app_labels = compute_stats_per_app(app_list)

    # Compute mean values for overall plot
    stats = {
        key: sum(values) / len(values) if values else 0.0
        for key, values in ratios.items()
    }

    plot_overall_average(stats)
    plot_per_app_bars(ratios, app_labels)

    print("=== Overall Averages ===")
    for k, v in stats.items():
        print(f"{k:30s}: {v:.4f}")


if __name__ == "__main__":
    run_plotting_segmentation()
