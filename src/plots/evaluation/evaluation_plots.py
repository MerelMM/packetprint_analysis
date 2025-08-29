# This script was developed with assistance from ChatGPT (OpenAI) and Github Copilot
# Final implementation and adaptation by Merel Haenaets.
import os
import glob
import json
import math
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines

RESULTS_DIR = "evaluation_results_conf2"


all_mccs = []
all_accuracies = []

# Load all results
for path in glob.glob(os.path.join(RESULTS_DIR, "result_*.jsonl")):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                result = json.loads(line)
            except json.JSONDecodeError:
                continue

            fp = result["false_positive_seg"]
            tp = result["true_positive_seg"]
            fn = result["false_negative_seg"]
            tn = result["true_negative_seg"]

            total = tp + tn + fp + fn
            accuracy = (tp + tn) / total if total > 0 else 0.0

            if fp == 0 and fn == 0:
                mcc = 1.0
            else:
                numerator = tp * tn - fp * fn
                denominator = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
                mcc = numerator / denominator if denominator != 0 else 0.0

            all_mccs.append(mcc)
            all_accuracies.append(accuracy)

# Ensure output folder exists
os.makedirs("thesis_figures_conf3", exist_ok=True)


# Helper function to draw a single boxplot
def draw_single_boxplot(data, label, filename, ylabel, PASTEL, under):
    plt.figure(figsize=(7, 7))
    plt.rcParams.update({"font.size": 22})

    bp = plt.boxplot(
        [data], patch_artist=True, showmeans=True, meanline=True, labels=[label]
    )

    for patch in bp["boxes"]:
        patch.set_facecolor(PASTEL)
        patch.set_edgecolor("black")
    for whisker in bp["whiskers"]:
        whisker.set_color("black")
    for cap in bp["caps"]:
        cap.set_color("black")
    for median in bp["medians"]:
        median.set_color("black")
    for mean in bp["means"]:
        mean.set_color("black")
        mean.set_linestyle("--")

    # Custom legend
    median_line = mlines.Line2D([], [], color="black", linestyle="-", label="Median")
    mean_line = mlines.Line2D([], [], color="black", linestyle="--", label="Mean")
    box_patch = mpatches.Patch(
        facecolor=PASTEL, edgecolor="black", label="IQR (25–75%)"
    )

    plt.legend(
        handles=[box_patch, median_line, mean_line], loc="lower right", fontsize=20
    )

    plt.ylabel(ylabel, fontsize=22)
    plt.ylim(under, 1.1)
    plt.xlim(0.75, 1.25)
    plt.tight_layout()
    plt.savefig(f"thesis_figures_conf3/{filename}.png", dpi=300)
    plt.savefig(f"thesis_figures_conf3/{filename}.pdf")
    plt.close()


# Plot MCC
draw_single_boxplot(
    data=all_mccs,
    label="MCC",
    filename="total_mcc_boxplot",
    ylabel="MCC",
    PASTEL="#d1b9e3",
    under=-1,
)

# Plot Accuracy
draw_single_boxplot(
    data=all_accuracies,
    label="Accuracy",
    filename="total_accuracy_boxplot",
    ylabel="Accuracy",
    PASTEL="#AEC6CF",
    under=0,
)
