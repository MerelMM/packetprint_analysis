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

            fp = result["false_positive_proposed"]
            tp = result["true_positive_proposed"]
            fn = result["false_negative_proposed"]
            tn = result["true_negative_proposed"]

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
    plt.savefig(f"plots/{filename}.png", dpi=300)
    plt.savefig(f"plots/{filename}.pdf")
    plt.close()


# Plot MCC
draw_single_boxplot(
    data=all_mccs,
    label="MCC",
    filename="rec_mcc_boxplot",
    ylabel="MCC",
    PASTEL="#d1b9e3",
    under=-1,
)

# Plot Accuracy
draw_single_boxplot(
    data=all_accuracies,
    label="Accuracy",
    filename="rec_accuracy_boxplot",
    ylabel="Accuracy",
    PASTEL="#AEC6CF",
    under=0,
)

# import os
# import json
# import math
# import matplotlib.pyplot as plt


# import math


# def evaluate_recognition(result, filename):
#     fp = result["false_positive_proposed"]
#     tp = result["true_positive_proposed"]
#     fn = result["false_negative_proposed"]
#     tn = result["true_negative_proposed"]

#     total = tp + tn + fp + fn

#     # Compute accuracy
#     if total == 0:
#         accuracy = 0.0
#     else:
#         accuracy = (tp + tn) / total

#     # Compute MCC
#     if fp == 0 and fn == 0:
#         print(f"{filename}: Perfect prediction (all TPs)")
#         return 1.0, accuracy

#     numerator = tp * tn - fp * fn
#     denominator = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))

#     if denominator == 0:
#         mcc = 0.0
#     else:
#         mcc = numerator / denominator

#     return mcc, accuracy


# def load_last_json_line(filepath):
#     with open(filepath, "r") as f:
#         lines = f.readlines()
#         if not lines:
#             return None
#         return json.loads(lines[-1])


# def main():
#     result_dir = "evaluation_results_conf2"
#     mcc_values = []
#     accuracy_values = []
#     app_labels = []

#     for filename in os.listdir(result_dir):
#         if filename.endswith(".jsonl"):
#             filepath = os.path.join(result_dir, filename)
#             result = load_last_json_line(filepath)
#             if result is not None:
#                 mcc, acc = evaluate_recognition(result, filename)
#                 mcc_values.append(mcc)
#                 accuracy_values.append(acc)
#                 app_labels.append(filename.replace("result_", "").replace(".jsonl", ""))

#     pastel_color = "#AEC6CF"  # light pastel blue
#     os.makedirs("plots", exist_ok=True)

#     # === MCC Plot ===
#     # plt.figure(figsize=(10, 6))
#     box1 = plt.boxplot(mcc_values, vert=True, patch_artist=True, labels=["All Apps"])
#     for patch in box1["boxes"]:
#         patch.set_facecolor("#d1b9e3")
#     plt.scatter([1] * len(mcc_values), mcc_values, color="black", alpha=0.6)
#     plt.ylabel("MCC", fontsize=22)
#     # plt.title("MCC Across Apps", fontsize=22)
#     plt.xticks(fontsize=22)
#     plt.yticks(fontsize=22)
#     plt.grid(True, axis="y")
#     plt.ylim(-1, 1)
#     plt.xlim(0.75, 1.25)
#     plt.tight_layout()
#     plt.savefig("plots/evaluation_mcc.png", dpi=300)
#     plt.close()

#     # === Accuracy Plot ===
#     # plt.figure(figsize=(10, 6))
#     box2 = plt.boxplot(
#         accuracy_values, vert=True, patch_artist=True, labels=["All Apps"]
#     )
#     for patch in box2["boxes"]:
#         patch.set_facecolor(pastel_color)
#     plt.scatter([1] * len(accuracy_values), accuracy_values, color="black", alpha=0.6)
#     plt.ylabel("Accuracy", fontsize=22)
#     # plt.title("Accuracy Across Apps", fontsize=22)
#     plt.xticks(fontsize=22)
#     plt.yticks(fontsize=22)
#     plt.grid(True, axis="y")
#     plt.xlim(0.75, 1.25)
#     plt.tight_layout()
#     plt.savefig("plots/evaluation_accuracy.png", dpi=300)
#     plt.close()

#     # Print per-app results
#     for label, mcc, acc in zip(app_labels, mcc_values, accuracy_values):
#         print(f"{label:40s}  MCC: {mcc:.4f}  Accuracy: {acc:.4f}")


# if __name__ == "__main__":
#     main()
