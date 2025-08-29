# This script was developed with assistance from ChatGPT (OpenAI) and Github Copilot
# Final implementation and adaptation by Merel Haenaets.
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import json
import os


# Load results
def load_sweep_results(path):
    with open(path) as f:
        records = [json.loads(line) for line in f]
    return pd.DataFrame(records)


# Compute F1 and print best config
def print_best_config(df):
    df["f1"] = (
        2 * df["precision"] * df["recall"] / (df["precision"] + df["recall"] + 1e-8)
    )
    best = df.sort_values(by="f1", ascending=False).iloc[0]
    print("Best configuration:")
    print(
        best[["phi_min", "epsilon", "voc_tree_depth", "alpha", "nf", "psi_min", "f1"]]
    )
    return best


# Plot heatmap for two parameters
def plot_f1_vs_two_params(df, param1, param2, out_dir="hyperparam_results"):
    # Compute F1 score
    df["f1"] = (
        2 * df["precision"] * df["recall"] / (df["precision"] + df["recall"] + 1e-8)
    )

    # Pivot table for heatmap
    pivot = df.pivot_table(values="f1", index=param1, columns=param2, aggfunc="mean")

    # Plot heatmap
    plt.figure(figsize=(10, 8))
    ax = sns.heatmap(
        pivot,
        annot=True,
        fmt=".2f",
        cmap="viridis",
        cbar_kws={"label": "F1 Score"},
        annot_kws={"fontsize": 18},
    )

    # Set axis labels with font size 22
    ax.set_xlabel(param2, fontsize=22)
    ax.set_ylabel(param1, fontsize=22)

    # Set tick label font sizes
    ax.tick_params(axis="both", labelsize=18)

    # Optional: remove title, use tight layout
    plt.tight_layout()

    # Save figure with high DPI
    os.makedirs(out_dir, exist_ok=True)
    plot_path = os.path.join(out_dir, f"f1_heatmap_{param1}_vs_{param2}.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()


# Generate all relevant heatmaps
def generate_all_heatmaps(df, out_dir="hyperparam_results"):
    os.makedirs(out_dir, exist_ok=True)
    param_pairs = [
        ("phi_min", "epsilon"),
        ("alpha", "nf"),
        ("psi_min", "voc_tree_depth"),
        ("phi_min", "psi_min"),
        ("nf", "voc_tree_depth"),
    ]
    for p1, p2 in param_pairs:
        plot_f1_vs_two_params(df, p1, p2, out_dir=out_dir)


# Run full analysis
def analyze_results(result_file):
    df = load_sweep_results(result_file)
    best = print_best_config(df)
    generate_all_heatmaps(df)
    df.to_csv("hyperparam_results/hyperparameter_sweep_summary.csv", index=False)
    return df, best


# Run the analysis on the user's data
result_file = "hyperparam_results/result_com.substack.app.jsonl"  # hyperparam_results/result_com.google.android.youtube.jsonl"
# result_file = "hyperparam_results/result_bbc.mobile.news.ww.jsonl"
analyze_results(result_file)
