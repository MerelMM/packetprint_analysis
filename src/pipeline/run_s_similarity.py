from filter_packages import (
    get_filtered_data,
    get_filtered_data_without_timestamps,
)
from s_similarity import (
    train_sxgboost_all_apps,
    compute_s_similarity,
    load_sxgboost_models,
    extract_app_name,
    hac_clustering,
    get_segments,
)
import numpy as np
import os
import json
import matplotlib.pyplot as plt
from collections import defaultdict


def test_training_sxgboost():
    filtered_data = get_filtered_data_without_timestamps(load_existing=True)

    three_apps = {}
    three_apps["App1"] = filtered_data["session_com.substack.app_20250721_135041"]
    three_apps["App2"] = filtered_data[
        "session_com.google.android.apps.maps_20250721_161501"
    ]
    three_apps["App3"] = filtered_data["session_com.bol.shop_20250721_162102"]
    train_sxgboost_all_apps(data=three_apps)


def train_sxgboost_on_all_captures():
    """
    Trains S-XGBoost models for all available captured apps using filtered training data.
    Models are saved per app name in the configured save directory.
    """
    # Load preprocessed data without timestamps (training only)
    filtered_data = get_filtered_data_without_timestamps(load_existing=True)
    print(f"Training models for {len(filtered_data.keys())} apps...")
    train_sxgboost_all_apps(data=filtered_data)
    print("All models have been trained and saved.")


def test_taking_s_similarity():
    # test for trained on App1, but on App2 as negative example, App3 is ood
    filtered_data = get_filtered_data_without_timestamps(
        load_existing=True, train=False
    )
    apps_to_test = [
        "session_com.substack.app_20250721_135041",
        "session_com.google.android.apps.maps_20250721_161501",
        "session_com.decathlon.app_20250721_132501",
    ]
    model_App1 = load_sxgboost_models("App1")
    scores = {}
    for session in apps_to_test:
        scores[session] = compute_s_similarity(model_App1, filtered_data[session])
        print(session, np.mean(scores[session]))


def run_all_s_similarity(save_path="data/s_sxgboost_pairwise_results.json"):
    """
    Computes and saves S-similarity scores for every trained model on every available test session.

    Args:
        save_path (str): Path where the results will be stored (JSON format)

    Returns:
        dict: {model_app: {test_app: mean_score}}
    """
    filtered_data = get_filtered_data_without_timestamps(
        load_existing=True, train=False
    )
    pairwise_scores = {}

    # Outer loop: one trained model per app/session
    for session_trained in filtered_data.keys():
        app_name_tested = extract_app_name(session_trained)
        models = load_sxgboost_models(app_name_tested)  # returns list of xgb models

        pairwise_scores[app_name_tested] = {}

        # Test the trained model on every session
        for session_testing, packets in filtered_data.items():
            app_name_testing = extract_app_name(session_testing)

            scores = compute_s_similarity(models, packets)
            avg_score = float(np.mean(scores))  # convert to float for JSON serializing

            print(f"Model {app_name_tested} on {app_name_testing}: {avg_score:.4f}")

            # Save score for this model vs this test session
            pairwise_scores[app_name_tested][app_name_testing] = avg_score

    # Save results to file
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w") as f:
        json.dump(pairwise_scores, f, indent=2)

    print(f"\nAll pairwise S-similarity scores saved to {save_path}")
    return pairwise_scores


def plot_s_similarity_pos_neg(
    results_path="data/s_sxgboost_pairwise_results.json",
    save_plot_path="plots/s_similarity_pos_vs_neg.png",
):
    """
    Plots a comparison of S-similarity scores for positive vs negative samples.

    Args:
        results_path (str): Path to JSON file with pairwise S-similarity scores.
        save_plot_path (str): File path to save the generated plot.
    """
    # Load results
    with open(results_path, "r") as f:
        pairwise_scores = json.load(f)

    positive_scores = []
    negative_scores = []

    # Separate positive vs negative scores
    for model_app, test_scores in pairwise_scores.items():
        for test_app, score in test_scores.items():
            if test_app == model_app:
                positive_scores.append(score)
            else:
                negative_scores.append(score)

    # Plotting
    plt.figure(figsize=(8, 5))
    plt.hist(
        positive_scores,
        bins=20,
        alpha=0.6,
        label="Positive (same app)",
        color="green",
        density=True,
    )
    plt.hist(
        negative_scores,
        bins=20,
        alpha=0.6,
        label="Negative (different app)",
        color="red",
        density=True,
    )

    plt.title("S-Similarity: Positive vs Negative Samples")
    plt.xlabel("S-Similarity Score")
    plt.ylabel("Density")
    plt.legend()
    plt.grid(True)

    os.makedirs(os.path.dirname(save_plot_path), exist_ok=True)
    plt.savefig(save_plot_path, dpi=300)
    plt.close()

    print(f"Plot saved to {save_plot_path}")


def serialized_input_s_similarity(
    training_phase=True, save_path="data/s_similarity_serialized.json", N=6, load=False
):
    """
    Serialize all traces, calculate S-similarity for each trained model,
    and keep track of true labels, timestamps, and packet sizes for hierarchical clustering.

    Args:
        save_path (str): Path to save the serialized results (JSON)
        N (int): N-gram window size used during training

    Returns:
        dict: {
            "timestamps": [...],
            "labels": [...],
            "packet_sizes": [...],
            "scores": {app_model: [score_per_packet...]},
        }
    """
    if load and os.path.exists(save_path):
        with open(save_path, "r") as f:
            return json.load(f)

    data_phase = "train" if training_phase else "test"
    # 1. Load full filtered dataset WITH timestamps
    filtered_data = get_filtered_data(load_existing=True)  # {session: [(t, size), ...]}
    session_names = list(filtered_data.keys())

    # Prepare storage
    serialized_scores = {extract_app_name(s): [] for s in session_names}

    # 2. Load all trained models
    all_apps = set(extract_app_name(s) for s in session_names)
    models_per_app = {app: load_sxgboost_models(app) for app in all_apps}

    # 3. Concatenate all sessions in temporal order
    combined_packets = []  # packet sizes
    combined_timestamps = []  # timestamps
    combined_labels = []  # app labels

    time_offset = 0.0

    for session in session_names:
        app_label = extract_app_name(session)
        trace = filtered_data[session]

        times = [t for t, _ in trace[data_phase]]
        sizes = [s for _, s in trace[data_phase]]

        if not times:
            continue

        # Shift timestamps by current offset
        shifted_times = [t + time_offset for t in times]

        combined_packets.extend(sizes)
        combined_timestamps.extend(shifted_times)
        combined_labels.extend([app_label] * len(trace[data_phase]))

        # Update offset for next session
        time_offset = shifted_times[-1] + 0.1  # add small gap

    # 4. Compute S-similarity for each model
    for app, model_list in models_per_app.items():
        try:
            scores = compute_s_similarity(model_list, combined_packets, N=N)
            padded_scores = np.concatenate([np.zeros(N), scores, np.zeros(N)])
            serialized_scores[app] = padded_scores.tolist()
        except Exception as e:
            print(f"Skipping {app} due to error in S-similarity: {e}")
            serialized_scores[app] = []

    # 5. Build final serialized dataset, now with packet sizes
    serialized_data = {
        "timestamps": combined_timestamps,
        "labels": combined_labels,
        "packet_sizes": combined_packets,
        "scores": serialized_scores,
    }

    # 6. Save to file
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w") as f:
        json.dump(serialized_data, f)

    print(f"Serialized S-similarity input saved to {save_path}")
    return serialized_data


def identify_anchor_packets(
    anchor_threshold=0.95,
    serialized_path="data/s_similarity_serialized.json",
    save_path="data/anchor_packet_timings_per_app.json",
):
    """
    Identify anchor packets above a threshold for each app and
    store them in a structured dict for per-app HAC clustering.

    Output format:
    {
      "com.app1": [
          {"timestamp": t1, "packet_size": s1, "score": score1},
          ...
      ],
      "com.app2": [
          {"timestamp": t2, "packet_size": s2, "score": score2},
          ...
      ]
    }
    """
    with open(serialized_path, "r") as f:
        serialized_traces = json.load(f)

    timestamps = serialized_traces["timestamps"]
    # packet_sizes = serialized_traces["packet_sizes"]
    scores = serialized_traces["scores"]
    # labels = serialized_traces["labels"]

    anchor_packets = defaultdict(list)

    # Collect anchor packets for each app
    for app, app_scores in scores.items():
        debug = []
        debug_ix = []
        for ix, s_sim in enumerate(app_scores):

            if s_sim >= anchor_threshold:
                anchor_packets[app].append(timestamps[ix])

    # results in timestamps of the list of anchor packets (time should be unique identifier)

    # Save per-app structure
    with open(save_path, "w") as f:
        json.dump(anchor_packets, f, indent=2)

    print(
        f"Identified anchor packets for {len(anchor_packets)} apps. Saved to {save_path}"
    )
    return anchor_packets


def run_hac_clustering(
    anchor_data="data/anchor_packet_timings_per_app.json", epsilon=30
):
    with open(anchor_data, "r") as f:
        anchor_data = json.load(f)
    return hac_clustering(anchor_data, epsilon)


def get_segments_input_c_similarity(
    hac_clusters="data/hac_clusters_per_app.json",
    filter_path: str = "data/filtered_packet_sizes.json",
    data_path: str = "capture_data",
):
    serialized_data = serialized_input_s_similarity(load=True)
    with open(hac_clusters, "r") as f:
        hac_clusters = json.load(f)
    return get_segments(hac_clusters, serialized_data)
