import numpy as np
import xgboost as xgb
from sklearn.utils import resample
from typing import Dict, List, Tuple, Any
import os
import json
from scipy.cluster.hierarchy import linkage, fcluster
import pandas as pd
from collections import defaultdict


def create_N_grams_per_app(packet_data, N=6):
    """
    Constructs N-gram features for each packet in a sequence.
    """
    features = np.zeros([len(packet_data) - 2 * N, 2 * N + 1])
    for i in range(N, len(packet_data) - N):
        features[i - N] = packet_data[i - N : i + N + 1]
    return features


def extract_app_name(session_name: str) -> str:
    """
    Extract the app name from a folder name formatted like:
    session_appname_timestamp
    If it cannot parse, returns the full folder name.
    """
    parts = session_name.split("_")
    if len(parts) >= 3:
        return parts[1]  # 'appname' is typically the second field
    return session_name


def train_sxgboost_all_apps(
    data: Dict[str, Dict[str, List[int]]], N=6, save_dir="saved_models"
):
    """
    Trains one XGBoost model per app, distinguishing it from all other apps.

    Args:
        data: Dict mapping app -> {"train": [sizes...], "test": [sizes...]}
        N: Optional hyperparameter for train_s_xgboost

    Returns:
        models: Dict mapping app -> trained model
    """
    models = {}

    for app, packets in data.items():
        positive_data = packets

        # Combine data from all other apps
        negative_data = []
        for other_app, other_packets in data.items():
            if other_app != app:
                negative_data.extend(other_packets)

        # Train binary model for this app

        # Train binary model for this app
        model_list = train_s_xgboost(positive_data, negative_data, N=N)

        # Save each sub-model
        app_name = extract_app_name(app)
        app_dir = os.path.join(save_dir, app_name)
        os.makedirs(app_dir, exist_ok=True)
        for idx, model in enumerate(model_list):
            model_file = os.path.join(app_dir, f"model_{idx}.json")
            model.save_model(model_file)
        models[app_name] = model_list
        print(f"Saved {len(model_list)} models for app '{app_name}' in {app_dir}")
    return models


def train_s_xgboost(X_pos, X_neg, N):
    """
    Train an XGBoost classifier for S-similarity with downsampled negatives.

    Args:
        X_pos: ndarray of positive samples (packets from target app)
        X_neg: ndarray of negative samples (packets from other apps)
        N: window size parameter

    Returns:
        List of trained XGBoost models (F_0 ... F_N)
    """
    models = []
    center_idx = N

    # Downsample negative samples to match positive count
    if len(X_neg) > len(X_pos):
        X_neg = resample(X_neg, replace=False, n_samples=len(X_pos), random_state=42)
    else:
        print(
            "Warning: Positive samples are more than negative samples – downsampling positives."
        )
        X_pos = resample(X_pos, replace=False, n_samples=len(X_neg), random_state=42)

    X_pos = create_N_grams_per_app(X_pos)
    X_neg = create_N_grams_per_app(X_neg)
    # Combine positive and negative samples
    X_all = np.vstack((X_pos, X_neg))
    y_all = np.concatenate((np.ones(len(X_pos)), np.zeros(len(X_neg))))

    # Shuffle data to avoid ordering bias
    perm = np.random.permutation(len(y_all))
    X_all = X_all[perm]
    y_all = y_all[perm]

    # Define training data (all pre-split externally; no additional 80/20 needed)
    X_train, y_train = X_all, y_all

    # Train F_0 using only x_i (center packet)
    X_k = X_train[:, center_idx].reshape(-1, 1)
    dtrain = xgb.DMatrix(X_k, label=y_train)
    params = {"objective": "binary:logistic", "verbosity": 0}
    model = xgb.train(params, dtrain)
    models.append(model)

    # Train F_k for k = 1 to N
    for k in range(1, N):
        left = max(center_idx - k, 0)
        right = min(center_idx + k + 1, X_train.shape[1])
        X_k = X_train[:, left:right]

        dtrain = xgb.DMatrix(X_k, label=y_train)

        # Set base margin from previous cumulative prediction
        cumulative_margin = sum(
            m.predict(
                xgb.DMatrix(
                    X_train[
                        :,
                        max(center_idx - j, 0) : min(
                            center_idx + j + 1, X_train.shape[1]
                        ),
                    ]
                ),
                output_margin=True,
            )
            for j, m in enumerate(models)
        )
        dtrain.set_base_margin(cumulative_margin)

        model = xgb.train(params, dtrain)
        models.append(model)

    return models


def load_sxgboost_models(
    app_name: str, save_dir: str = "saved_models"
) -> List[xgb.Booster]:
    """
    Load all trained XGBoost models for a given app.

    Args:
        app_name: Name of the app (directory under save_dir)
        save_dir: Base directory where models are stored

    Returns:
        List of xgb.Booster objects for the given app
    """
    app_dir = os.path.join(save_dir, app_name)
    if not os.path.isdir(app_dir):
        raise FileNotFoundError(
            f"No model directory found for app '{app_name}' in '{save_dir}'"
        )

    # Find all model files, sort by index
    model_files = [
        f for f in os.listdir(app_dir) if f.startswith("model_") and f.endswith(".json")
    ]
    if not model_files:
        raise FileNotFoundError(
            f"No model files found in '{app_dir}' for app '{app_name}'"
        )

    # Sort numerically by model index
    model_files.sort(key=lambda f: int(f.split("_")[1].split(".")[0]))

    # Load models
    models = []
    for mf in model_files:
        model_path = os.path.join(app_dir, mf)
        booster = xgb.Booster()
        booster.load_model(model_path)
        models.append(booster)

    print(f"Loaded {len(models)} models for app '{app_name}' from '{app_dir}'")
    return models


def compute_s_similarity(
    s_xgboost_models: Dict[str, List[xgb.Booster]],
    packet_sequence: List[int],
    N: int = 6,
) -> Dict[str, np.ndarray]:
    """
    Compute S-similarity scores for each app on a given packet sequence.

    Args:
        s_xgboost_models: [xgb_model_F0, xgb_model_F1, ...]
        packet_sequence: list of packet sizes for this session
        N: context window size used during training

    Returns:
        dict: {app_name: np.ndarray of shape (num_samples,)}
              Each array contains S-similarity scores per center packet.
    """
    # Build N-gram features for the test sequence
    if len(packet_sequence) < 2 * N + 1:
        raise ValueError("Packet sequence too short for N-gram creation.")

    X_test = create_N_grams_per_app(packet_sequence, N=N)
    num_samples = X_test.shape[0]
    #
    s_scores = {}

    # # Compute S-similarity for each app
    # for app, model_list in models_per_app.items():
    cumulative_score = np.zeros(num_samples)

    for k, model in enumerate(s_xgboost_models):
        left = max(N - k, 0)
        right = min(N + k + 1, X_test.shape[1])
        X_k = X_test[:, left:right]

        dtest = xgb.DMatrix(X_k)
        pred_margin = model.predict(dtest, output_margin=True)
        cumulative_score += pred_margin

    # Convert logits to probability-like scores
    s_scores = 1.0 / (1.0 + np.exp(-cumulative_score))

    return s_scores


def hac_clustering(
    anchor_dict,
    epsilon=300,
    save_path="data/hac_clusters_per_app.json",
):
    """
    Efficient HAC for 1D time data:
    Groups anchor packets into clusters separated by gaps > epsilon.
    Complexity: O(n) instead of O(n²).
    """

    results = {}

    for app, timestamps in anchor_dict.items():
        if not timestamps:
            results[app] = []
            continue

        ts_sorted = sorted(timestamps)
        clusters = []
        current_cluster = [ts_sorted[0]]

        for t in ts_sorted[1:]:
            if t - current_cluster[-1] <= epsilon:
                current_cluster.append(t)
            else:
                clusters.append(current_cluster)
                current_cluster = [t]
        clusters.append(current_cluster)  # last cluster

        # Convert clusters to target segments
        results[app] = [
            {
                "cluster_id": i + 1,
                "start_time": float(min(c)),
                "end_time": float(max(c)),
                "anchor_count": len(c),
            }
            for i, c in enumerate(clusters)
        ]

    # Save results
    with open(save_path, "w") as f:
        json.dump(results, f, indent=2)

    print(
        f"Efficient HAC clustering complete for {len(results)} apps. Saved to {save_path}"
    )
    return results


def get_segments(
    segment_clusters: Dict[str, list],
    data: Dict[str, Any],
    save_dir: str = "data/segments_per_app",
    train: bool = True,
) -> Dict[str, list]:
    """
    Extract packet times and sizes that fall within the clustered target segments
    and save them per app.

    Args:
        segment_clusters: {app_name: [ {"start_time": float, "end_time": float}, ... ]}
        data: dict with keys ['timestamps', 'labels', 'packet_sizes', 'scores']
        save_dir: directory where per-app JSON files will be saved
        train: unused for now, placeholder if you later split data

    Returns:
        segments: {app_name: [ { "timestamps": [...], "labels": [...], "packet_sizes": [...] }, ... ]}
    """
    segments = defaultdict(list)
    os.makedirs(save_dir, exist_ok=True)

    for app, clusters in segment_clusters.items():
        for cluster_info in clusters:
            start_time = cluster_info["start_time"]
            end_time = cluster_info["end_time"]

            current_segment_time = []
            current_segment_label = []
            current_segment_packet = []

            for ix, timestamp in enumerate(data["timestamps"]):
                if start_time <= timestamp <= end_time:
                    current_segment_time.append(timestamp)
                    current_segment_label.append(1 if data["labels"][ix] == app else 0)
                    current_segment_packet.append(data["packet_sizes"][ix])
                elif timestamp > end_time:
                    break  # timestamps assumed sorted

            segments[app].append(
                {
                    "timestamps": current_segment_time,
                    "labels": current_segment_label,
                    "packet_sizes": current_segment_packet,
                }
            )

        # Save this app's segments to its own JSON file
        app_file = os.path.join(save_dir, f"{app}.json")
        with open(app_file, "w") as f:
            json.dump(segments[app], f, indent=2)

        print(f"Saved {len(segments[app])} segments for {app} to {app_file}")

    return segments


def load_segment(
    app: str, input_dir: str = "data/segments_per_app"
) -> List[Dict[str, Any]]:
    """
    Load the saved segments for a single app.

    Args:
        app: The app name (without .json extension).
        input_dir: Directory where per-app segment JSON files are stored.

    Returns:
        List of segment dicts, each with keys:
          - "timestamps": list[float]
          - "labels": list[int]
          - "packet_sizes": list[int]
    """
    file_path = os.path.join(input_dir, f"{app}.json")
    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"No saved segments found for app '{app}' at {file_path}"
        )

    with open(file_path, "r") as f:
        segments = json.load(f)

    return segments


# def compute_s_similarity(model : xgb.XGBClassifier, X):
#     """
#     Computes the S-similarity as the probability from XGBoost model.
#     """
#     probas = model.predict_proba(X)[:, 1]  # Probability that y == 1
#     return probas


# Example usage
# if __name__ == "__main__":
#     # Example data: directional packet sizes, where sign = direction
#     # Positive = outbound, Negative = inbound
#     packet_sizes = [100, -120, 110, -130, 90, -150, 95, -110, 105, -115]

#     # Simulated labels: 1 = app of interest, 0 = noise
#     labels = [1, 1, 1, 0, 0, 0, 1, 1, 0, 0]

#     # Create N-gram sequential contexts
#     N = 2
#     X = generate_ngram_context(packet_sizes, N)
#     y = np.array(labels)

#     # Train model
#     X_train, X_test, y_train, y_test = train_test_split(
#         X, y, test_size=0.3, random_state=42
#     )
