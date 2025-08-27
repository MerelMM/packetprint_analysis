from helper.helper import get_app_key
from typing import Dict, List
import os
import xgboost as xgb
from sklearn.utils import resample
import numpy as np
from collections import defaultdict


def train_sxgboost(
    target_app: str,
    data_per_sessions: Dict[str, Dict[str, List[int]]] = {},
    N=6,
    save_dir="saved_models",
    load_pretrained=False,
):
    """
    Trains XGBoost models for a single app (vs all others).

    Args:
        target_app: The app key to train a model for.
        data_per_sessions: Dict mapping session_id -> {"train": [(time, size)...], "test": [(time, size)...]}
        N: Number of sub-models to train (default=6)
        save_dir: Directory where trained models will be saved

    Returns:
        models: List of trained XGBoost models for the app
    """
    if load_pretrained:
        return load_sxgboost_models(target_app, save_dir=save_dir)
    # Convert to app-wise representation
    data = convert_sessions_to_size_dict_per_app(data_per_sessions)

    if target_app not in data:
        raise ValueError(f"App '{target_app}' not found in data.")

    positive_data = data[target_app]

    # Combine data from all other apps
    negative_data = []
    for app, packets in data.items():
        if app != target_app:
            negative_data.extend(packets)

    # Train models
    model_list = sxgboost_trainer(positive_data, negative_data, N=N)

    # Save each model
    app_name = get_app_key(target_app)
    app_dir = os.path.join(save_dir, app_name)
    os.makedirs(app_dir, exist_ok=True)
    for idx, model in enumerate(model_list):
        model_file = os.path.join(app_dir, f"model_{idx}.json")
        model.save_model(model_file)

    print(f"Saved {len(model_list)} models for app '{app_name}' in {app_dir}")
    return model_list


# def train_sxgboost_all_apps(
#     data_per_sessions: Dict[str, Dict[str, List[int]]],
#     apps_to_train=None,
#     N=6,
#     save_dir="saved_models",
# ):
#     """
#     Trains one XGBoost model per app, distinguishing it from all other apps.

#     Args:
#         data: Dict mapping app -> {"train": [sizes...], "test": [sizes...]}
#         N: Optional hyperparameter for sxgboost_trainer

#     Returns:
#         models: Dict mapping app -> trained model
#     """
#     data = convert_sessions_to_size_dict_per_app(data_per_sessions)  # format
#     models = {}

#     for app, packets in data.items():
#         positive_data = packets

#         # Combine data from all other apps
#         negative_data = []
#         for other_app, other_packets in data.items():
#             if other_app != app:
#                 negative_data.extend(other_packets)

#         # Train binary model for this app

#         # Train binary model for this app
#         model_list = sxgboost_trainer(positive_data, negative_data, N=N)

#         # Save each sub-model
#         app_name = get_app_key(app)
#         app_dir = os.path.join(save_dir, app_name)
#         os.makedirs(app_dir, exist_ok=True)
#         for idx, model in enumerate(model_list):
#             model_file = os.path.join(app_dir, f"model_{idx}.json")
#             model.save_model(model_file)
#         models[app_name] = model_list
#         print(f"Saved {len(model_list)} models for app '{app_name}' in {app_dir}")
#     return models


def sxgboost_trainer(X_pos, X_neg, N):
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


def create_N_grams_per_app(packet_data, N=6):
    """
    Constructs N-gram features for each packet in a sequence.
    """
    features = np.zeros([len(packet_data) - 2 * N, 2 * N + 1])
    for i in range(N, len(packet_data) - N):
        features[i - N] = packet_data[i - N : i + N + 1]
    return features


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
    packet_sequence = packet_sequence[0]
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


def convert_sessions_to_size_dict_per_app(data_sessions):
    """data_session: Dict(Session_name: [(time, size)])"""
    training_data = defaultdict(list)
    for session_key, session_data in data_sessions.items():
        app_key = get_app_key(session_key)
        data = [size for _time, size in session_data]
        training_data[app_key].extend(data)
    return training_data
