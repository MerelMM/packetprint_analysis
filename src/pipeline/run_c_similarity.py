from c_similarity.input_lfm import (
    construct_input_lfm_per_app,
)
from c_similarity.recursive_lfm_training import (
    recursive_lfm_training,
    get_segment_features,
)
from c_similarity.greedy_feature_representation import greedy_feature_representation
import json

import json
from typing import Dict, Any
import os
from s_similarity import load_segment


def create_input_features_lfm(
    segments_path: str = "data/segments.json",
    N: int = 1,
    nf: int = 20,
    alpha: float = 1.0,
    min_pos_fraction: float = 0.5,
    max_depth: int = 5,
) -> Dict[str, Any]:
    """
    Full pipeline to construct LFM features from segmented packet data.

    Args:
        segments_path: Path to the JSON file containing packet segments.
        N: Half-width of N-gram window.
        nf: Number of merged words (final feature dimension).
        alpha: Weight for balancing positive and negative samples in greedy merging.
        min_pos_fraction: Minimum fraction of positive samples in a leaf to consider it positive.
        max_depth: Max depth for decision trees.

    Returns:
        A dictionary containing vocabulary, models, final cj list, and merged feature data.
    """
    # with open(segments_path, "r") as f:
    #     segments_all_apps = json.load(f)

    # Step 1: Extract hierarchical features from all apps
    # features_all = construct_input_lfm_all_apps(
    #     segments_all_apps, N=N, load_features=True
    # )
    app = "com.meetup"
    segments = load_segment(app)
    features = construct_input_lfm_per_app(app, segments, load_features=True)
    # feature will have label 1 when there exists a positive sample in the window
    recursive_lfm_training(features, app)
    d_pos, d_neg = get_segment_features(app)
    c = greedy_feature_representation(d_pos, d_neg)

    # Step 2: Build packet-level training set for app A vs others
    # packets = []
    # window_2s = []
    # window_5s = []
    # all_window_words = []

    # for app, feature_list in features_all.items():
    #     for feat in feature_list:
    #         label = 1 if app == "A" else 0
    #         for pkt in feat["burst_features"]:
    #             packets.append((pkt, label))
    #         window_2s.append((feat["burst_features"], label))
    #         window_5s.append((feat["behavior_features"], label))
    #         all_window_words.append(feat["burst_features"])
    #         all_window_words.append(feat["behavior_features"])

    # Step 3: Compute IDF #if-dict: {'packet_features', 'burst_features', 'behavior_f'}
    # idf_dict = compute_packet_idf_for_app(app, features)

    # Step 4: Recursive LFM Training (get vocabulary and models)

    # Step 5: Prepare inputs for greedy word merging


def check_size_lfm_features_together(path="data/lfm_features.pkl"):
    size_bytes = os.path.getsize(path)
    size_mb = size_bytes / (1024 * 1024)
    print(f"File size: {size_mb:.2f} MB")
