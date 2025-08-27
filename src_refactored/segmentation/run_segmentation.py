# from s_similarity import train_sxgboost_all_apps
from segmentation.compute_ssimilarity import (
    train_sxgboost,
    compute_s_similarity,
)
from segmentation.identify_anchor_packets import identify_anchor_packets
from segmentation.segments import get_segments
from segmentation.hac_segment_clustering import hac_clustering


def run_segmentation(
    app_key,
    concatenated_training_data,
    split_filtered_data_to_train={},
    epsilon=30,
    load_precomputed=False,
):
    if load_precomputed:
        return get_segments(app_key)
    models = train_sxgboost(app_key, split_filtered_data_to_train, load_pretrained=True)
    scores = compute_s_similarity(models, concatenated_training_data)
    anchor_timings = identify_anchor_packets(concatenated_training_data, scores)
    clusters = hac_clustering(anchor_timings, epsilon=epsilon)
    segments = get_segments(app_key, clusters, concatenated_training_data)
    return segments
