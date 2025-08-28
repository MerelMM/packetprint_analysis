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
    concatenated_data=None,
    split_filtered_data_to_train={},
    epsilon=30,
    load_precomputed=False,
    load_pretrained=False,
):
    if load_precomputed:
        return get_segments(app_key, load_precomputed=load_precomputed)
    models = train_sxgboost(
        app_key, split_filtered_data_to_train, load_pretrained=load_pretrained
    )
    scores = compute_s_similarity(models, concatenated_data)
    anchor_timings = identify_anchor_packets(concatenated_data, scores)
    clusters = hac_clustering(anchor_timings, epsilon=epsilon)
    segments = get_segments(app_key, clusters, concatenated_data)
    return segments
