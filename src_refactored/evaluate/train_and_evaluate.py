# This script was developed with assistance from ChatGPT (OpenAI) and Github Copilot
# Final implementation and adaptation by Merel Haenaets.
from preprocessing.run_preprocessing import run_preprocessing
from preprocessing.packet_size_filtering import filter_packet_size_for_app
from segmentation.run_segmentation import run_segmentation
from preprocessing.concat_traces import concat_traces
from recognition.run_recognition import run_recognition
from evaluate.evaluate import evaluate
from recognition.extract_raw_features import extract_raw_features_segment
from recognition.lfm import recursive_lfm_training, raw_features_to_lfm
from recognition.extract_raw_features import extract_raw_features_segment
from recognition.greedy_feature_representation import (
    train_greedy_feature_representation,
    create_D_pos_and_neg_direct_from_lfm,
    compress_segment_fv,
)
from recognition.c_similarity import train_c_similarity_classifier, compute_c_similarity
from collections import Counter
import os
import json


def train_and_evaluate(app_key="com.google.android.youtube"):
    path = "evaluation_results_conf2"
    os.makedirs(path, exist_ok=True)
    # phi_min = 0.75
    # epsilon = 20
    # depth = 10
    # alpha = 0.2
    # nf = 5
    # psi_min = 0.2

    phi_min = 0.75
    epsilon = 30
    depth = 12
    alpha = 0.3
    nf = 7
    psi_min = 0.3

    ######################
    # Testing data setup #
    ######################

    concatenated_test_data = concat_traces(
        app_key,
        data_path="capture_data_test",
        save_path="data/concatenated_test_trace.pkl",
        load_existing=True,
        load_existing_filter=True,
    )
    _concatenated_sizes, concatenated_timings, concatenated_labels = (
        concatenated_test_data
    )

    # Step 2: Extract ground-truth segments (start_time, end_time)
    segment_times = []
    ix = 0
    while ix < len(concatenated_labels):
        if concatenated_labels[ix] == app_key:
            start = concatenated_timings[ix]
            while ix < len(concatenated_labels) and concatenated_labels[ix] == app_key:
                ix += 1
            end = concatenated_timings[ix - 1]
            segment_times.append((start, end))
        else:
            ix += 1

    filtered_data = run_preprocessing(
        app_key, load_existing=False
    )  # will recompute filtering sizes and return filtered data for training s-xgboost
    concatenated_train_data = concat_traces(
        app_key, load_existing=True, load_existing_filter=True
    )

    segments_train = run_segmentation(
        app_key,
        concatenated_data=concatenated_train_data,
        split_filtered_data_to_train=filtered_data,
        epsilon=epsilon,
        anchor_threshold=phi_min,
        load_precomputed=False,
    )
    segments_test = run_segmentation(
        app_key,
        concatenated_data=concatenated_test_data,
        load_precomputed=False,  # don't load precomputed segments
        epsilon=epsilon,
        anchor_threshold=phi_min,
        load_pretrained=True,  # do use pretrained xgboost models
    )
    # Step 4: Extract center timestamp of each proposed segment
    reference_times_proposed_segments = []
    windows_of_reference_times = {}
    for seg in segments_test:
        t = seg["timestamps"][0] + (seg["timestamps"][-1] - seg["timestamps"][0]) / 2
        reference_times_proposed_segments.append(t)
        windows_of_reference_times[t] = (
            seg["timestamps"][0],
            seg["timestamps"][-1],
        )

    raw_segment_features, _ = extract_raw_features_segment(
        app_key,
        segments_train,
        load_features=False,  # both training and testing write to the same one!!
    )
    raw_segment_features_test, reference_times_proposed_segments = (
        extract_raw_features_segment(
            app_key,
            segments_test,
            load_features=False,  # both training and testing write to the same one!!
            seg_timings=reference_times_proposed_segments,
        )
    )
    reference_times_proposed_segments_exp = reference_times_proposed_segments
    # training:
    recursive_lfm_training(app_key, raw_segment_features, max_tree_depth=depth)
    # Convert raw -> LFM features + labels
    lfm_fvs, lfm_labels, _ = raw_features_to_lfm(app_key, raw_segment_features)

    lfm_fvs_test, _lfm_labels_test, reference_times_proposed_segments = (
        raw_features_to_lfm(
            app_key,
            raw_segment_features_test,
            seg_timings=reference_times_proposed_segments_exp,
        )
    )

    # for training
    D_pos, D_neg, lfm_fvs, lfm_labels = create_D_pos_and_neg_direct_from_lfm(
        app_key, lfm_fvs, lfm_labels
    )

    _ = train_greedy_feature_representation(app_key, D_pos, D_neg, alpha=alpha, nf=nf)

    fvs = compress_segment_fv(app_key, lfm_fvs, training=True)
    fvs_test = compress_segment_fv(app_key, lfm_fvs_test, training=False)

    train_c_similarity_classifier(app_key, fvs, lfm_labels)

    predictions = compute_c_similarity(app_key, fvs_test, threshold=psi_min)

    matched_gt = set()
    match_counts = Counter()

    true_positive_proposed = 0
    false_positive_proposed = 0
    false_negative_proposed = 0
    true_negative_proposed = 0

    # on proposed segment level
    for pred_time, prediction in zip(reference_times_proposed_segments, predictions):
        matched = False
        for i, (start, end) in enumerate(segment_times):
            if start <= pred_time <= end:
                if prediction == 1:
                    true_positive_proposed += 1
                    break
                else:
                    false_negative_proposed += 1
            else:
                if prediction == 1:
                    false_positive_proposed += 1
                    break
                else:
                    true_negative_proposed += 1

    true_positive_seg = 0
    false_positive_seg = 0
    false_negative_seg = 0
    true_negative_seg = 0
    # per segment level
    for pred_time, prediction in zip(reference_times_proposed_segments, predictions):
        if prediction == 1:
            matched = False
            for i, (start, end) in enumerate(segment_times):
                if start <= pred_time <= end:
                    match_counts[i] += 1  # count even if matched before
                    if i not in matched_gt:
                        matched_gt.add(i)
                        true_positive_seg += 1
                    matched = True
                    break
            if not matched:
                false_positive_seg += 1

    false_negative_seg = len(segment_times) - len(matched_gt)
    all = len(os.listdir("capture_data_test"))
    true_negative_seg = all - 5 - false_positive_seg
    # 5 is hardcoded here, but easy to change in a loop to check how many true positives
    # can not just get from tp + fp since 2 might have been concatenated and only count as 1 then

    # Final reporting
    print(f"Detected correctly (true positives): {true_positive_seg}")
    print(f"Missed segments (false negatives): {false_negative_seg}")
    print(f"Wrongfully detected (false positives): {false_positive_seg}")

    result = {
        "n_segments_proposed": len(predictions),
        "reference_times_proposed_segments": reference_times_proposed_segments,
        "predictions": predictions.tolist(),
        "windows_of_reference_times": windows_of_reference_times,
        "phi_min": phi_min,
        "epsilon": epsilon,
        "voc_tree_depth": depth,
        "alpha": alpha,
        "nf": nf,
        "psi_min": psi_min,
        # Original metrics
        # "true_positives": true_positives,
        # "false_positives": false_positives,
        # "false_negatives": false_negatives,
        # "precision": true_positives / (true_positives + false_positives + 1e-8),
        # "recall": true_positives / (true_positives + false_negatives + 1e-8),
        # Proposed-segment level metrics
        "true_positive_proposed": true_positive_proposed,
        "false_positive_proposed": false_positive_proposed,
        "false_negative_proposed": false_negative_proposed,
        "true_negative_proposed": true_negative_proposed,
        # Consolidated segment-level metrics
        "true_positive_seg": true_positive_seg,
        "false_positive_seg": false_positive_seg,
        "false_negative_seg": false_negative_seg,
        "true_negative_seg": true_negative_seg,
    }
    result_file = os.path.join(path, f"result_{app_key}.jsonl")
    with open(result_file, "a") as f:
        f.write(json.dumps(result) + "\n")


def run_for_different_apps():

    app_list = [
        "bbc.mobile.news.ww",
        "com.alltrails.alltrails",
        "com.decathlon.app",
        "com.google.android.youtube",
        "com.instagram.basel",
        "com.kiloo.subwaysurf",
        "com.pinterest",
        "com.spotify.music",
        # "com.substack.app",
        "com.trainingboard.moon",
    ]
    for app in app_list:
        train_and_evaluate(app)
