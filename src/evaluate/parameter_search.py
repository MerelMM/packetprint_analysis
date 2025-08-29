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


def compare_hyper_parameters(app_key="com.google.android.youtube"):
    os.makedirs("hyperparam_results", exist_ok=True)

    ###########################
    # Segmentation parameters #
    ###########################

    # Both already after modeltraining, for the segmentation itself
    phi_mins = [
        0.6,
        0.75,
        0.95,
    ]  # 0.95 in pp -> making it lower can increase number of segments
    epsilons = [20, 30, 50]  # 300s in pp

    ##########################
    # Recognition parameters #
    ##########################

    # lfm parameter
    voc_tree_depth = [7, 10, 12]

    # greedy feature representation parameters
    alphas = [0.1, 0.2, 0.3]  # 0.1 in pp
    nfs = [3, 5, 7]  # compression_size, PP=3

    # c_similarity parameter
    psi_mins = [0.05, 0.1, 0.2, 0.3]  # c_sim threshold, PP=0.1

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

    ######################
    # Training data setup #
    ######################
    filtered_data = run_preprocessing(
        app_key, load_existing=True
    )  # will recompute filtering sizes and return filtered data for training s-xgboost
    concatenated_train_data = concat_traces(
        app_key, load_existing=True, load_existing_filter=True
    )

    for epsilon in epsilons:
        for phi_min in phi_mins:
            # trains models and returns segments of the concatenated training data
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
                t = (
                    seg["timestamps"][0]
                    + (seg["timestamps"][-1] - seg["timestamps"][0]) / 2
                )
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
            for depth in voc_tree_depth:
                # training:
                recursive_lfm_training(
                    app_key, raw_segment_features, max_tree_depth=depth
                )
                # Convert raw -> LFM features + labels
                lfm_fvs, lfm_labels, _ = raw_features_to_lfm(
                    app_key, raw_segment_features
                )

                lfm_fvs_test, _lfm_labels_test, reference_times_proposed_segments = (
                    raw_features_to_lfm(
                        app_key,
                        raw_segment_features_test,
                        seg_timings=reference_times_proposed_segments_exp,
                    )
                )

                # for training
                D_pos, D_neg, lfm_fvs, lfm_labels = (
                    create_D_pos_and_neg_direct_from_lfm(app_key, lfm_fvs, lfm_labels)
                )

                for alpha in alphas:
                    for nf in nfs:
                        _ = train_greedy_feature_representation(
                            app_key, D_pos, D_neg, alpha=alpha, nf=nf
                        )
                        fvs = compress_segment_fv(app_key, lfm_fvs, training=True)
                        fvs_test = compress_segment_fv(
                            app_key, lfm_fvs_test, training=False
                        )

                        train_c_similarity_classifier(app_key, fvs, lfm_labels)

                        for psi_min in psi_mins:
                            # result = compute_c_similarity(
                            #     app_key, fvs, threshold=psi_min
                            # ) nothing is done with training result
                            predictions = compute_c_similarity(
                                app_key, fvs_test, threshold=psi_min
                            )

                            true_positives = 0
                            false_positives = 0
                            matched_gt = set()
                            match_counts = Counter()

                            for pred_time, prediction in zip(
                                reference_times_proposed_segments, predictions
                            ):
                                if prediction == 1:
                                    matched = False
                                    for i, (start, end) in enumerate(segment_times):
                                        if start <= pred_time <= end:
                                            match_counts[
                                                i
                                            ] += 1  # count even if matched before
                                            if i not in matched_gt:
                                                matched_gt.add(i)
                                                true_positives += 1
                                            matched = True
                                            break
                                    if not matched:
                                        false_positives += 1

                            false_negatives = len(segment_times) - len(matched_gt)

                            # Final reporting
                            print(
                                f"Detected correctly (true positives): {true_positives}"
                            )
                            print(
                                f"Missed segments (false negatives): {false_negatives}"
                            )
                            print(
                                f"Wrongfully detected (false positives): {false_positives}"
                            )

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
                                "true_positives": true_positives,
                                "false_positives": false_positives,
                                "false_negatives": false_negatives,
                                "precision": true_positives
                                / (true_positives + false_positives + 1e-8),
                                "recall": true_positives
                                / (true_positives + false_negatives + 1e-8),
                            }

                            result_file = os.path.join(
                                "hyperparam_results", f"result_{app_key}.jsonl"
                            )
                            with open(result_file, "a") as f:
                                f.write(json.dumps(result) + "\n")
