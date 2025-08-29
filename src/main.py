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
from evaluate.train_and_evaluate import run_for_different_apps
from evaluate.parameter_search import compare_hyper_parameters
from evaluate.evaluate_segmentation import run_plotting_segmentation


def split_train_test():
    from train_test_split.train_test_split_sessions import (
        train_test_split_sessions,
        add_open_world_traces_to_train_test,
        # remove_empty_dirs,
        # remove_sessions_by_date,
    )

    # remove_empty_dirs("capture_data_train")
    # remove_empty_dirs("capture_data_test")
    # remove_empty_dirs("capture_data_many_apps")
    # train_test_split_sessions()
    add_open_world_traces_to_train_test()

    # remove_sessions_by_date(data_path="capture_data_test")
    # remove_sessions_by_date(data_path="capture_data_train")


def plots_preprocessing(app_key=None):
    from plots.plot_percent_packages_kept_per_app import (
        plot_percent_packets_kept_per_app,
    )

    plot_percent_packets_kept_per_app(app_key)
    # from plots.plot_packet_size_historgram import (
    #     plot_packet_size_histogram,
    # )

    # plot_packet_size_histogram()
    # done
    # from plots.plot_filtering_process import plot_filtering_process

    # plot_filtering_process()

    # # done
    # from plots.plot_top_packet_sizes_per_app import plot

    # plot()
    pass


if __name__ == "__main__":
    # # split_train_test()
    # # plots_preprocessing()
    app_key = "com.substack.app"
    # plots_preprocessing(app_key)
    # segments = run_segmentation(app_key, load_precomputed=True)
    # results = run_recognition_wrapper(app_key, segments, training=False)
    # print(results)
    # evaluate(app_key)
    # app_key = "com.pinterest"

    # app_key = "bbc.mobile.news.ww"
    # compare_hyper_parameters(app_key)
    # run_for_different_apps()
    run_plotting_segmentation()
""" 
Pipeline for in README:
- capture data in capture_data path (or chosen one, but easiest)
- 1. Move to training/test dirs
- 2. Run preprocessing: filters data and only keeps the most representative packet sizes in the trace for that specific app.
- 3. Train S-XGboost and use for segmentation
"""
