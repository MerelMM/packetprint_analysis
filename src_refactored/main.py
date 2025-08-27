from preprocessing.run_preprocessing import run_preprocessing
from preprocessing.packet_size_filtering import filter_packet_size_for_app
from segmentation.run_segmentation import run_segmentation
from preprocessing.concat_training_traces import concat_training_traces
from recognition.run_recognition import run_recognition


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
    # from plots.plot_percent_packages_kept_per_app import (
    #     plot_percent_packets_kept_per_app,
    # )

    # plot_percent_packets_kept_per_app(app_key)
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


def train_from_scratch(app_key="com.google.android.youtube"):
    filtered_data = run_preprocessing(
        app_key, load_existing=False
    )  # will recompute filtering sizes and return filtered data for training s-xgboost
    serialized_traces = concat_training_traces(
        app_key, load_existing=False, load_existing_filter=False
    )

    # trains models and returns segments of the concatenated training data
    segments = run_segmentation(
        app_key,
        concatenated_training_data=serialized_traces,
        split_filtered_data_to_train=filtered_data,
        load_precomputed=False,
    )


if __name__ == "__main__":
    # split_train_test()
    # plots_preprocessing()
    app_key = "com.google.android.youtube"
    segments = run_segmentation(app_key, load_precomputed=True)
    run_recognition(
        app_key, segments, load_precomputed_features=True, pretrained_lfm=True
    )
""" 
Pipeline for in README:
- capture data in capture_data path (or chosen one, but easiest)
- 1. Move to training/test dirs
- 2. Run preprocessing: filters data and only keeps the most representative packet sizes in the trace for that specific app.
- 3. Train S-XGboost and use for segmentation
"""
