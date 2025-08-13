# from filter_packages import filter_packages
# from filter_packages import get_filtered_data
# from s_similarity import train_sxgboost_all_apps, compute_s_similarity
from pipeline.run_s_similarity import (
    test_training_sxgboost,
    test_taking_s_similarity,
    train_sxgboost_on_all_captures,
    run_all_s_similarity,
    plot_s_similarity_pos_neg,
    serialized_input_s_similarity,
    run_hac_clustering,
    identify_anchor_packets,
    run_hac_clustering,
    get_segments_input_c_similarity,
)
from pipeline.run_c_similarity import (
    create_input_features_lfm,
    check_size_lfm_features_together,
)

if __name__ == "__main__":
    # packages = filter_packages()
    # test_training_sxgboost()
    # test_taking_s_similarity()
    # train_sxgboost_on_all_captures()
    # run_all_s_similarity()
    # plot_s_similarity_pos_neg(
    # results_path="data/s_sxgboost_pairwise_results.json",
    # save_plot_path="plots/s_similarity_pos_vs_neg.png",
    # )
    # serialized_input_s_similarity()
    # run_hac_clustering()
    # identify_anchor_packets()
    # run_hac_clustering()
    # get_segments_input_c_similarity()
    create_input_features_lfm()
    # check_size_lfm_features_together()
