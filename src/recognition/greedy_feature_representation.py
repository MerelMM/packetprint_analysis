# This script was developed with assistance from ChatGPT (OpenAI) and Github Copilot
# Final implementation and adaptation by Merel Haenaets.
import numpy as np
import os
import pickle
from typing import List, Tuple
import random
from helper.helper import get_app_key
from recognition.lfm import raw_features_to_lfm
from recognition.extract_raw_features import extract_raw_features_segment


def compress_segment_fv(
    app_key, fv_lfm, load_path="data/compression_groups/", training=False
):
    """
    Compresses LFM features into lower-dim vectors using the learned C matrix and applies tf-idf weighting.

    Args:
        app_key (str): App name used to load the corresponding C matrix.
        fv_lfm (List[np.ndarray]): Original high-dimensional binary feature vectors.
        load_path (str): Directory where the C matrix and IDF are stored.
        training (bool): Whether to compute and save the IDF vector.

    Returns:
        np.ndarray: Compressed feature vectors z_t with tf-idf weighting applied.
    """
    # Load C matrix
    c_path = os.path.join(load_path, f"{app_key}_C.pkl")
    with open(c_path, "rb") as f:
        C = pickle.load(f)
    print(f"Loaded learned C matrix for {app_key} from {c_path}")

    num_docs = len(fv_lfm)

    # Step 1: compute tf for each sample
    tf_matrix = []
    for st in fv_lfm:
        zt = np.sum(C[st.astype(bool)], axis=0)  # tf: sum cj(k) for j in W(st)
        tf_matrix.append(zt)
    tf_matrix = np.array(tf_matrix)

    if training:
        # Step 2: compute idf and save it
        df = np.sum(tf_matrix >= 1, axis=0)
        idf = np.log(num_docs / (df + 1e-8))
        idf_path = os.path.join(load_path, f"{app_key}_idf.npy")
        np.save(idf_path, idf)
        print(f"Saved IDF vector for {app_key} to {idf_path}")
    else:
        # Load precomputed IDF
        idf_path = os.path.join(load_path, f"{app_key}_idf.npy")
        idf = np.load(idf_path)
        print(f"Loaded IDF vector for {app_key} from {idf_path}")

    # Step 3: apply tf * idf
    compressed_fvs = tf_matrix * idf

    return compressed_fvs


def create_D_pos_and_neg_direct_from_lfm(
    app_key: str,
    lfm_features: np.ndarray,
    lfm_labels: np.ndarray,
    training_data_dir="capture_data_train",
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    Create positive and negative training sets directly from LFM features.
    If there are not enough negative samples, supplement them using filtered traffic from other apps.
    """
    D_pos = []
    D_neg = []

    # Split based on labels
    for ix, label in enumerate(lfm_labels):
        if label == 1:
            D_pos.append(lfm_features[ix].astype(int))
        elif label == 0:
            D_neg.append(lfm_features[ix].astype(int))
        else:
            raise Exception("Label should be 0 or 1")

    # If not enough negatives, supplement from other apps
    if True:  # len(D_pos) > len(D_neg) -- but now always 10 at least
        n_extra_segs = max((len(D_pos) - len(D_neg)), 0) + 10

        # Load filtered traffic
        traffic_path = f"data/split_filtered_data_{app_key}.pkl"
        with open(traffic_path, "rb") as f:
            filtered_packets = pickle.load(f)

        # Get shuffled list of session paths
        paths = os.listdir(training_data_dir)
        random.seed(42)
        random.shuffle(paths)

        new_segs = []
        used = 1

        for path in paths:
            if get_app_key(path) == app_key:  # don' add positive segments
                continue
            if used > n_extra_segs:
                break
            # Get segments from other app
            if path not in filtered_packets:  # data isn't there
                print(
                    "check, is weird that path isnt in filtered packets (D_neg creation when dpos>dneg)"
                )
                continue

            data = filtered_packets[path]
            timestamps = [t for (t, _) in data]
            packet_sizes = [s for (_, s) in data]
            new_segs.append(
                {
                    "timestamps": timestamps,
                    "packet_sizes": packet_sizes,
                    "labels": [0] * len(packet_sizes),
                }
            )
            used += 1

        if not new_segs:
            raise Exception("No new segments found to supplement negatives.")

        # Extract features from new segments
        raw_segment_features, _ = extract_raw_features_segment(
            app=app_key,
            segments=new_segs,
            load_features=False,
            seg_timings=None,
        )
        lfm_fvs_neg, _, _ = raw_features_to_lfm(
            app_key,
            raw_segment_features,
            seg_timings=None,
        )
        for fv in lfm_fvs_neg:
            D_neg.append(fv.astype(int))

    if len(D_pos) == 0:
        raise Exception("D_pos should not be empty")
    if len(D_neg) == 0:
        raise Exception("D_neg should not be empty")
    lfm_training_features_new = np.vstack([lfm_features, lfm_fvs_neg])
    lfm_training_labels_new = np.vstack([lfm_labels, np.zeros((len(lfm_fvs_neg), 1))])
    return D_pos, D_neg, lfm_training_features_new, lfm_training_labels_new


def train_greedy_feature_representation(
    app_key,
    D_pos: List[np.ndarray],
    D_neg: List[np.ndarray],
    save_path="data/compression_groups/",
    alpha=0.1,
    nf=3,
):
    """
    Trains a greedy feature representation (Algorithm 3) and saves it.

    Args:
        app_key (str): The app identifier (used for saving).
        D_pos (list[list[int]]): Positive training samples.
        D_neg (list[list[int]]): Negative training samples.
        save_path (str): Directory where result will be saved.
        alpha (float): Weighting factor for positive samples.
        nf (int): Number of features per vector.

    Returns:
        C (np.ndarray): Learned feature vector matrix.
    """
    if not D_pos:
        raise ValueError("D_pos should not be empty")
    if not D_neg:
        raise ValueError("D_neg should not be empty")

    vocab_size = len(D_pos[0])
    C = np.zeros((vocab_size, nf))

    # all combinations are checked
    for _ in range(vocab_size):

        # initiate the difference that adding a word to whatever group would make to 0
        delta_L = np.zeros((vocab_size, nf))

        for j in range(vocab_size):
            if (
                np.linalg.norm(C[j], ord=1) > 0
            ):  # jth word is already added to one of the compression groups
                continue

            # iterate through the groups
            for k in range(nf):

                # iterate through positive examples
                for stilde in D_pos:
                    if stilde[j]:  # check if jth word present in the positive segment
                        h_t = np.sum(C[stilde.astype(bool)], axis=0)  #
                        if h_t[k] == 0:
                            delta_L[j, k] += alpha / len(D_pos)

                for stilde in D_neg:
                    if stilde[j]:
                        h_t = np.sum(C[stilde.astype(bool)], axis=0)
                        if h_t[k] == 0:
                            delta_L[j, k] -= 1 / len(D_neg)

        # find the optimal group to add word j to
        j_star, k_star = np.unravel_index(np.argmax(delta_L), delta_L.shape)

        # if the total effect is worse, then don't assign it to the group
        if delta_L[j_star, k_star] <= 0:
            break

        # otherwise assign it to the group
        C[j_star, k_star] = 1

    # Save result
    os.makedirs(save_path, exist_ok=True)
    out_path = os.path.join(save_path, f"{app_key}_C.pkl")
    with open(out_path, "wb") as f:
        pickle.dump(C, f)
        print(f"Saved learned C matrix for {app_key} to {out_path}")

    return C
