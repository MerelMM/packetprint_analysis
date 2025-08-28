import numpy as np
import os
import pickle
from typing import List, Tuple


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
    lfm_features: np.ndarray, lfm_labels: np.ndarray
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """this function for if enough negative segments were already in the proposed segments"""
    D_pos = []
    D_neg = []
    for ix, label in enumerate(lfm_labels):
        if label == 1:
            D_pos.append(lfm_features[ix].astype(int))
        elif label == 0:
            D_neg.append(lfm_features[ix].astype(int))
        else:
            raise Exception("label should be 0 or 1")
    if len(D_pos) > len(D_neg):
        print(
            "There are more positive segments than negative ones so might want to add some additional ones here."
        )
    if len(D_pos) == 0:
        raise Exception("D_pos should not be empty")
    if len(D_neg) == 0:
        raise Exception
    return D_pos, D_neg


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
