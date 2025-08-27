import numpy as np
import os
import pickle


def compress_segment_fv(app_key, fv_lfm, load_path="data/compression_groups/"):
    """
    Compresses LFM features into lower-dim vectors using the learned C matrix and applies tf-idf weighting.

    Args:
        app_key (str): App name used to load the corresponding C matrix.
        fv_lfm (List[np.ndarray]): Original high-dimensional binary feature vectors.
        load_path (str): Directory where the C matrix is stored.

    Returns:
        List[np.ndarray]: Compressed feature vectors z_t with tf-idf weighting applied.
    """
    # Load C matrix
    c_path = os.path.join(load_path, f"{app_key}_C.pkl")
    with open(c_path, "rb") as f:
        C = pickle.load(f)
    print(f"Loaded learned C matrix for {app_key} from {c_path}")

    # Ensure fv_lfm is a list of lists or a 2D NumPy array
    if isinstance(fv_lfm, np.ndarray):
        fv_lfm = fv_lfm.tolist()

    n_features = C.shape[1]
    num_docs = len(fv_lfm)

    # Step 1: compute tf for each sample
    tf_matrix = []
    for st in fv_lfm:
        zt = np.sum(C[st], axis=0)  # tf: sum cj(k) for j in W(st)
        tf_matrix.append(zt)
    tf_matrix = np.array(tf_matrix)

    # Step 2: compute idf vector
    df = np.sum(
        tf_matrix >= 1, axis=0
    )  # count how many docs have nonzero tf for each feature
    idf = np.log(num_docs / (df + 1e-8))  # prevent division by zero

    # Step 3: apply tf * idf
    compressed_fvs = tf_matrix * idf  # shape: [num_docs, n_features]

    return compressed_fvs


def create_D_pos_and_neg_direct_from_lfm(lfm_features, lfm_labels):
    """this function for if enough negative segments were already in the proposed segments"""
    D_pos = []
    D_neg = []
    for ix, label in enumerate(lfm_labels):
        if label == 1:
            D_pos.append(lfm_features[ix])
        elif label == 0:
            D_neg.append(lfm_features[ix])
        else:
            raise Exception("label should be 0 or 1")
    if len(D_pos) < len(D_neg):
        print(
            "There are more negative segments than positive ones so some opportunity to rectify is here."
        )
    if len(D_pos) == 0:
        raise Exception("D_pos should not be empty")
    if len(D_neg) == 0:
        raise Exception
    return D_pos, D_neg


def train_greedy_feature_representation(
    app_key, D_pos, D_neg, save_path="data/compression_groups/", alpha=0.1, nf=3
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

    while True:
        delta_L = np.zeros((vocab_size, nf))

        for j in range(vocab_size):
            if np.linalg.norm(C[j], ord=1) > 0:
                continue

            for k in range(nf):
                for stilde in D_pos:
                    h_t = np.sum(C[stilde], axis=0)
                    if h_t[k] == 0:
                        delta_L[j, k] += alpha / len(D_pos)

                for stilde in D_neg:
                    h_t = np.sum(C[stilde], axis=0)
                    if h_t[k] == 0:
                        delta_L[j, k] -= 1 / len(D_neg)

        j_star, k_star = np.unravel_index(np.argmax(delta_L), delta_L.shape)

        if delta_L[j_star, k_star] <= 0:
            break

        C[j_star, k_star] = 1

    # Save result
    os.makedirs(save_path, exist_ok=True)
    out_path = os.path.join(save_path, f"{app_key}_C.pkl")
    with open(out_path, "wb") as f:
        pickle.dump(C, f)
        print(f"Saved learned C matrix for {app_key} to {out_path}")

    return C
