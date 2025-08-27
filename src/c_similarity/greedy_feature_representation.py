import numpy as np


def greedy_feature_representation(D_pos, D_neg, alpha, vocab_size, nf):
    """
    Implements Algorithm 3: Greedy Feature Representation.

    Parameters:
        D_pos (list of list of int): Positive training samples, each as list of vocabulary indices
        D_neg (list of list of int): Negative training samples, each as list of vocabulary indices
        alpha (float): Weighting factor for positive samples
        vocab_size (int): Size of vocabulary |V|
        nf (int): Number of features per vector

    Returns:
        C (np.ndarray): Matrix of shape (vocab_size, nf) with learned feature vectors cj
    """
    # Step 1: Initialize cj ← 0 (shape: |V| x nf)
    C = np.zeros((vocab_size, nf))

    while True:
        delta_L = np.zeros((vocab_size, nf))

        # Step 2–3: For all j not yet selected (‖cj‖1 == 0)
        for j in range(vocab_size):
            if np.linalg.norm(C[j], ord=1) > 0:
                continue

            # Step 7–21: For each feature k
            for k in range(nf):
                # Step 9–14: Positive samples
                for stilde in D_pos:
                    h_t = np.sum(C[stilde], axis=0)
                    if h_t[k] == 0:
                        delta_L[j, k] += alpha / len(D_pos)

                # Step 15–20: Negative samples
                for stilde in D_neg:
                    h_t = np.sum(C[stilde], axis=0)
                    if h_t[k] == 0:
                        delta_L[j, k] -= 1 / len(D_neg)

        # Step 23: Select the best (j*, k*) pair
        j_star, k_star = np.unravel_index(np.argmax(delta_L), delta_L.shape)

        # Step 24: Check stopping condition
        if delta_L[j_star, k_star] <= 0:
            break

        # Step 27: Assign 1 to cj*(k*)
        C[j_star, k_star] = 1

    return C
