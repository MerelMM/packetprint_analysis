import numpy as np
from typing import List, Tuple, Dict, Any, Set
import numpy as np
from sklearn.tree import DecisionTreeClassifier
import math
import pickle
import numpy as np
from collections import defaultdict
import json
import os
import pandas as pd


def greedy_feature_representation(D_pos, D_neg, alpha=0.1, nf=3):
    """
    Direct transcription of Algorithm 3 pseudocode into Python.
    D_pos: list of sets of word indices (positive samples)
    D_neg: list of sets of word indices (negative samples)
    alpha: float
    nf: number of merged features
    vocab_size: |V| (number of original words)
    Returns:
        C: np.ndarray of shape (vocab_size, nf) where each row = c_j
    """
    vocab_size = len(D_pos(0))
    # 1. cj ← 0^(1×nf) for 1 ≤ j ≤ |V|
    C = np.zeros((vocab_size, nf), dtype=int)

    # 2. for i = 1, 2, ..., |V| do
    for i in range(vocab_size):
        # 3. for j = 1, 2, ..., |V| do
        for j in range(vocab_size):
            # 4. if ‖c_j‖_1 > 0 then
            if np.sum(C[j]) > 0:
                # 5. continue;
                continue
            # 7. for k = 1, 2, ..., nf do
            for k in range(nf):
                # 8. ΔL(j, k) ← 0;
                delta_L = 0.0
                # 9. foreach s̃_t ∈ D⁺_train do
                for s_t in D_pos:
                    # 10. h_t ← Σ_{l ∈ W(s̃_t)} c_l;
                    W_st = list(s_t)
                    h_t = np.sum(C[W_st], axis=0) if W_st else np.zeros(nf, dtype=int)
                    # 11. if h_t(k) == 0 then
                    if h_t[k] == 0:
                        # 12. ΔL(j, k) ← ΔL(j, k) + α / |D⁺_train|;
                        delta_L += alpha / len(D_pos)
                # 15. foreach s̃_t ∈ D⁻_train do
                for s_t in D_neg:
                    # 16. h_t ← Σ_{l ∈ W(s̃_t)} c_l;
                    W_st = list(s_t)
                    h_t = np.sum(C[W_st], axis=0) if W_st else np.zeros(nf, dtype=int)
                    # 17. if h_t(k) == 0 then
                    if h_t[k] == 0:
                        # 18. ΔL(j, k) ← ΔL(j, k) − 1 / |D⁻_train|;
                        delta_L -= 1.0 / len(D_neg)
                # store ΔL(j,k) for argmax
                if i == 0 and j == 0 and k == 0:
                    # create the storage on first use
                    delta_matrix = np.full((vocab_size, nf), -np.inf)
                delta_matrix[j, k] = delta_L
        # 23. j*, k* ← arg max_{j,k} ΔL(j, k);
        j_star, k_star = np.unravel_index(np.argmax(delta_matrix), delta_matrix.shape)
        # 24. if ΔL(j*, k*) ≤ 0 then
        if delta_matrix[j_star, k_star] <= 0:
            # 25. break;
            break
        else:
            # 27. c_{j*}(k*) ← 1;
            C[j_star, k_star] = 1
    return C
