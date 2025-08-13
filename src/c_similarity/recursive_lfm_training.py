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
from c_similarity.input_lfm import construct_input_lfm_per_app


#####
# TODO: maak segment features ipv window 5 features!!
######
def get_segment_features(
    app, lfm_dir="data/lfm_features"
) -> Tuple[List[set], List[set]]:
    """
    Construeert de trainingsdata D+ en D- voor greedy feature selectie.

    Returns:
        D_pos: Lijst van sets met indices van actieve woorden voor positieve samples.
        D_neg: Idem voor negatieve samples.
    """
    D_pos = []
    D_neg = []

    # Laad de gelabelde LFM feature mapping voor de target app
    load_path = f"data/{app}/lfm_models.pkl"
    with open(load_path, "rb") as f:
        V, model1, stage2_info, stage3_info = pickle.load(f)

    idf_2 = stage2_info["idf"]
    model_s2 = stage2_info["model_s2"]
    non_empty_burst_window = stage2_info["non_empty_burst_window"]

    idf_3 = stage3_info["idf"]
    model_s3 = stage3_info["model_s3"]
    non_empty_behavior_window = stage3_info["non_empty_behavior_window"]
    labels_corr_app = stage3_info["labels_corr_app"]
    features_vectors_corr_app = stage3_info["features_vectors_corr_app"]

    # Models for level 1, 2, 3
    models = [None, model1, model_s2, model_s3]
    V0 = V[0]
    V1 = V[1]

    # Voeg positieve en negatieve voorbeelden van target app toe
    for i, label in enumerate(labels_corr_app):
        active_indices = set(np.where(features_vectors_corr_app[i] > 0)[0])
        if label == 1:
            D_pos.append(active_indices)
        else:
            D_neg.append(active_indices)

    if len(D_neg) >= len(D_pos):
        return D_neg, D_pos
    else:
        print(
            "Here we have a case that there where not enough negative parts in the proposed windows"
        )

    # Loop over ALLE ANDERE apps (excl. target app)
    for other_app in os.listdir(lfm_dir):
        if other_app == app:
            continue

        app_path = os.path.join(lfm_dir, other_app)
        try:
            features = construct_input_lfm_per_app(other_app, load_features=True)
            features = features[0]
        except Exception as e:
            print(f"[!] Fout bij laden van features voor {other_app}: {e}")
            continue

        window_2s = features["burst_lvl"]
        window_5s = features["behavior_lvl"]

        cur_window = 0
        fv = []
        y = []

        while cur_window < len(window_5s):
            if len(window_5s[cur_window]["behavior_features"]) == 0:
                cur_window += 1
                continue

            z_s2, _, _, _ = zt_bursts(
                window_2s[cur_window : cur_window + 5],
                V,
                models,
                idf_2,
                non_empty_burst_window,
            )
            if len(z_s2) == 0:
                cur_window += 1
                continue

            words_packets = np.unique(
                window_5s[cur_window]["behavior_features"], axis=0
            )
            leafs1 = models[1].apply(words_packets)
            leafs2 = models[2].apply(z_s2)

            z_t = np.zeros(len(V0) + len(V1), dtype=float)
            for ix, v in enumerate(V0):
                if np.any(leafs1 == v):
                    z_t[ix] = 1.0
            for ix, v in enumerate(V1):
                if np.any(leafs2 == v):
                    z_t[len(V0) + ix] = 1.0

            label = window_5s[cur_window]["label"]
            fv.append(z_t)
            y.append(label)
            cur_window += 1

        # IDF normalisatie
        if fv:
            fv = np.vstack(fv)
            for ix, df in idf_3.items():
                fv[:, ix] *= math.log(non_empty_behavior_window / df)
            y = np.asarray(y)

            for i, label in enumerate(y):
                active_indices = set(np.where(fv[i] > 0)[0])
                if label == 1:
                    D_pos.append(active_indices)
                else:
                    D_neg.append(active_indices)

    return D_pos, D_neg


def recursive_lfm_training(
    features,
    app,
) -> Tuple[
    List[List[int]],
    DecisionTreeClassifier,
    DecisionTreeClassifier,
    DecisionTreeClassifier,
]:
    """
    Recursive LFM Training as in Algorithm 2.

    Returns:
        (V, M1, M2, M3) where V = [V0, V1, V2] (lists of positive-leaf IDs per level)
    """
    packets = features["packet_lvl"]
    window_2s = features["burst_lvl"]
    window_5s = features["behavior_lvl"]
    V: List[List[int]] = []  # FIX: make it explicit this is a list per level
    models: dict[int, DecisionTreeClassifier] = {}

    for s in [1, 2, 3]:
        if s == 1:
            z = np.asarray(packets["packets"])
            y = np.asarray(packets["labels"])
        elif s == 2:
            z, y, idf_2, non_empty_burst_window = zt_bursts(window_2s, V, models)
        else:  # s == 3
            z, y, idf_3, non_empty_behavior_window = zt_behavior(
                window_5s, window_2s, V, idf_2, models, non_empty_burst_window
            )

        # train mapper at the current scale
        model = DecisionTreeClassifier(max_depth=7)
        if len(z) > 0:
            model.fit(z, y)
        models[s] = model
        # extract positive leaves as words for this level
        pos_leaves = extract_positive_leaves(model, z, y)  # returns list[int]
        V.append(list(pos_leaves))

    result = (
        V,
        models[1],
        {
            "idf": idf_2,
            "model_s2": models[2],
            "non_empty_burst_window": non_empty_burst_window,
        },
        {
            "idf": idf_3,
            "model_s3": models[3],
            "non_empty_behavior_window": non_empty_behavior_window,
            "features_vectors_corr_app": z,
            "labels_corr_app": y,
        },
    )
    save_path = f"data/{app}/lfm_models.pkl"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "wb") as f:
        pickle.dump(result, f)
    return result


def zt_behavior(window_5s, window_2s, V, idf_2, models, non_empty_burst_window):
    V0 = V[0]
    V1 = V[1]
    labels = []
    fv = []
    labels = []
    non_empty_behavior_window = 0  # extend id for the new features
    cur_window = 0
    idf = defaultdict(int)
    while cur_window < len(window_5s):
        if len(window_5s[cur_window]["behavior_features"]) == 0:
            cur_window += 1
            continue

        z, y, _, _ = zt_bursts(
            window_2s[cur_window : cur_window + 5],
            V,
            models,
            idf_2,
            non_empty_burst_window,
        )
        if len(z) == 0:
            cur_window += 1
            continue  # even when the behavior window is not empty, there might be too little data per second to have burst features
        # no guidelines where given in this case of how to handle, so I will just decide to not add it as feature_vectors if there's 0 bursts
        #  but will add it once there's some burst.
        # Currently using a clustering treshold of 30 s since 300s merged all segments, so this might increase this occurence. However, collecting more data might alleviate the 300 s issue further possibly
        non_empty_behavior_window += 1
        words_packets = np.unique(window_5s[cur_window]["behavior_features"], axis=0)

        leafs1 = models[1].apply(words_packets)
        leafs2 = models[2].apply(z)

        z_t = np.zeros(len(V0) + len(V1), dtype=float)
        for ix, v in enumerate(V0):  # FIX: iterate V0, not V
            if np.any(leafs1 == v):
                z_t[ix] = 1.0
                idf[ix] += 1
        for ix, v in enumerate(V1):  # FIX: iterate V0, not V
            if np.any(leafs2 == v):
                z_t[len(V0) + ix] = 1.0
                idf[len(V0) + ix] += 1

        labels.append(window_5s[cur_window]["label"])
        fv.append(z_t)
        cur_window += 1

    if fv:
        fv = np.vstack(fv)
        # Boolean tf * idf using count of non-empty windows i
        for ix, df in idf.items():
            fv[:, ix] *= math.log(non_empty_behavior_window / df)
        z = fv
        y = np.asarray(labels)
    else:
        z = np.empty((0, len(V0)))
        y = np.empty((0,))


def zt_bursts(
    window_2s, V, models, idf_already_computed=None, non_empty_burst_window=None
):
    assert len(V) >= 1, "V0 must exist before level 2"
    V0 = V[0]
    data = window_2s
    idf = defaultdict(int)
    fv = []
    labels = []
    non_empty_windows = 0
    for window in data:
        words = window.get("burst_features", [])
        if len(words) == 0:
            continue
        non_empty_windows += 1
        words = np.unique(words, axis=0)
        # apply M1 to all packet-level words in the burst
        leafs = models[1].apply(np.asarray(words))
        # Boolean TF vector over V0
        z_t = np.zeros(len(V0), dtype=float)  # FIX: length is len(V0)
        for ix, v in enumerate(V0):  # FIX: iterate V0, not V
            if np.any(leafs == v):
                z_t[ix] = 1.0
                idf[ix] += 1
        labels.append(window["label"])
        fv.append(z_t)
    if fv:
        fv = np.vstack(fv)
        if non_empty_burst_window is not None:
            non_empty_windows = non_empty_burst_window
        # Boolean tf * idf using count of non-empty windows i
        if idf_already_computed is not None:
            idf = idf_already_computed
        for ix, df in idf.items():
            fv[:, ix] *= math.log(non_empty_windows / df)
        z = fv
        y = np.asarray(labels)
    else:
        z = np.empty((0, len(V0)))
        y = np.empty((0,))
    return z, y, idf, non_empty_windows


def extract_positive_leaves(model, z, y, min_pos_fraction=0.5):
    """
    Return the set of 'words' = leaf IDs whose positive fraction >= threshold.
    """
    leaf_ids = model.apply(z)  # (n_samples,)
    vocab = set()
    for leaf in np.unique(leaf_ids):
        idx = leaf_ids == leaf
        pos_frac = (y[idx] == 1).mean()
        if pos_frac >= min_pos_fraction:
            vocab.add(int(leaf))  # this leaf ID is a word
    return vocab
