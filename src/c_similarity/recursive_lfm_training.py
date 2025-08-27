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
# def get_segment_features(
#     app, app_segments, lfm_dir="data/lfm_features"
# ) -> Tuple[List[set], List[set]]:
#     """
#     Construeert de trainingsdata D+ en D- voor greedy feature selectie.

#     Returns:
#         D_pos: Lijst van sets met indices van actieve woorden voor positieve samples.
#         D_neg: Idem voor negatieve samples.
#     """
#     D_pos = []
#     D_neg = []

#     # Laad de gelabelde LFM feature mapping voor de target app
#     load_path = f"data/{app}/lfm_models.pkl"
#     with open(load_path, "rb") as f:
#         V, model1, stage2_info, stage3_info = pickle.load(f)

#     idf_2 = stage2_info["idf"]
#     model_s2 = stage2_info["model_s2"]
#     non_empty_burst_window = stage2_info["non_empty_burst_window"]

#     idf_3 = stage3_info["idf"]
#     model_s3 = stage3_info["model_s3"]
#     non_empty_behavior_window = stage3_info["non_empty_behavior_window"]
#     labels_corr_app = stage3_info["labels_corr_app"]
#     features_vectors_corr_app = stage3_info["features_vectors_corr_app"]

#     # Models for level 1, 2, 3
#     models = [None, model1, model_s2, model_s3]
#     V0 = V[0]
#     V1 = V[1]

#     # Voeg positieve en negatieve voorbeelden van target app toe
#     if len(D_neg) >= len(D_pos):
#         return D_neg, D_pos
#     else:
#         print(
#             "Here we have a case that there where not enough negative parts in the proposed windows"
#         )

#     # Loop over ALLE ANDERE apps (excl. target app)
#     for other_app in os.listdir(lfm_dir):
#         if other_app == app:
#             continue

#         app_path = os.path.join(lfm_dir, other_app)
#         try:
#             for segment in segments:
#                 features = construct_input_lfm_per_app(other_app, load_features=True)
#                 features = features[0]
#         except Exception as e:
#             print(f"[!] Fout bij laden van features voor {other_app}: {e}")
#             continue

#         window_2s = features["burst_lvl"]
#         window_5s = features["behavior_lvl"]

#         cur_window = 0
#         fv = []
#         y = []

#         while cur_window < len(window_5s):
#             if len(window_5s[cur_window]["behavior_features"]) == 0:
#                 cur_window += 1
#                 continue

#             z_s2, _, _, _ = zt_bursts(
#                 window_2s[cur_window : cur_window + 5],
#                 V,
#                 models,
#                 idf_2,
#                 non_empty_burst_window,
#             )
#             if len(z_s2) == 0:
#                 cur_window += 1
#                 continue

#             words_packets = np.unique(
#                 window_5s[cur_window]["behavior_features"], axis=0
#             )
#             leafs1 = models[1].apply(words_packets)
#             leafs2 = models[2].apply(z_s2)

#             z_t = np.zeros(len(V0) + len(V1), dtype=float)
#             for ix, v in enumerate(V0):
#                 if np.any(leafs1 == v):
#                     z_t[ix] = 1.0
#             for ix, v in enumerate(V1):
#                 if np.any(leafs2 == v):
#                     z_t[len(V0) + ix] = 1.0

#             label = window_5s[cur_window]["label"]
#             fv.append(z_t)
#             y.append(label)
#             cur_window += 1

#         # IDF normalisatie
#         if fv:
#             fv = np.vstack(fv)
#             for ix, df in idf_3.items():
#                 fv[:, ix] *= math.log(non_empty_behavior_window / df)
#             y = np.asarray(y)

#             for i, label in enumerate(y):
#                 active_indices = set(np.where(fv[i] > 0)[0])
#                 if label == 1:
#                     D_pos.append(active_indices)
#                 else:
#                     D_neg.append(active_indices)

#     return D_pos, D_neg


def recursive_lfm_training(
    segment_features_list,
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

    packets = {
        "packets": [],
        "labels": [],
    }

    window_2s = []
    window_5s = []

    for seg in segment_features_list:
        # packet-level
        packets["packets"].extend(seg["packet_lvl"]["packets"])
        packets["labels"].extend(seg["packet_lvl"]["labels"])

        # burst-level
        window_2s.extend(seg["burst_lvl"])

        # behavior-level
        window_5s.extend(seg["behavior_lvl"])

    V: List[List[int]] = []
    models: dict[int, DecisionTreeClassifier] = {}
    idf: dict[int, defaultdict] = {}
    non_empty_window_counts: dict[int, int] = {}

    for s in [1, 2, 3]:
        if s == 1:
            z = np.asarray(packets["packets"])
            y = np.asarray(packets["labels"])
        elif s == 2:
            z, y, idf[s], non_empty_window_counts[s] = zt_bursts(window_2s, V, models)
        else:  # s == 3
            z, y, idf[s], non_empty_window_counts[s] = zt_behavior(
                window_5s, window_2s, V, idf[2], models, non_empty_window_counts[2]
            )

        # train mapper at the current scale
        model = DecisionTreeClassifier(max_depth=7)
        if len(z) > 0:
            model.fit(z, y)
        models[s] = model
        # extract positive leaves as words for this level
        pos_leaves = extract_positive_leaves(model, z, y)  # returns list[int]
        V.append(list(pos_leaves))

    result = {
        "vocabulary": V,
        "models": models,
        "idf": idf,
        "non_empty_window_counts": non_empty_window_counts,
    }

    save_path = f"data/lfm_models/{app}.pkl"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "wb") as f:
        pickle.dump(result, f)
    return result


def convert_segments_to_feature_vectors(
    segment_features, app, saved_path="data/lfm_models/"
):

    # segment will be with its features
    saved_path = os.path.join(saved_path, f"{app}.pkl")
    with open(saved_path, "rb") as f:
        converting_info = pickle.load(f)
    V = converting_info["vocabulary"]
    models = converting_info["models"]
    idf = converting_info["idf"]
    non_empty_window_counts = converting_info["non_empty_window_counts"]

    # packets = segment_features["packet_lvl"]
    seg_fv = []
    for seg in segment_features:
        window_2s = seg["burst_lvl"]
        window_5s = seg["behavior_lvl"]

        zt = np.zeros(np.sum(len(v) for v in V))

        for i in range(len(window_5s) - 4):
            burst_input = np.empty(len(V[0]))
            fv_bursts = []
            for burst_packets in window_2s[i : i + 5]:
                # per burst door model halen
                if len(burst_packets["burst_features"]) == 0:
                    continue  # burst window was empty
                burst_input_leafs = np.vstack(
                    models[1].apply(burst_packets["burst_features"])
                )

                if len(burst_input) == 0:
                    continue  # behavior window was empty
                for ix, v in enumerate(V[0]):
                    if np.any(burst_input_leafs == v):
                        burst_input[ix] = 1 * math.log(
                            non_empty_window_counts[2] / idf[2][ix]
                        )
                fv_bursts.append(burst_input)
            leafs2 = models[2].apply(np.vstack(fv_bursts))
            # at end for -> should have 5 burst fv
            for ix, v in enumerate(V[1]):
                if np.any(leafs2 == v):
                    zt[ix + len(V[0])] = 1 * math.log(
                        non_empty_window_counts[3] / idf[3][ix]
                    )

            leafs1 = models[1].apply(window_5s[i]["behavior_features"])
            for ix, v in enumerate(V[1]):
                if np.any(leafs1 == v):
                    zt[ix] = 1 * math.log(non_empty_window_counts[3] / idf[3][ix])
            for ix, v in enumerate(V[0]):
                if np.any(leafs1 == v):
                    zt[ix] = 1 * math.log(non_empty_window_counts[3] / idf[3][ix])

        leafs3 = models[3].apply(zt[:, : len(V[0]) + len(V[1])])
        for ix, v in enumerate(V[2]):
            if np.any(leafs3 == v):
                zt[ix + len(V[0]) + len(V[1])] = 1
        seg_fv.append(zt)
    return np.vstack(seg_fv)


# make sure to include labels of the segment apps


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

        z, _, _, _ = zt_bursts(
            window_2s[cur_window : cur_window + 5],
            V,
            models,
            idf_2,
            non_empty_burst_window,
        )  # process 5 bursts at a time, use the precomputed tdf of level s=2
        if len(z) == 0:
            cur_window += 1
            continue  # even when the behavior window is not empty, there might be too little data per second to have burst features
        # no guidelines where given in this case of how to handle, so I will just decide to not add it as feature_vectors if there's 0 bursts
        #  but will add it once there's some burst.
        # Currently using a clustering treshold of 30 s since 300s merged all segments, so this might increase this occurence. However, collecting more data might alleviate the 300 s issue further possibly
        non_empty_behavior_window += 1
        words_packets = np.unique(window_5s[cur_window]["behavior_features"], axis=0)

        leafs1 = models[1].apply(
            words_packets
        )  # -> this is already z? But should not normalized with idf of s=2 (burst lvl), maar voor behavior lvl
        # ???? test if this gives the same
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
    return z, y, idf, non_empty_behavior_window


def zt_bursts(
    window_2s, V, models, idf_already_computed=None, non_empty_burst_window=None
):
    assert len(V) >= 1, "V0 must exist before level 2"
    V0 = V[0]
    idf = defaultdict(int)
    fv = []
    labels = []
    non_empty_windows = 0

    for window in window_2s:
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
                idf[ix] += 1  # per burstwindow present +1
        labels.append(window["label"])  # total label of the burst
        fv.append(z_t)  # add to the processed burst windows
    if fv:  # if not all bursts were empty
        fv = np.vstack(fv)  # make fv numpy array by stacking elements

        # if we are just using the function (like in behavior) -> use the precomputed idf
        if non_empty_burst_window is not None:
            non_empty_windows = non_empty_burst_window
        # same comment
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
