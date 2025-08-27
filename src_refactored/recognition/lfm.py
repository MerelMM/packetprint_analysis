from typing import TypedDict, DefaultDict, List, Tuple, Dict, Any
import numpy as np
from sklearn.tree import DecisionTreeClassifier
import math
import pickle
import numpy as np
from collections import defaultdict
import os


class LFMTrainingResult(TypedDict):
    vocabulary: List[List[int]]
    models: Dict[int, DecisionTreeClassifier]
    idf: Dict[int, DefaultDict[int, float]]
    non_empty_window_counts: Dict[int, int]


def recursive_lfm_training(
    app,
    segment_features_list,
) -> LFMTrainingResult:
    """
    Recursive LFM Training as in Algorithm 2.

    Returns:
    A dictionary with:
        - 'vocabulary': List of leaf vocabularies per level [V0, V1, V2]
        - 'models': dict of trained DecisionTreeClassifiers {1: M1, 2: M2, 3: M3}
        - 'idf': dict of IDF maps per level {2: ..., 3: ...}
        - 'non_empty_window_counts': dict of counts {2: ..., 3: ...}
    """

    packets = {
        "packets": [],
        "labels": [],
    }

    window_2s = []
    window_5s = []

    for seg in segment_features_list:
        # packet-level
        if (
            len(seg["packet_lvl"]["packets"]) < 10
        ):  # not in the paper, but too short otherwise, a bit randomly chosen, but so are other things
            print("should not be here anymore, should already be filtered?")
            continue
        packets["packets"].extend(seg["packet_lvl"]["packets"])
        packets["labels"].extend(seg["packet_lvl"]["labels"])

        # burst-level
        window_2s.extend(seg["burst_lvl"])

        # behavior-level
        window_5s.extend(seg["behavior_lvl"])

    V: List[List[int]] = []
    models: Dict[int, DecisionTreeClassifier] = {}
    idf: Dict[int, DefaultDict[int, float]] = {}
    non_empty_window_counts: dict[int, int] = {}

    for s in [1, 2, 3]:
        if s == 1:
            z = np.asarray(packets["packets"])
            y = np.asarray(packets["labels"])
        elif s == 2:
            z, y, idf[s], non_empty_window_counts[s] = _zt_bursts(window_2s, V, models)
        else:  # s == 3
            z, y, idf[s], non_empty_window_counts[s] = _zt_behavior(
                window_5s, window_2s, V, idf[2], models, non_empty_window_counts[2]
            )

        # train mapper at the current scale
        model = DecisionTreeClassifier(max_depth=7)
        if len(z) > 0:
            model.fit(z, y)
        models[s] = model
        # extract positive leaves as words for this level
        pos_leaves = _extract_positive_leaves(model, z, y)  # returns list[int]
        V.append(list(pos_leaves))

    result: LFMTrainingResult = {
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


def raw_features_to_lfm(app, segment_features, saved_path="data/lfm_models/"):

    # segment will be with its features
    saved_path = os.path.join(saved_path, f"{app}.pkl")
    with open(saved_path, "rb") as f:
        converting_info = pickle.load(f)

    V = converting_info["vocabulary"]
    assert len(V) == 3, "Expected 3 vocabulary levels: packet, burst, behavior"

    models = converting_info[
        "models"
    ]  # a bit strange, but dict with 1, 2 and 3 as keys
    assert len(models) == 3, "Expected 3 tree models"

    idf = converting_info["idf"]  # a bit strange, but dict with 2 and 3 as keys
    assert (
        len(idf) == 2
    ), "Expected 2 idfs (one for input computing burst fvs, one for behavior fvs)"

    non_empty_window_counts = converting_info[
        "non_empty_window_counts"
    ]  # a bit strange, but dict with 2 and 3 as keys
    assert (
        len(idf) == 2
    ), "Expected 2 non_empty_window_counts (one for input computing burst fvs, one for behavior fvs)"

    # packets = segment_features["packet_lvl"]
    seg_fv = []
    seg_labels = []

    for seg in segment_features:
        label = 0

        # extract all burst raw features and behavior raw features of the segment
        window_2s = seg["burst_lvl"]
        window_5s = seg["behavior_lvl"]

        # initalize the segment fv
        zt = np.zeros(np.sum(len(v) for v in V))

        # initialize behavior level loop
        fv_behavior = []  # collects the behavior fvs in the segment

        for i in range(len(window_5s) - 4):

            # intiialize behavioral fv as zero vector
            behavior_input = np.zeros(len(V[0]) + len(V[1]))

            # initialize burst level loop
            fv_bursts = []  # collects the burst fvs in the behavior window
            for burst_packets in window_2s[i : i + 5]:

                # initalize burst fv as zero vector
                burst_input = np.zeros(len(V[0]))

                if len(burst_packets["burst_features"]) == 0:
                    continue  # empty burst window

                packet_lvl_leafs_bursts = set(
                    models[1].apply(burst_packets["burst_features"])
                )

                for ix, v in enumerate(V[0]):
                    if v in packet_lvl_leafs_bursts:
                        burst_input[ix] = 1 * math.log(
                            non_empty_window_counts[2] / max(idf[2][ix], 1)
                        )  # burst input -> needs to be idf'd with the idf of bursts (on which the model M2 was trained)

                # collect the burst feature input in the behavioral window
                fv_bursts.append(burst_input)

            if len(fv_bursts) == 0:
                continue  # the behavior window was empty -> go to the next window

            fv_bursts = np.vstack(fv_bursts)
            # add the packet leafs of the behavior window -- keep track of which are in the whole segment
            burst_lvl_leafs = set(models[2].apply(fv_bursts))

            # add which burst features words (V1) are present in the behavioral window
            for ix, v in enumerate(V[1]):
                if v in burst_lvl_leafs:
                    behavior_input[ix + len(V[0])] = 1 * math.log(
                        non_empty_window_counts[3] / max(idf[3][ix + len(V[0])], 1)
                    )

            # if any of the burst windows has the packet feature, the packet feature is present in the behavioral window
            # weight with idf of
            for col in range(len(V[0])):
                if np.any(fv_bursts[:, col] != 0):
                    behavior_input[col] = math.log(
                        non_empty_window_counts[3] / max(idf[3][col], 1)
                    )

            # collect the behavioral feature input of the segment
            fv_behavior.append(behavior_input)
            label = max(label, window_5s[i]["label"])

        if len(fv_behavior) == 0:
            print(seg)
            continue
        # if any of the behavior features have the word, the word is in the segment
        fv_behavior = np.vstack(fv_behavior)
        for col in range(len(V[0]) + len(V[1])):
            if np.any(fv_behavior[:, col] != 0):
                zt[col] = 1

        # also add which behavior feature words are there
        behavior_lvl_leafs = set(models[3].apply(fv_behavior))
        for ix, v in enumerate(V[2]):
            if v in behavior_lvl_leafs:
                zt[ix + len(V[0]) + len(V[1])] = 1

        # lfm feature of segment completed
        seg_fv.append(zt)
        seg_labels.append(label)

    return np.vstack(seg_fv), np.vstack(seg_labels)


def _zt_behavior(window_5s, window_2s, V, idf_2, models, non_empty_burst_window):
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

        z, _, _, _ = _zt_bursts(
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

        # initialize behavior window fv with zero-vector
        z_t = np.zeros(len(V0) + len(V1), dtype=float)

        # if packet level words present, value is 1
        for ix, v in enumerate(V0):
            if np.any(leafs1 == v):
                z_t[ix] = 1.0
                idf[ix] += 1
        # if burst level words present, value is 1
        for ix, v in enumerate(V1):
            if np.any(leafs2 == v):
                z_t[len(V0) + ix] = 1.0
                idf[len(V0) + ix] += 1

        labels.append(window_5s[cur_window]["label"])

        # collect all behavior input vectors to train model
        fv.append(z_t)
        cur_window += 1

    # based on the idfs, reweight the behavior input fvs
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

    # return the behavioral fvs with labels to train model as well as idf and non_empty_behavior_window number later for during inference time
    return z, y, idf, non_empty_behavior_window


def _zt_bursts(
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


def _extract_positive_leaves(model, z, y, min_pos_fraction=0.5):
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
