from recognition.lfm import recursive_lfm_training, raw_features_to_lfm
from recognition.extract_raw_features import extract_raw_features_segment
from recognition.greedy_feature_representation import (
    train_greedy_feature_representation,
    create_D_pos_and_neg_direct_from_lfm,
    compress_segment_fv,
)
from recognition.c_similarity import train_c_similarity_classifier, compute_c_similarity


def run_recognition(
    app_key,
    segments=None,
    load_precomputed_features=False,
    pretrained_lfm=False,
    pretrained_compression=False,
    pretrained_classifier=False,
):
    # Step 1: Raw feature extraction
    raw_segment_features = extract_raw_features_segment(
        app_key, segments, load_features=load_precomputed_features
    )

    # Step 2: LFM training (optional)
    if not pretrained_lfm:
        recursive_lfm_training(app_key, raw_segment_features)

    # Step 3: Convert raw -> LFM features + labels
    lfm_fvs, lfm_labels = raw_features_to_lfm(app_key, raw_segment_features)

    # Step 4: Compression training (optional)
    if not pretrained_compression:
        D_pos, D_neg = create_D_pos_and_neg_direct_from_lfm(lfm_fvs, lfm_labels)
        _ = train_greedy_feature_representation(app_key, D_pos, D_neg)

    # Step 5: Apply compression to get final feature vectors
    fvs = compress_segment_fv(app_key, lfm_fvs)

    # Step 6: Train classifier (optional)
    if not pretrained_classifier:
        train_c_similarity_classifier(app_key, fvs, lfm_labels)

    # Step 7: Compute final similarity scores
    result = compute_c_similarity(app_key, fvs)

    return result


def run_recognition_wrapper(
    app_key: str,
    segments,
    training: bool = True,
    load_precomputed_features: bool = False,
):
    return run_recognition(
        app_key,
        segments,
        load_precomputed_features=load_precomputed_features,  # because this will need to be computed when testing and one time when trainign
        pretrained_lfm=not training,
        pretrained_compression=not training,
        pretrained_classifier=not training,
    )
    # create d+ and d_ for training greedy: so

    # if already trained: raw_features_to_lfm
