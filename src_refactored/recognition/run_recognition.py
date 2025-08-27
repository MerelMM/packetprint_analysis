from recognition.lfm import recursive_lfm_training, raw_features_to_lfm
from recognition.extract_raw_features import extract_raw_features_segment


def run_recognition(
    app_key, segments=None, load_precomputed_features=False, pretrained_lfm=False
):
    raw_segment_features = extract_raw_features_segment(
        app_key, segments, load_features=load_precomputed_features
    )

    # training the model
    if not pretrained_lfm:
        recursive_lfm_training(app_key, raw_segment_features)
    else:
        lfm_fvs = raw_features_to_lfm(app_key, raw_segment_features)
    lfm_fvs

    # create d+ and d_ for training greedy: so

    # if already trained: raw_features_to_lfm
