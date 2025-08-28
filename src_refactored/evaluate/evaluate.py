from segmentation.run_segmentation import run_segmentation
from preprocessing.concat_traces import concat_traces
from recognition.run_recognition import run_recognition_wrapper
from preprocessing.concat_traces import concat_traces
from collections import Counter


def evaluate(app_key, threshold=0.1):
    # Step 1: Preprocess test data and build ground-truth segments
    concatenated_test_data = concat_traces(
        app_key,
        data_path="capture_data_test",
        save_path="data/concatenated_test_trace.pkl",
        load_existing=False,
        load_existing_filter=True,
    )

    _concatenated_sizes, concatenated_timings, concatenated_labels = (
        concatenated_test_data
    )

    # Step 2: Extract ground-truth segments (start_time, end_time)
    segment_times = []
    ix = 0
    while ix < len(concatenated_labels):
        if concatenated_labels[ix] == app_key:
            start = concatenated_timings[ix]
            while ix < len(concatenated_labels) and concatenated_labels[ix] == app_key:
                ix += 1
            end = concatenated_timings[ix - 1]
            segment_times.append((start, end))
        else:
            ix += 1

    # Step 3: Run segmentation
    proposal_segments = run_segmentation(
        app_key,
        concatenated_data=concatenated_test_data,
        load_precomputed=False,
        load_pretrained=True,
    )

    # Step 4: Extract center timestamp of each proposed segment
    reference_times_proposed_segments = [
        seg["timestamps"][0] + (seg["timestamps"][-1] - seg["timestamps"][0]) / 2
        for seg in proposal_segments
    ]

    # Step 5: Run recognition to get scores and skipped segments
    predictions, reference_times_proposed_segments = run_recognition_wrapper(
        app_key,
        proposal_segments,
        training=False,
        c_similarity_threshold=threshold,
        seg_timings=reference_times_proposed_segments,
    )

    # Step 6: Analyze predictions
    true_positives = 0
    false_positives = 0
    matched_gt = set()
    match_counts = Counter()

    for pred_time, prediction in zip(reference_times_proposed_segments, predictions):
        if prediction == 1:
            matched = False
            for i, (start, end) in enumerate(segment_times):
                if start <= pred_time <= end:
                    match_counts[i] += 1  # count even if matched before
                    if i not in matched_gt:
                        matched_gt.add(i)
                        true_positives += 1
                    matched = True
                    break
            if not matched:
                false_positives += 1

    false_negatives = len(segment_times) - len(matched_gt)

    # Final reporting
    print(f"Detected correctly (true positives): {true_positives}")
    print(f"Missed segments (false negatives): {false_negatives}")
    print(f"Wrongfully detected (false positives): {false_positives}")
