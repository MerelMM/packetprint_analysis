def identify_anchor_packets(
    serialized_data,
    scores,
    anchor_threshold=0.95,
):
    _, timings, _ = serialized_data
    ixs = [i for i, score in enumerate(scores) if score >= anchor_threshold]
    anchor_timings = [timings[i] for i in ixs]

    return anchor_timings
