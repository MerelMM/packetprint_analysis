def hac_clustering(
    anchor_timings,
    epsilon=300,
):
    """
    HAC for 1D time data with single linkage
    Groups anchor packets into clusters separated by gaps > epsilon.
    Complexity: O(n) instead of O(n²).
    """
    ts_sorted = sorted(anchor_timings)
    if not anchor_timings:
        return []
    clusters = []
    current_cluster = [ts_sorted[0]]

    for t in ts_sorted[1:]:
        if t - current_cluster[-1] <= epsilon:
            current_cluster.append(t)
        else:
            clusters.append(current_cluster)
            current_cluster = [t]
    clusters.append(current_cluster)  # last cluster

    # Convert clusters to target segments
    return [
        {
            "cluster_id": i + 1,
            "start_time": float(min(c)),
            "end_time": float(max(c)),
            "anchor_count": len(c),
        }
        for i, c in enumerate(clusters)
    ]
