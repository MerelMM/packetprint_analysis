# This script was developed with assistance from ChatGPT (OpenAI) and Github Copilot
# Final implementation and adaptation by Merel Haenaets.

from collections import defaultdict
import pytest


def test_packet_rate_estimation():
    # Simulated packet sizes per app
    all_packets = {
        "appA": [1, 1, 1, 1, 1],  # r(1) = 5/5 = 1
        "appB": [1, 1, 2, 2],  # r(1) = 1/2, r(2) = 1/2
        "appC": [3, 2],  # r(3) = 1/2 = r(2)
        "appD": [4, 4, 4, 4, 4],  # r(4) = 5/5 = 1
        # app A = r(1) = 1, rest = 0
        # r_(1) = (1/2)/3 = 1/6, r_(2) =(1/2 + 1/2)/3 = 1/3, r(3) = 1/6, r(4) = 1/3
    }

    # Simulated durations in seconds
    durations = {
        "appA": {"duration": 5.0},
        "appB": {"duration": 4.0},
        "appC": {"duration": 2.0},
        "appD": {"duration": 5.0},
    }

    app_key = "appA"

    # --- Build Sp ---
    Sp = set()
    for packet_list in all_packets.values():
        Sp.update(packet_list)
    assert Sp == {1, 2, 3, 4}, "Sp should include all unique packet sizes"

    # --- r_pos ---
    packet_sizes = all_packets[app_key]
    duration = durations[app_key]["duration"]
    r_pos = defaultdict(float)
    for size in packet_sizes:
        r_pos[size] += 1
    for size in r_pos:
        r_pos[size] /= duration

    # Should only contain packet size 1
    assert set(r_pos.keys()) == {1}
    assert r_pos[1] == 5 / 5.0  # 1.0

    # --- r_neg ---
    r_neg_sum = defaultdict(float)

    for other_app, other_sizes in all_packets.items():
        if other_app == app_key:
            continue
        other_duration = durations[other_app]["duration"]
        counts = defaultdict(int)
        for s in other_sizes:
            counts[s] += 1
        for s, c in counts.items():
            r_neg_sum[s] += c / other_duration

    num_other_apps = len(all_packets) - 1
    r_neg = {s: r_neg_sum[s] / num_other_apps for s in r_neg_sum}

    # Expected: r_(1) = (1/2)/3 = 1/6, r_(2) =(1/2 + 1/2)/3 = 1/3, r(3) = 1/6, r(4) = 1/3
    assert r_neg[1] == pytest.approx(1 / 6)
    assert r_neg[2] == pytest.approx(1 / 3)
    assert r_neg[3] == pytest.approx(1 / 6)
    assert r_neg[4] == pytest.approx(1 / 3)

    print("test_packet_rate_estimation passed.")


# Run directly
if __name__ == "__main__":
    test_packet_rate_estimation()
