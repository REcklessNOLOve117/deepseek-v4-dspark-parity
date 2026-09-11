from dsv4_parity.alignment import align_capture_rows, select_replay_positions, shared_prefix


def test_shared_prefix_identical():
    result = shared_prefix([1, 2, 3], [1, 2, 3])
    assert result.shared_length == 3
    assert result.first_divergence is None
    assert result.termination == "identical"


def test_shared_prefix_first_and_middle_divergence():
    assert shared_prefix([1], [2]).first_divergence == 0
    result = shared_prefix([1, 2, 3], [1, 2, 9])
    assert result.shared_length == 2
    assert result.first_divergence == 2


def test_shared_prefix_eos_or_truncation_boundary():
    result = shared_prefix([1, 2], [1, 2, 3])
    assert result.first_divergence == 2
    assert result.termination == "length_or_eos"


def test_capture_alignment_uses_position_and_history_not_ordinal():
    left = [
        {"prediction_position": 10, "history_sha256": "a"},
        {"prediction_position": 11, "history_sha256": "b"},
    ]
    right = [
        {"prediction_position": 11, "history_sha256": "wrong", "branch_status": "rejected"},
        {"prediction_position": 10, "history_sha256": "a"},
        {"prediction_position": 11, "history_sha256": "b"},
    ]
    aligned = align_capture_rows(left, right)
    assert [(a["prediction_position"], b["prediction_position"]) for a, b in aligned] == [(10, 10), (11, 11)]


def test_replay_windows_record_short_and_duplicate_cases():
    assert select_replay_positions(10, 4, None)[0]["reason"] == "output_shorter_than_window"
    rows = select_replay_positions(10, 8, None)
    assert rows[0]["status"] == "selected"
    assert rows[1]["reason"] == "duplicate_window"

