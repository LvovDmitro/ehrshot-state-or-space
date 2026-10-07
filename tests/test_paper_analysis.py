import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "paper" / "analysis"))
from characterize_selected_episodes import top_ids, RAW, ERA
from matched_stability_and_shift_profiles import matched_size_stability


def predictions():
    return pd.DataFrame([{"task": task, "compression_version": version, "seed": seed,
        "row_id": row, "subject_id": row // 2, "y_true": row % 2,
        "risk_calibrated": row / 100} for task in ("guo_readmission", "guo_icu")
        for version in (RAW, ERA) for seed in range(42, 47) for row in range(21)])


def test_capacity_ceil_and_tie_order():
    frame = pd.DataFrame({"row_id": [3, 1, 2], "risk_calibrated": [0.5] * 3})
    assert top_ids(frame) == {1}
    assert top_ids(pd.DataFrame({"row_id": range(21), "risk_calibrated": [0.5] * 21})) == {0, 1, 2}


def test_matched_pairs_have_expected_sizes_and_no_overlap():
    result = matched_size_stability(predictions())
    assert len(result) == 12
    assert np.allclose(result.mean_jaccard, 1)
    assert set(result[result.ensemble_size == 1].n_pairs) == {5, 10}
    assert set(result[result.ensemble_size == 2].n_pairs) == {10, 15}


def test_missing_episode_fails_closed():
    frame = predictions()
    with pytest.raises(ValueError, match="complete"):
        matched_size_stability(frame.drop(index=0))


def test_duplicate_prediction_fails_closed():
    frame = predictions()
    with pytest.raises(ValueError, match="Duplicate"):
        matched_size_stability(pd.concat([frame, frame.iloc[:1]]))


def test_label_mismatch_fails_closed():
    frame = predictions()
    frame.loc[0, "y_true"] = 1
    with pytest.raises(ValueError, match="identities"):
        matched_size_stability(frame)


def test_nonfinite_risk_fails_closed():
    frame = predictions()
    frame.loc[0, "risk_calibrated"] = np.inf
    with pytest.raises(ValueError, match="probability"):
        matched_size_stability(frame)


def test_wrong_seed_set_fails_closed():
    frame = predictions()
    frame.loc[frame.seed == 42, "seed"] = 47
    with pytest.raises(ValueError, match="complete"):
        matched_size_stability(frame)
