import pytest

from examples.libero.goal_long_eval import goal_metrics


def test_separately_achieved_subgoals_are_not_full_success():
    out = goal_metrics([[True, False], [False, True]])
    assert out["goal_ever"] == [True, True]
    assert not out["success"]
    assert out["max_simultaneous_goals"] == 1


def test_conjunction_success_and_single_goal():
    assert goal_metrics([[False, True], [True, True]])["success"]
    assert goal_metrics([[False], [True]])["success"]
    assert not goal_metrics([[False], [False]])["success"]
    with pytest.raises(ValueError, match="Expected nonempty"):
        goal_metrics([])
