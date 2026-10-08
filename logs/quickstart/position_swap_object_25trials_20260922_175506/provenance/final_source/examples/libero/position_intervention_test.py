from types import SimpleNamespace

import numpy as np
import pytest

from examples.libero.position_intervention import apply_layout
from examples.libero.position_intervention import validate_settled
from examples.libero.summarize_position_ablation import demo_targets


class FakeEnv:
    def __init__(self):
        self.qpos = np.array(
            [8.0, 9.0, -0.12, -0.24, 0.1, 1, 0, 0, 0, 0.05, -0.1, 0.2, 0.7, 0, 0.7, 0, 2, 3, 4, 1, 0, 0, 0]
        )
        self.qvel = np.ones(20)
        self.address = {"milk_1": (2, 9), "tomato_sauce_1": (9, 16), "basket_1": (16, 23)}
        self.velocity_address = {"milk_1": (2, 8), "tomato_sauce_1": (8, 14), "basket_1": (14, 20)}
        self.env = SimpleNamespace(objects_dict={name: SimpleNamespace(joints=[name]) for name in self.address})
        self.sim = SimpleNamespace(
            model=SimpleNamespace(get_joint_qpos_addr=self.address.get, get_joint_qvel_addr=self.velocity_address.get),
            data=SimpleNamespace(
                qpos=self.qpos,
                qvel=self.qvel,
                ncon=0,
                get_joint_qpos=lambda name: self.qpos[slice(*self.address[name])],
                set_joint_qpos=lambda name, value: self.qpos.__setitem__(slice(*self.address[name]), value),
                get_joint_qvel=lambda name: self.qvel[slice(*self.velocity_address[name])],
                set_joint_qvel=lambda name, value: self.qvel.__setitem__(slice(*self.velocity_address[name]), value),
            ),
        )

    def get_sim_state(self):
        return np.concatenate([self.qpos, self.qvel])

    def regenerate_obs_from_state(self, state):
        np.testing.assert_array_equal(state, self.get_sim_state())
        return {"fresh_qpos": self.qpos.copy()}


def test_swap_changes_only_xy_and_target_velocities_and_refreshes_observation():
    env = FakeEnv()
    before = env.qpos.copy()
    obs, meta = apply_layout(env, "swap_milk_tomato")
    expected = before.copy()
    expected[2:4], expected[9:11] = before[9:11], before[2:4]
    np.testing.assert_array_equal(env.qpos, expected)
    np.testing.assert_array_equal(obs["fresh_qpos"], expected)
    np.testing.assert_array_equal(env.qvel[2:14], 0)
    np.testing.assert_array_equal(env.qvel[:2], 1)
    np.testing.assert_array_equal(env.qvel[14:], 1)
    assert meta["before_poses"]["milk_1"][:2] != meta["intervened_poses"]["milk_1"][:2]
    apply_layout(env, "swap_milk_tomato")
    np.testing.assert_array_equal(env.qpos, before)


def test_original_preserves_pose_and_unknown_layout_fails_before_mutation():
    env = FakeEnv()
    before = env.qpos.copy()
    apply_layout(env, "original")
    np.testing.assert_array_equal(env.qpos, before)
    state = env.get_sim_state().copy()
    with pytest.raises(ValueError, match="Unknown layout"):
        apply_layout(env, "invalid")
    np.testing.assert_array_equal(env.get_sim_state(), state)


def test_unstable_layout_is_rejected():
    env = FakeEnv()
    _, meta = apply_layout(env, "original")
    env.qpos[2] += 0.04
    with pytest.raises(ValueError, match="Unstable"):
        validate_settled(env, meta)


@pytest.mark.parametrize("language", ["milk", "tomato_sauce"])
@pytest.mark.parametrize("mode", ["correct", "wrong"])
def test_swapping_separates_demo_identity_from_original_position(language, mode):
    identity, original_occupant = demo_targets(language, mode, "original")
    same_identity, swapped_occupant = demo_targets(language, mode, "swap_milk_tomato")
    assert identity == original_occupant == same_identity
    assert swapped_occupant != identity
    assert demo_targets(language, "no", "swap_milk_tomato") == (None, None)
