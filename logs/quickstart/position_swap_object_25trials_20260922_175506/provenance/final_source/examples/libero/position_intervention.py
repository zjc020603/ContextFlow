"""Runtime position interventions, applied after restoring LIBERO initial states."""

import hashlib

import numpy as np

LAYOUTS = ("original", "swap_milk_tomato")
OBJECTS = ("milk_1", "tomato_sauce_1")


def state_hash(state):
    return hashlib.sha256(np.asarray(state).tobytes()).hexdigest()


def object_poses(env):
    return {name: env.sim.data.get_joint_qpos(obj.joints[-1]).copy() for name, obj in env.env.objects_dict.items()}


def apply_layout(env, layout):
    """Swap only XY, preserve each object's height/orientation, refresh observations."""
    if layout not in LAYOUTS:
        raise ValueError(f"Unknown layout: {layout}")
    source_state = env.get_sim_state().copy()
    before = object_poses(env)
    expected_qpos = env.sim.data.qpos.copy()
    expected_qvel = env.sim.data.qvel.copy()
    poses = [before[name].copy() for name in OBJECTS]
    if layout == "swap_milk_tomato":
        poses[0][:2], poses[1][:2] = poses[1][:2].copy(), poses[0][:2].copy()
    for index, name in enumerate(OBJECTS):
        pose = poses[index]
        joint = env.env.objects_dict[name].joints[-1]
        qstart, qend = env.sim.model.get_joint_qpos_addr(joint)
        vstart, vend = env.sim.model.get_joint_qvel_addr(joint)
        if qend - qstart != 7 or vend - vstart != 6:
            raise ValueError(f"Expected free joint for {name}")
        env.sim.data.set_joint_qpos(joint, pose)
        env.sim.data.set_joint_qvel(joint, np.zeros(6))
        expected_qpos[qstart:qend] = pose
        expected_qvel[vstart:vend] = 0
    # Robot, other objects, and all non-target velocities must be untouched.
    np.testing.assert_array_equal(env.sim.data.qpos, expected_qpos)
    np.testing.assert_array_equal(env.sim.data.qvel, expected_qvel)
    observation = env.regenerate_obs_from_state(env.get_sim_state().copy())
    metadata = {
        "layout": layout,
        "source_state_sha256": state_hash(source_state),
        "before_poses": {name: pose.tolist() for name, pose in before.items()},
        "intervened_poses": {name: pose.tolist() for name, pose in object_poses(env).items()},
        "operation": "swap XY only; keep own Z/quaternion; zero both target velocities in both layouts",
    }
    return observation, metadata


def validate_settled(env, metadata):
    """Reject unstable initializations instead of silently replacing trial indices."""
    settled = object_poses(env)
    measurements = {}
    for name in OBJECTS:
        expected = np.asarray(metadata["intervened_poses"][name])
        actual = settled[name]
        xy_drift = float(np.linalg.norm(actual[:2] - expected[:2]))
        z_drift = float(abs(actual[2] - expected[2]))
        quat_similarity = float(abs(np.dot(actual[3:], expected[3:])))
        if not np.isfinite(actual).all() or xy_drift > 0.03 or z_drift > 0.08 or quat_similarity < 0.94:
            raise ValueError(f"Unstable initialization: {name}, XY={xy_drift}, Z={z_drift}, quat={quat_similarity}")
        joint = env.env.objects_dict[name].joints[-1]
        velocity = env.sim.data.get_joint_qvel(joint).copy()
        if np.linalg.norm(velocity[:3]) > 0.05 or np.linalg.norm(velocity[3:]) > 0.5:
            raise ValueError(f"Object has not settled: {name}, velocity={velocity}")
        measurements[name] = {
            "velocity": velocity.tolist(),
            "xy_drift_m": xy_drift,
            "z_drift_m": z_drift,
            "quaternion_similarity": quat_similarity,
        }
    # Small resting support contacts are expected. Deep penetration is not.
    min_distance = min((float(env.sim.data.contact[i].dist) for i in range(env.sim.data.ncon)), default=0.0)
    if min_distance < -0.005:
        raise ValueError(f"Deep initial contact penetration: {min_distance} m")
    metadata.update(
        settled_poses={name: pose.tolist() for name, pose in settled.items()},
        stability=measurements,
        minimum_contact_distance_m=min_distance,
        settled_state_sha256=state_hash(env.get_sim_state()),
    )
    return metadata
