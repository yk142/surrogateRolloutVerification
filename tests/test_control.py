import numpy as np

from src.control import make_true_rollout_fn, run_mpc


def test_oracle_mpc_swings_up_and_stabilizes_near_inverted():
    """オラクル(真の物理モデル)MPCが最下点静止から倒立近傍へ到達・安定化すること。"""
    rollout_fn = make_true_rollout_fn(c=0.15)
    traj, _ = run_mpc(rollout_fn, np.array([0.0, 0.0]), n_steps=400, c=0.15, seed=0)

    final_theta = traj[-50:, 0]
    final_theta_dot = traj[-50:, 1]

    angular_distance = np.abs(np.arctan2(np.sin(final_theta - np.pi), np.cos(final_theta - np.pi)))
    assert angular_distance.mean() < 0.1
    assert np.abs(final_theta_dot).mean() < 0.1
