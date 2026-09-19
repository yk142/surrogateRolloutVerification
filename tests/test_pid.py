import numpy as np

from src.pid import angular_error_to_inverted, run_pid_true


def test_pid_stabilizes_true_plant_near_inverted():
    """真の物理モデルに対し、倒立近傍からPID制御で安定化できること。"""
    initial_state = np.array([np.pi - 0.3, 0.0])
    traj, _ = run_pid_true(initial_state, n_steps=400, c=0.15)

    final_error = np.abs(angular_error_to_inverted(traj[-50:, 0]))
    final_theta_dot = np.abs(traj[-50:, 1])

    assert final_error.mean() < 0.05
    assert final_theta_dot.mean() < 0.1
