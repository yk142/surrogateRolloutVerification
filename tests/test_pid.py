import numpy as np

from src.pid import angular_error, angular_error_to_inverted, run_pid_true


def test_pid_stabilizes_true_plant_near_inverted():
    """真の物理モデルに対し、倒立近傍からPID制御で安定化できること。"""
    initial_state = np.array([np.pi - 0.3, 0.0])
    traj, _ = run_pid_true(initial_state, n_steps=400, c=0.15)

    final_error = np.abs(angular_error_to_inverted(traj[-50:, 0]))
    final_theta_dot = np.abs(traj[-50:, 1])

    assert final_error.mean() < 0.05
    assert final_theta_dot.mean() < 0.1


def test_pid_ptp_tracks_non_inverted_target_with_gravity_feedforward():
    """重力フィードフォワード込みのPIDが、倒立以外の目標角度にも追従できること。

    フィードフォワードなしの単純PIDでは、目標角度が倒立(pi)からズレると
    定常的な重力トルクを補償できず倒立から転落してしまう(修正前の既知バグ)。
    """
    target = np.pi + 0.3
    traj, _ = run_pid_true(np.array([np.pi, 0.0]), n_steps=150, c=0.15, target=target)

    final_error = np.abs(angular_error(traj[-30:, 0], target)).mean()
    assert final_error < 0.1
