import numpy as np

from src.diagnose_amplitude_sweep import THETA0, THETA_DOT0_RANGE
from src.pid import angular_error_to_inverted, run_pid_true


def test_true_physics_fails_in_midrange_velocity():
    """真の物理モデルは、この振幅スイープの中間域(トルク飽和で復帰できない)で
    実際に失敗すること(#25で見つかった、真値自身の失敗領域があるという前提を固定)。
    """
    ic = np.array([THETA0, -1.6])  # スイープ中間域の代表点
    traj, _ = run_pid_true(ic, n_steps=400, c=0.15)
    final_error = np.abs(angular_error_to_inverted(traj[-50:, 0])).mean()
    assert final_error > 1.0
