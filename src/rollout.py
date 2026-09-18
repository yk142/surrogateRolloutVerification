"""自己回帰ロールアウトと誤差指標の計算。"""
import numpy as np

from src.physics import energy, simulate


def true_rollout(initial_states: np.ndarray, dt: float, n_steps: int) -> np.ndarray:
    """shape (n_traj, n_steps+1, 2)"""
    return np.stack([simulate(s0, dt, n_steps) for s0 in initial_states], axis=0)


def rmse_curve(true_traj: np.ndarray, pred_traj: np.ndarray) -> np.ndarray:
    """horizon step ごとの (theta, theta_dot) RMSE を返す。

    true_traj, pred_traj: shape (n_traj, n_steps+1, 2) or (n_steps+1, n_traj, 2)
    どちらも time axis=0 に揃えて呼び出すこと。

    Returns: shape (n_steps+1,)
    """
    theta_err = np.arctan2(
        np.sin(true_traj[..., 0] - pred_traj[..., 0]),
        np.cos(true_traj[..., 0] - pred_traj[..., 0]),
    )
    theta_dot_err = true_traj[..., 1] - pred_traj[..., 1]
    sq_err = theta_err**2 + theta_dot_err**2
    # sq_err: (n_steps+1, n_traj) -> mean over trajectories then sqrt
    return np.sqrt(sq_err.mean(axis=1))


def energy_deviation(pred_traj: np.ndarray) -> np.ndarray:
    """予測軌道のエネルギー時系列を返す。shape (n_steps+1, n_traj)"""
    return energy(pred_traj)
