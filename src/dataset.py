"""学習・評価用の軌道データセット生成。

初期条件は theta in [-pi, pi] (倒立近傍含む), theta_dot を広めのレンジでサンプリングする。
"""
import numpy as np

from src.physics import simulate

THETA_RANGE = (-np.pi, np.pi)
THETA_DOT_RANGE = (-6.0, 6.0)


def sample_initial_states(n: int, rng: np.random.Generator) -> np.ndarray:
    theta = rng.uniform(*THETA_RANGE, size=n)
    theta_dot = rng.uniform(*THETA_DOT_RANGE, size=n)
    return np.stack([theta, theta_dot], axis=-1)


def generate_trajectories(
    n_trajectories: int, dt: float, n_steps: int, seed: int
) -> np.ndarray:
    """shape (n_trajectories, n_steps + 1, 2) の軌道データを返す。"""
    rng = np.random.default_rng(seed)
    initial_states = sample_initial_states(n_trajectories, rng)
    trajectories = np.stack(
        [simulate(s0, dt, n_steps) for s0 in initial_states], axis=0
    )
    return trajectories


def make_transition_pairs(trajectories: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """軌道群から1-step遷移ペア (x_t, x_{t+1}) を作る。

    trajectories: shape (n_traj, n_steps + 1, 2)
    """
    x = trajectories[:, :-1, :].reshape(-1, 2)
    x_next = trajectories[:, 1:, :].reshape(-1, 2)
    return x, x_next
