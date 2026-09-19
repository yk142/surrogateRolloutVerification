"""ランダムシューティング(サンプリングベース)MPCによるスイングアップ制御。

各制御ステップで、ホライズン H のランダムトルク列候補を多数サンプリングし、
「予測モデル」(NSSサロゲート または 真の物理モデル)でロールアウトして
コスト最小の候補の先頭トルクだけを実プラントに適用する(receding horizon)。
実プラント側は常に真の物理モデルでステップし、予測モデルだけを差し替えることで
サロゲートの制御用途での実用性を検証する。
"""
from typing import Callable

import numpy as np

from src.model import AutoregressiveModel
from src.physics import G, L, rk4_step, simulate

DT = 0.02
TAU_RANGE = (-4.0, 4.0)
HORIZON = 60  # 1.2秒先まで予測(スイングアップに必要な往復を1プランで捉えられる程度)
N_CANDIDATES = 300
W_TERMINAL_POSITION = 5.0  # ホライズン終端で倒立に近いことを重視
W_TERMINAL_VELOCITY = 0.1  # ホライズン終端で静止していることを重視(安定化)
W_CONTROL = 0.01

OMEGA_NATURAL = np.sqrt(G / L)
OMEGA_RANGE = (0.3 * OMEGA_NATURAL, 1.3 * OMEGA_NATURAL)

RolloutFn = Callable[[np.ndarray, np.ndarray], np.ndarray]


def make_true_rollout_fn(c: float, m: float = 1.0) -> RolloutFn:
    """真の物理モデルで予測するロールアウト関数(オラクルMPC用)。"""

    def fn(state_batch: np.ndarray, tau_seq: np.ndarray) -> np.ndarray:
        return simulate(state_batch, DT, tau_seq.shape[0], c=c, tau=tau_seq, m=m)

    return fn


def make_nss_rollout_fn(model: AutoregressiveModel) -> RolloutFn:
    """NSSサロゲートで予測するロールアウト関数。"""

    def fn(state_batch: np.ndarray, tau_seq: np.ndarray) -> np.ndarray:
        return model.rollout(state_batch, tau_seq.shape[0], tau_seq=tau_seq)

    return fn


def position_cost(state: np.ndarray) -> np.ndarray:
    """倒立(theta=pi)からの角度的な乖離コスト。state: shape (..., 2)"""
    theta = state[..., 0]
    return 1.0 - np.cos(theta - np.pi)


def total_cost(traj: np.ndarray, tau_seq: np.ndarray) -> np.ndarray:
    """traj: (H+1, n_candidates, 2), tau_seq: (H, n_candidates) -> (n_candidates,)

    途中経過は角度コストのみ(速度に罰則を課すとスイングアップに必要な加速自体が
    抑制されてしまうため)。ホライズン終端では位置・速度の両方を重く評価し、
    「倒立に到達して静止する」ことを促す(receding horizonで実質的な安定化制御になる)。
    """
    running = position_cost(traj[1:-1]).sum(axis=0)
    terminal_theta, terminal_theta_dot = traj[-1, ..., 0], traj[-1, ..., 1]
    terminal = W_TERMINAL_POSITION * (1.0 - np.cos(terminal_theta - np.pi)) + (
        W_TERMINAL_VELOCITY * terminal_theta_dot**2
    )
    control = W_CONTROL * (tau_seq**2).sum(axis=0)
    return running + terminal + control


def sample_sinusoidal_candidates(
    horizon: int, n_candidates: int, tau_range: tuple[float, float], rng: np.random.Generator
) -> np.ndarray:
    """振幅・周波数・位相をランダムサンプリングした正弦波トルク列候補を生成する。

    純粋な一様乱数の区分定数トルクでは、スイングアップに必要な多周期にわたる
    協調的な「エネルギー汲み上げ」パターンをランダムショューティングで発見できない
    (1回の水平線内でランダムに当たる確率が非常に低い)ため、振り子の固有振動数
    近傍の正弦波で探索空間を物理的に妥当な部分空間に絞り込む。振幅を0近傍まで
    含むため、倒立近傍での微小な安定化トルクも表現できる。

    Returns: shape (horizon, n_candidates)
    """
    amplitude = rng.uniform(0.0, tau_range[1], size=n_candidates)
    omega = rng.uniform(*OMEGA_RANGE, size=n_candidates)
    phase = rng.uniform(0.0, 2 * np.pi, size=n_candidates)
    t = np.arange(horizon)[:, None] * DT
    return amplitude[None, :] * np.sin(omega[None, :] * t + phase[None, :])


def plan_random_shooting(
    rollout_fn: RolloutFn,
    state: np.ndarray,
    rng: np.random.Generator,
    horizon: int = HORIZON,
    n_candidates: int = N_CANDIDATES,
    tau_range: tuple[float, float] = TAU_RANGE,
) -> float:
    """現在状態から候補トルク列を多数評価し、コスト最小の先頭トルクを返す。"""
    candidates = sample_sinusoidal_candidates(horizon, n_candidates, tau_range, rng)
    state_batch = np.tile(state, (n_candidates, 1))
    traj = rollout_fn(state_batch, candidates)
    costs = total_cost(traj, candidates)
    best = np.argmin(costs)
    return float(candidates[0, best])


def run_mpc(
    rollout_fn: RolloutFn,
    initial_state: np.ndarray,
    n_steps: int,
    c: float,
    seed: int,
    horizon: int = HORIZON,
    n_candidates: int = N_CANDIDATES,
    tau_range: tuple[float, float] = TAU_RANGE,
) -> tuple[np.ndarray, np.ndarray]:
    """receding horizon MPCを実行する。実プラントは常に真の物理モデル。

    Returns: traj shape (n_steps+1, 2), taus shape (n_steps,)
    """
    rng = np.random.default_rng(seed)
    state = initial_state.copy()
    traj = [state]
    taus = []
    for _ in range(n_steps):
        u0 = plan_random_shooting(rollout_fn, state, rng, horizon, n_candidates, tau_range)
        state = rk4_step(state, DT, c=c, tau=u0)
        traj.append(state)
        taus.append(u0)
    return np.stack(traj, axis=0), np.array(taus)
