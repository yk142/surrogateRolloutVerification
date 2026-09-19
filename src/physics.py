"""振り子(減衰・トルク入力あり)の真値シミュレータ。

状態は x = [theta, theta_dot]。運動方程式:
    theta_ddot = -(g / L) * sin(theta) - c * theta_dot + tau / (m * L^2)

c=0, tau=0 (デフォルト) で無減衰・無入力の単振り子(M1)に一致する。
tau はステップ内で一定(ゼロ次ホールド)として扱う。
"""
import numpy as np

G = 9.81
L = 1.0
M = 1.0
C = 0.0  # デフォルトは無減衰(M1と後方互換)
TAU = 0.0  # デフォルトは無入力(M1/M2と後方互換)


def dynamics(state: np.ndarray, c: float = C, tau: float = TAU, m: float = M) -> np.ndarray:
    """状態の時間微分 dx/dt を返す。state: shape (..., 2)。tau はスカラーまたは (...,) 形状。"""
    theta, theta_dot = state[..., 0], state[..., 1]
    theta_ddot = -(G / L) * np.sin(theta) - c * theta_dot + tau / (m * L**2)
    return np.stack([theta_dot, theta_ddot], axis=-1)


def rk4_step(
    state: np.ndarray, dt: float, c: float = C, tau: float = TAU, m: float = M
) -> np.ndarray:
    """RK4で1ステップ積分する。tau はこのステップ内で一定とみなす。"""
    k1 = dynamics(state, c, tau, m)
    k2 = dynamics(state + 0.5 * dt * k1, c, tau, m)
    k3 = dynamics(state + 0.5 * dt * k2, c, tau, m)
    k4 = dynamics(state + dt * k3, c, tau, m)
    return state + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


def simulate(
    initial_state: np.ndarray,
    dt: float,
    n_steps: int,
    c: float = C,
    tau: np.ndarray | float = TAU,
    m: float = M,
) -> np.ndarray:
    """初期状態から n_steps + 1 点の軌道を生成する。

    tau: スカラー(全ステップ一定)、または shape (n_steps, ...) の区分定数トルク列。

    Returns: shape (n_steps + 1, ..., 2)
    """
    tau_seq = np.full(n_steps, tau, dtype=float) if np.ndim(tau) == 0 else np.asarray(tau)
    traj = [initial_state]
    state = initial_state
    for t in range(n_steps):
        state = rk4_step(state, dt, c, tau_seq[t], m)
        traj.append(state)
    return np.stack(traj, axis=0)


def energy(state: np.ndarray, m: float = 1.0) -> np.ndarray:
    """力学的エネルギー E = 0.5*m*L^2*theta_dot^2 + m*g*L*(1-cos(theta))。

    最下点(theta=0)を基準としたポテンシャルエネルギー。トルク入力による仕事は含まない。
    """
    theta, theta_dot = state[..., 0], state[..., 1]
    kinetic = 0.5 * m * L**2 * theta_dot**2
    potential = m * G * L * (1 - np.cos(theta))
    return kinetic + potential
