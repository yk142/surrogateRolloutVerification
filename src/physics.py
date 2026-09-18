"""振り子(減衰・無入力)の真値シミュレータ。

状態は x = [theta, theta_dot]。運動方程式:
    theta_ddot = -(g / L) * sin(theta) - c * theta_dot

c=0 (デフォルト) で無減衰の単振り子に一致する。
"""
import numpy as np

G = 9.81
L = 1.0
C = 0.0  # デフォルトは無減衰(M1と後方互換)


def dynamics(state: np.ndarray, c: float = C) -> np.ndarray:
    """状態の時間微分 dx/dt を返す。state: shape (..., 2)."""
    theta, theta_dot = state[..., 0], state[..., 1]
    theta_ddot = -(G / L) * np.sin(theta) - c * theta_dot
    return np.stack([theta_dot, theta_ddot], axis=-1)


def rk4_step(state: np.ndarray, dt: float, c: float = C) -> np.ndarray:
    """RK4で1ステップ積分する。"""
    k1 = dynamics(state, c)
    k2 = dynamics(state + 0.5 * dt * k1, c)
    k3 = dynamics(state + 0.5 * dt * k2, c)
    k4 = dynamics(state + dt * k3, c)
    return state + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


def simulate(initial_state: np.ndarray, dt: float, n_steps: int, c: float = C) -> np.ndarray:
    """初期状態から n_steps + 1 点の軌道を生成する。

    Returns: shape (n_steps + 1, ..., 2)
    """
    traj = [initial_state]
    state = initial_state
    for _ in range(n_steps):
        state = rk4_step(state, dt, c)
        traj.append(state)
    return np.stack(traj, axis=0)


def energy(state: np.ndarray, m: float = 1.0) -> np.ndarray:
    """力学的エネルギー E = 0.5*m*L^2*theta_dot^2 + m*g*L*(1-cos(theta))。

    最下点(theta=0)を基準としたポテンシャルエネルギー。
    """
    theta, theta_dot = state[..., 0], state[..., 1]
    kinetic = 0.5 * m * L**2 * theta_dot**2
    potential = m * G * L * (1 - np.cos(theta))
    return kinetic + potential
