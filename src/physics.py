"""単振り子(無減衰・無入力)の真値シミュレータ。

状態は x = [theta, theta_dot]。運動方程式:
    theta_ddot = -(g / L) * sin(theta)
"""
import numpy as np

G = 9.81
L = 1.0


def dynamics(state: np.ndarray) -> np.ndarray:
    """状態の時間微分 dx/dt を返す。state: shape (..., 2)."""
    theta, theta_dot = state[..., 0], state[..., 1]
    theta_ddot = -(G / L) * np.sin(theta)
    return np.stack([theta_dot, theta_ddot], axis=-1)


def rk4_step(state: np.ndarray, dt: float) -> np.ndarray:
    """RK4で1ステップ積分する。"""
    k1 = dynamics(state)
    k2 = dynamics(state + 0.5 * dt * k1)
    k3 = dynamics(state + 0.5 * dt * k2)
    k4 = dynamics(state + dt * k3)
    return state + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


def simulate(initial_state: np.ndarray, dt: float, n_steps: int) -> np.ndarray:
    """初期状態から n_steps + 1 点の軌道を生成する。

    Returns: shape (n_steps + 1, ..., 2)
    """
    traj = [initial_state]
    state = initial_state
    for _ in range(n_steps):
        state = rk4_step(state, dt)
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
