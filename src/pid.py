"""倒立(theta=pi)を目標としたPID制御による安定化。

M4のMPCは予測モデルを使ったスイングアップ制御だったが、M5ではより単純な
モデルフリーのPIDフィードバック制御で倒立近傍を安定化できるかを検証する。
同一ゲインを真の物理モデル・NSSサロゲートモデルそれぞれに閉ループで適用し、
挙動の一致度からサロゲートの倒立近傍における局所ダイナミクスの妥当性を見る。
"""
import numpy as np

from src.model import NSSModel
from src.physics import rk4_step

DT = 0.02
TAU_MAX = 4.0
INTEGRAL_CLIP = 2.0

# 真の物理モデル(c=0.15)でチューニング済みのゲイン。
KP = 20.0
KI = 2.0
KD = 5.0


def angular_error_to_inverted(theta: np.ndarray) -> np.ndarray:
    """theta と pi(倒立)との符号付き誤差。wrap済みで (-pi, pi] に収まる。"""
    return np.arctan2(np.sin(theta - np.pi), np.cos(theta - np.pi))


class PIDController:
    """倒立目標のPIDコントローラ。ステップごとに状態を観測してトルクを計算する。"""

    def __init__(
        self,
        kp: float = KP,
        ki: float = KI,
        kd: float = KD,
        dt: float = DT,
        tau_max: float = TAU_MAX,
        integral_clip: float = INTEGRAL_CLIP,
    ):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.dt = dt
        self.tau_max = tau_max
        self.integral_clip = integral_clip
        self.integral = 0.0

    def reset(self) -> None:
        self.integral = 0.0

    def compute(self, state: np.ndarray) -> float:
        theta, theta_dot = state[0], state[1]
        e = angular_error_to_inverted(theta)
        self.integral = np.clip(
            self.integral + e * self.dt, -self.integral_clip, self.integral_clip
        )
        tau = -(self.kp * e + self.ki * self.integral + self.kd * theta_dot)
        return float(np.clip(tau, -self.tau_max, self.tau_max))


def run_pid_true(
    initial_state: np.ndarray, n_steps: int, c: float, controller: PIDController | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """真の物理モデルを閉ループでPID制御する。

    Returns: traj shape (n_steps+1, 2), taus shape (n_steps,)
    """
    controller = controller or PIDController()
    controller.reset()
    state = initial_state.copy()
    traj = [state.copy()]
    taus = []
    for _ in range(n_steps):
        tau = controller.compute(state)
        state = rk4_step(state, DT, c=c, tau=tau)
        traj.append(state.copy())
        taus.append(tau)
    return np.stack(traj, axis=0), np.array(taus)


def run_pid_surrogate(
    model: NSSModel,
    initial_state: np.ndarray,
    n_steps: int,
    controller: PIDController | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """NSSサロゲートモデルを閉ループでPID制御する(サロゲート自身の状態を観測)。

    Returns: traj shape (n_steps+1, 2), taus shape (n_steps,)
    """
    controller = controller or PIDController()
    controller.reset()
    state = initial_state.copy()
    traj = [state.copy()]
    taus = []
    for _ in range(n_steps):
        tau = controller.compute(state)
        state = model.rollout(state, 1, tau_seq=np.array([tau]))[-1]
        traj.append(state.copy())
        taus.append(tau)
    return np.stack(traj, axis=0), np.array(taus)
