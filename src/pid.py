"""任意の目標角度(倒立近傍)を追従するPID制御(PTP制御)。

M4のMPCは予測モデルを使ったスイングアップ制御だったが、M5ではより単純な
モデルフリーのPIDフィードバック制御で倒立近傍を安定化できるかを検証した。
M6ではPIDの目標角度を一般化し、倒立近傍の異なる目標角度間を移動するPTP制御を
検証する(最下点からのスイングアップはPID単体では原理的に困難なため対象外)。
同一ゲインを真の物理モデル・NSSサロゲートモデルそれぞれに閉ループで適用し、
挙動の一致度からサロゲートの倒立近傍における局所ダイナミクスの妥当性を見る。
"""
import numpy as np

from src.model import AutoregressiveModel
from src.physics import G, L, M, rk4_step

DT = 0.02
TAU_MAX = 4.0
INTEGRAL_CLIP = 2.0

# 真の物理モデル(c=0.15)でチューニング済みのゲイン。
KP = 20.0
KI = 2.0
KD = 5.0


def angular_error(theta: np.ndarray, target: np.ndarray) -> np.ndarray:
    """theta と target との符号付き誤差。wrap済みで (-pi, pi] に収まる。"""
    return np.arctan2(np.sin(theta - target), np.cos(theta - target))


def gravity_feedforward(target: float) -> float:
    """目標角度targetで静止するために必要な定常トルク(重力補償)。

    theta=pi(倒立)ではsin(pi)=0のためゼロになりM5の挙動と一致するが、
    倒立近傍の他の目標角度では重力が theta_ddot に -(g/L)sin(theta) の
    定常的な偏りを与えるため、PIDの積分項だけに頼らずこれを直接打ち消す。
    """
    return M * G * L * np.sin(target)


def angular_error_to_inverted(theta: np.ndarray) -> np.ndarray:
    """theta と pi(倒立)との符号付き誤差(angular_errorのtarget=pi特殊形)。"""
    return angular_error(theta, np.pi)


class PIDController:
    """任意目標角度のPIDコントローラ。ステップごとに状態を観測してトルクを計算する。"""

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
        self._last_target: float | None = None

    def reset(self) -> None:
        self.integral = 0.0
        self._last_target = None

    def compute(self, state: np.ndarray, target: float = np.pi) -> float:
        # 目標角度が変わった(PTPで次の目標に切り替わった)瞬間は積分項をリセットし、
        # 前の目標に対するアンチワインドアップの蓄積を持ち越さないようにする。
        if self._last_target is not None and target != self._last_target:
            self.integral = 0.0
        self._last_target = target

        theta, theta_dot = state[0], state[1]
        e = angular_error(theta, target)
        self.integral = np.clip(
            self.integral + e * self.dt, -self.integral_clip, self.integral_clip
        )
        tau = gravity_feedforward(target) - (
            self.kp * e + self.ki * self.integral + self.kd * theta_dot
        )
        return float(np.clip(tau, -self.tau_max, self.tau_max))


def _as_target_seq(target: float | np.ndarray, n_steps: int) -> np.ndarray:
    """スカラーなら全ステップ一定、配列ならそのまま(PTPの目標系列)として扱う。"""
    return np.full(n_steps, target, dtype=float) if np.ndim(target) == 0 else np.asarray(target)


def run_pid_true(
    initial_state: np.ndarray,
    n_steps: int,
    c: float,
    target: float | np.ndarray = np.pi,
    controller: PIDController | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """真の物理モデルを閉ループでPID制御する。

    target: スカラー(倒立などの固定目標)、または shape (n_steps,) のPTP目標系列。

    Returns: traj shape (n_steps+1, 2), taus shape (n_steps,)
    """
    controller = controller or PIDController()
    controller.reset()
    target_seq = _as_target_seq(target, n_steps)
    state = initial_state.copy()
    traj = [state.copy()]
    taus = []
    for t in range(n_steps):
        tau = controller.compute(state, target_seq[t])
        state = rk4_step(state, DT, c=c, tau=tau)
        traj.append(state.copy())
        taus.append(tau)
    return np.stack(traj, axis=0), np.array(taus)


def run_pid_surrogate(
    model: AutoregressiveModel,
    initial_state: np.ndarray,
    n_steps: int,
    target: float | np.ndarray = np.pi,
    controller: PIDController | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """NSSサロゲートモデルを閉ループでPID制御する(サロゲート自身の状態を観測)。

    target: スカラー(倒立などの固定目標)、または shape (n_steps,) のPTP目標系列。

    Returns: traj shape (n_steps+1, 2), taus shape (n_steps,)
    """
    controller = controller or PIDController()
    controller.reset()
    target_seq = _as_target_seq(target, n_steps)
    state = initial_state.copy()
    traj = [state.copy()]
    taus = []
    for t in range(n_steps):
        tau = controller.compute(state, target_seq[t])
        state = model.rollout(state, 1, tau_seq=np.array([tau]))[-1]
        traj.append(state.copy())
        taus.append(tau)
    return np.stack(traj, axis=0), np.array(taus)
