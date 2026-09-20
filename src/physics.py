"""振り子(減衰・クーロン摩擦・トルク入力あり)の真値シミュレータ。

状態は x = [theta, theta_dot]。運動方程式:
    theta_ddot = -(g / L) * sin(theta) - c * theta_dot
                 - c_coulomb * sign(theta_dot) + tau / (m * L^2)

M15 (#31) でクーロン摩擦(乾摩擦)項を追加した。これはグレーボックスモデルが
既知として埋め込んでいない「本当の未知項」であり、不連続なためNNにとっても
学習が難しい。モジュール定数 C_COULOMB をデフォルト値にすることで、全ての
呼び出し元が自動的に同じ真の系を使う(スクリプトごとの設定漏れを防ぐ)。

c=0, c_coulomb=0, tau=0 を明示的に渡せば、無減衰・無摩擦・無入力の単振り子
(M1の系)に一致する。tau はステップ内で一定(ゼロ次ホールド)として扱う。
"""
import numpy as np

# --- 真の系のパラメータ(実機そのもの。設計者は正確な値を知らない) ---
G = 9.81
L = 1.05  # M16 (#33): 公称値1.0に対して+5%のキャリブレーション誤差
M = 0.95  # M16 (#33): 公称値1.0に対して-5%のキャリブレーション誤差

# --- 設計時に既知と仮定する公称値(CAD値・実測値) ---
# グレーボックスモデルの既知項と、PIDの重力フィードフォワードはこちらを使う。
# 真値(L, M)とはキャリブレーション誤差があり、これがM16の検証対象。
L_NOMINAL = 1.0
M_NOMINAL = 1.0
C = 0.0  # 粘性減衰のデフォルトは0(M1と後方互換。学習・評価では0.15を明示的に渡す)
C_COULOMB = 0.3  # クーロン摩擦(M15で追加。真の系の「未知項」として常に有効)
TAU = 0.0  # デフォルトは無入力(M1/M2と後方互換)


def friction_acceleration(theta_dot: np.ndarray, c: float, c_coulomb: float) -> np.ndarray:
    """摩擦による加速度(粘性 + クーロン)。グレーボックスが学ぶべき未知項。"""
    return -c * theta_dot - c_coulomb * np.sign(theta_dot)


def dynamics(
    state: np.ndarray,
    c: float = C,
    tau: float = TAU,
    m: float = M,
    c_coulomb: float = C_COULOMB,
    length: float = L,
) -> np.ndarray:
    """状態の時間微分 dx/dt を返す。state: shape (..., 2)。tau はスカラーまたは (...,) 形状。"""
    theta, theta_dot = state[..., 0], state[..., 1]
    theta_ddot = (
        -(G / length) * np.sin(theta)
        + friction_acceleration(theta_dot, c, c_coulomb)
        + tau / (m * length**2)
    )
    return np.stack([theta_dot, theta_ddot], axis=-1)


def rk4_step(
    state: np.ndarray,
    dt: float,
    c: float = C,
    tau: float = TAU,
    m: float = M,
    c_coulomb: float = C_COULOMB,
    length: float = L,
) -> np.ndarray:
    """RK4で1ステップ積分する。tau はこのステップ内で一定とみなす。"""
    k1 = dynamics(state, c, tau, m, c_coulomb, length)
    k2 = dynamics(state + 0.5 * dt * k1, c, tau, m, c_coulomb, length)
    k3 = dynamics(state + 0.5 * dt * k2, c, tau, m, c_coulomb, length)
    k4 = dynamics(state + dt * k3, c, tau, m, c_coulomb, length)
    return state + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


def simulate(
    initial_state: np.ndarray,
    dt: float,
    n_steps: int,
    c: float = C,
    tau: np.ndarray | float = TAU,
    m: float = M,
    c_coulomb: float = C_COULOMB,
    length: float = L,
) -> np.ndarray:
    """初期状態から n_steps + 1 点の軌道を生成する。

    tau: スカラー(全ステップ一定)、または shape (n_steps, ...) の区分定数トルク列。

    Returns: shape (n_steps + 1, ..., 2)
    """
    tau_seq = np.full(n_steps, tau, dtype=float) if np.ndim(tau) == 0 else np.asarray(tau)
    traj = [initial_state]
    state = initial_state
    for t in range(n_steps):
        state = rk4_step(state, dt, c, tau_seq[t], m, c_coulomb, length)
        traj.append(state)
    return np.stack(traj, axis=0)


def energy(state: np.ndarray, m: float = M, length: float = L) -> np.ndarray:
    """力学的エネルギー E = 0.5*m*L^2*theta_dot^2 + m*g*L*(1-cos(theta))。

    最下点(theta=0)を基準としたポテンシャルエネルギー。トルク入力による仕事は含まない。
    """
    theta, theta_dot = state[..., 0], state[..., 1]
    kinetic = 0.5 * m * length**2 * theta_dot**2
    potential = m * G * length * (1 - np.cos(theta))
    return kinetic + potential
