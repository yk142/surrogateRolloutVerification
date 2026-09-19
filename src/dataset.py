"""学習・評価用の軌道データセット生成(c=減衰係数を指定可能)。

M1では一様サンプリングのみだったが、倒立近傍・完全回転域(セパラトリクス超え)が
著しく過小表現され、NSSサロゲートがその領域で正しくロールアウトできないことが
判明した(Issue #1)。M2では3層のIC分布からサンプリングする:

- normal:    通常域。theta, theta_dot を一様にサンプリング(M1と同じ)。
- rotation:  完全回転域。theta によらずセパラトリクスエネルギーを確実に超える
             |theta_dot| を与える。
- inverted:  倒立近傍。theta ~= pi、theta_dot は小さい。
"""
import numpy as np

from src.physics import G, L, simulate

THETA_RANGE = (-np.pi, np.pi)
THETA_DOT_RANGE = (-6.0, 6.0)

# theta=0 で完全回転に必要な最小 |theta_dot| = sqrt(4*g/L)(セパラトリクスエネルギー)。
# これを超えれば theta によらず完全回転になる。
ROTATION_THRESHOLD = np.sqrt(4 * G / L)
ROTATION_THETA_DOT_RANGE = (ROTATION_THRESHOLD + 0.1, ROTATION_THRESHOLD + 3.0)

INVERTED_THETA_HALF_WIDTH = 0.4
INVERTED_THETA_DOT_RANGE = (-1.0, 1.0)

STRATUM_WEIGHTS = {"normal": 0.5, "rotation": 0.25, "inverted": 0.25}


def _sample_normal(n: int, rng: np.random.Generator) -> np.ndarray:
    theta = rng.uniform(*THETA_RANGE, size=n)
    theta_dot = rng.uniform(*THETA_DOT_RANGE, size=n)
    return np.stack([theta, theta_dot], axis=-1)


def _sample_rotation(n: int, rng: np.random.Generator) -> np.ndarray:
    theta = rng.uniform(*THETA_RANGE, size=n)
    magnitude = rng.uniform(*ROTATION_THETA_DOT_RANGE, size=n)
    sign = rng.choice([-1.0, 1.0], size=n)
    theta_dot = sign * magnitude
    return np.stack([theta, theta_dot], axis=-1)


def _sample_inverted(n: int, rng: np.random.Generator) -> np.ndarray:
    sign = rng.choice([-1.0, 1.0], size=n)
    theta = sign * np.pi + rng.uniform(
        -INVERTED_THETA_HALF_WIDTH, INVERTED_THETA_HALF_WIDTH, size=n
    )
    theta_dot = rng.uniform(*INVERTED_THETA_DOT_RANGE, size=n)
    return np.stack([theta, theta_dot], axis=-1)


def sample_initial_states(n: int, rng: np.random.Generator) -> np.ndarray:
    """3層(normal/rotation/inverted)から層化サンプリングする。"""
    n_rotation = int(round(n * STRATUM_WEIGHTS["rotation"]))
    n_inverted = int(round(n * STRATUM_WEIGHTS["inverted"]))
    n_normal = n - n_rotation - n_inverted

    states = np.concatenate(
        [
            _sample_normal(n_normal, rng),
            _sample_rotation(n_rotation, rng),
            _sample_inverted(n_inverted, rng),
        ],
        axis=0,
    )
    rng.shuffle(states)
    return states


def generate_trajectories(
    n_trajectories: int, dt: float, n_steps: int, seed: int, c: float = 0.0
) -> np.ndarray:
    """shape (n_trajectories, n_steps + 1, 2) の軌道データを返す。"""
    rng = np.random.default_rng(seed)
    initial_states = sample_initial_states(n_trajectories, rng)
    trajectories = np.stack(
        [simulate(s0, dt, n_steps, c=c) for s0 in initial_states], axis=0
    )
    return trajectories


def make_transition_pairs(trajectories: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """軌道群から1-step遷移ペア (x_t, x_{t+1}) を作る。

    trajectories: shape (n_traj, n_steps + 1, 2)
    """
    x = trajectories[:, :-1, :].reshape(-1, 2)
    x_next = trajectories[:, 1:, :].reshape(-1, 2)
    return x, x_next


# --- M3: トルク入力を含む制御データセット ---

TAU_RANGE = (-4.0, 4.0)
TORQUE_HOLD_STEPS = 5  # このステップ数ごとにトルクを変更する(区分定数=ゼロ次ホールド)


def sample_torque_sequence(
    n_steps: int,
    rng: np.random.Generator,
    tau_range: tuple[float, float] = TAU_RANGE,
    hold_steps: int = TORQUE_HOLD_STEPS,
) -> np.ndarray:
    """ランダムな区分定数トルク列を生成する。shape (n_steps,)"""
    n_holds = int(np.ceil(n_steps / hold_steps))
    hold_values = rng.uniform(*tau_range, size=n_holds)
    return np.repeat(hold_values, hold_steps)[:n_steps]


def generate_controlled_trajectories(
    n_trajectories: int, dt: float, n_steps: int, seed: int, c: float = 0.0
) -> tuple[np.ndarray, np.ndarray]:
    """ランダムトルク列で駆動した軌道データセットを生成する。

    Returns:
        trajectories: shape (n_traj, n_steps + 1, 2)
        tau_seqs:     shape (n_traj, n_steps)
    """
    rng = np.random.default_rng(seed)
    initial_states = sample_initial_states(n_trajectories, rng)
    tau_seqs = np.stack(
        [sample_torque_sequence(n_steps, rng) for _ in range(n_trajectories)], axis=0
    )
    trajectories = np.stack(
        [
            simulate(s0, dt, n_steps, c=c, tau=tau_seq)
            for s0, tau_seq in zip(initial_states, tau_seqs)
        ],
        axis=0,
    )
    return trajectories, tau_seqs


def make_transition_pairs_with_control(
    trajectories: np.ndarray, tau_seqs: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """軌道群+トルク列から1-step遷移ペア (x_t, u_t, x_{t+1}) を作る。

    trajectories: shape (n_traj, n_steps + 1, 2), tau_seqs: shape (n_traj, n_steps)
    """
    x = trajectories[:, :-1, :].reshape(-1, 2)
    x_next = trajectories[:, 1:, :].reshape(-1, 2)
    u = tau_seqs.reshape(-1, 1)
    return x, u, x_next
