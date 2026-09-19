"""NSSサロゲートMPC vs オラクル(真の物理モデル)MPC vs ベースライン(M3の開ループ
共振スイングアップトルク、フィードバックなし)のスイングアップ制御検証。

3つのPNGを outputs/ に生成する:
1. swingup_trajectory.png : 代表試行のtheta(t)推移
2. swingup_phase_portrait.png : 位相空間軌道の比較
3. swingup_stabilization_error.png : 複数試行での倒立安定化誤差の分布比較
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

plt.rcParams["font.family"] = "Noto Sans CJK JP"

from src.control import make_nss_rollout_fn, make_true_rollout_fn, run_mpc
from src.model import NSSModel
from src.physics import G, L, simulate

DT = 0.02
N_STEPS = 400  # 8秒分(スイングアップ+安定化の収束を見るのに十分な長さ)
DAMPING = 0.15
N_TRIALS = 5
SEED_BASE = 2000

# ベースライン: M3で検証した開ループ共振スイングアップトルク(フィードバックなし)
RESONANT_AMPLITUDE = 2.0
RESONANT_OMEGA = np.sqrt(G / L)

MODEL_PATH = "outputs/nss_model.pt"
OUT_DIR = "outputs"


def load_model() -> NSSModel:
    model = NSSModel()
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    model.eval()
    return model


def run_baseline_open_loop(seed: int) -> tuple[np.ndarray, np.ndarray]:
    t = np.arange(N_STEPS) * DT
    tau_seq = RESONANT_AMPLITUDE * np.sin(RESONANT_OMEGA * t)
    traj = simulate(np.array([0.0, 0.0]), DT, N_STEPS, c=DAMPING, tau=tau_seq)
    return traj, tau_seq


def angular_distance_to_inverted(theta: np.ndarray) -> np.ndarray:
    """theta と pi(倒立)との角距離。0で倒立、piで最下点。"""
    return np.abs(np.arctan2(np.sin(theta - np.pi), np.cos(theta - np.pi)))


def run_all_conditions(model: NSSModel, seed: int):
    nss_rollout_fn = make_nss_rollout_fn(model)
    true_rollout_fn = make_true_rollout_fn(c=DAMPING)

    nss_traj, _ = run_mpc(nss_rollout_fn, np.array([0.0, 0.0]), N_STEPS, c=DAMPING, seed=seed)
    oracle_traj, _ = run_mpc(true_rollout_fn, np.array([0.0, 0.0]), N_STEPS, c=DAMPING, seed=seed)
    baseline_traj, _ = run_baseline_open_loop(seed)

    return {
        "NSSサロゲートMPC": nss_traj,
        "オラクルMPC(真の物理モデル)": oracle_traj,
        "ベースライン(開ループ共振トルク)": baseline_traj,
    }


def plot_trajectory(trajs: dict[str, np.ndarray]) -> None:
    t = np.arange(N_STEPS + 1) * DT
    plt.figure(figsize=(8, 4.5))
    for label, traj in trajs.items():
        plt.plot(t, traj[:, 0], label=label)
    plt.axhline(np.pi, color="gray", linestyle="--", linewidth=1, label="θ=π(倒立)")
    plt.axhline(-np.pi, color="gray", linestyle="--", linewidth=1)
    plt.xlabel("time [s]")
    plt.ylabel("theta [rad]")
    plt.title("スイングアップ試行のθ(t)推移(代表試行 seed=2000)")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/swingup_trajectory.png", dpi=150)
    plt.close()
    print(f"saved {OUT_DIR}/swingup_trajectory.png")


def plot_phase_portrait(trajs: dict[str, np.ndarray]) -> None:
    fig, axes = plt.subplots(1, len(trajs), figsize=(6 * len(trajs), 5))
    for ax, (label, traj) in zip(axes, trajs.items()):
        ax.plot(traj[:, 0], traj[:, 1], linewidth=1.0)
        ax.scatter([traj[0, 0]], [traj[0, 1]], color="green", marker="o", label="開始", zorder=3)
        ax.scatter([traj[-1, 0]], [traj[-1, 1]], color="red", marker="x", label="終了", zorder=3)
        ax.axvline(np.pi, color="gray", linestyle="--", linewidth=1)
        ax.axvline(-np.pi, color="gray", linestyle="--", linewidth=1)
        ax.set_xlabel("theta [rad]")
        ax.set_ylabel("theta_dot [rad/s]")
        ax.set_title(label, fontsize=10)
        ax.legend(fontsize=8)

    fig.suptitle("スイングアップ試行の位相空間軌道(代表試行 seed=2000)")
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/swingup_phase_portrait.png", dpi=150)
    plt.close()
    print(f"saved {OUT_DIR}/swingup_phase_portrait.png")


def plot_stabilization_error(model: NSSModel) -> None:
    nss_rollout_fn = make_nss_rollout_fn(model)
    true_rollout_fn = make_true_rollout_fn(c=DAMPING)

    errors = {"NSSサロゲートMPC": [], "オラクルMPC(真の物理モデル)": [], "ベースライン(開ループ共振トルク)": []}
    window = 50  # 最後の1秒(=50ステップ)の平均角距離で安定化誤差を評価

    for i in range(N_TRIALS):
        seed = SEED_BASE + i
        nss_traj, _ = run_mpc(nss_rollout_fn, np.array([0.0, 0.0]), N_STEPS, c=DAMPING, seed=seed)
        oracle_traj, _ = run_mpc(
            true_rollout_fn, np.array([0.0, 0.0]), N_STEPS, c=DAMPING, seed=seed
        )
        baseline_traj, _ = run_baseline_open_loop(seed)

        errors["NSSサロゲートMPC"].append(
            angular_distance_to_inverted(nss_traj[-window:, 0]).mean()
        )
        errors["オラクルMPC(真の物理モデル)"].append(
            angular_distance_to_inverted(oracle_traj[-window:, 0]).mean()
        )
        errors["ベースライン(開ループ共振トルク)"].append(
            angular_distance_to_inverted(baseline_traj[-window:, 0]).mean()
        )
        print(f"trial {i} (seed={seed}) done")

    plt.figure(figsize=(6, 4.5))
    labels = list(errors.keys())
    data = [errors[label] for label in labels]
    plt.boxplot(data, tick_labels=[l.replace("(", "\n(") for l in labels])
    plt.ylabel("最終1秒間の平均角距離 |θ-π| [rad] (0=倒立で完全静止)")
    plt.title(f"倒立安定化誤差の分布 (n={N_TRIALS} trials)")
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/swingup_stabilization_error.png", dpi=150)
    plt.close()
    print(f"saved {OUT_DIR}/swingup_stabilization_error.png")

    for label in labels:
        vals = errors[label]
        print(f"[{label}] mean={np.mean(vals):.4f} std={np.std(vals):.4f} vals={vals}")


if __name__ == "__main__":
    m = load_model()
    trajs = run_all_conditions(m, seed=SEED_BASE)
    plot_trajectory(trajs)
    plot_phase_portrait(trajs)
    plot_stabilization_error(m)
