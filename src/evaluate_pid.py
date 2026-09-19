"""同一PIDゲインを真の物理モデル・NSSサロゲートモデルに閉ループ適用したときの
倒立近傍安定化挙動を比較する。3つのPNGを outputs/ に生成する。

1. pid_trajectory.png      : 代表試行のtheta(t)推移比較
2. pid_phase_portrait.png  : 位相空間軌道の比較
3. pid_error_scatter.png   : 複数初期条件での安定化誤差(真値 vs サロゲート)の散布図
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

plt.rcParams["font.family"] = "Noto Sans CJK JP"

from src.model import NSSModel
from src.pid import angular_error_to_inverted, run_pid_surrogate, run_pid_true

DT = 0.02
N_STEPS = 400  # 8秒分
DAMPING = 0.15
WINDOW = 50  # 最終1秒間で安定化誤差を評価

REPRESENTATIVE_IC = np.array([np.pi - 0.3, 0.0])

ANGLE_OFFSETS = [-0.5, -0.3, -0.1, 0.1, 0.3, 0.5]
VELOCITY_OFFSETS = [-1.0, 0.0, 1.0]

MODEL_PATH = "outputs/nss_model.pt"
OUT_DIR = "outputs"


def load_model() -> NSSModel:
    model = NSSModel()
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    model.eval()
    return model


def stabilization_error(traj: np.ndarray) -> float:
    return float(np.abs(angular_error_to_inverted(traj[-WINDOW:, 0])).mean())


def unwrap_theta(traj: np.ndarray) -> np.ndarray:
    """NSSモデルはステップごとにatan2でthetaをデコードするため、軌道が±πを
    跨ぐと見た目上2πジャンプする(物理的には連続な同一角度)。プロット用に
    np.unwrapで連続な表現に直す(角速度など他の量はwrapの影響を受けないため
    theta列だけ置き換える)。
    """
    unwrapped = traj.copy()
    unwrapped[:, 0] = np.unwrap(traj[:, 0])
    return unwrapped


def plot_trajectory(true_traj: np.ndarray, surrogate_traj: np.ndarray) -> None:
    surrogate_traj = unwrap_theta(surrogate_traj)
    t = np.arange(N_STEPS + 1) * DT
    plt.figure(figsize=(7, 4.5))
    plt.plot(t, true_traj[:, 0], label="真の物理モデル + PID")
    plt.plot(t, surrogate_traj[:, 0], label="NSSサロゲート + PID(同一ゲイン)")
    plt.axhline(np.pi, color="gray", linestyle="--", linewidth=1, label="θ=π(倒立)")
    plt.axhline(-np.pi, color="gray", linestyle="--", linewidth=1)
    plt.xlabel("time [s]")
    plt.ylabel("theta [rad]")
    plt.title(f"PID安定化のθ(t)推移 (IC: θ0=π-0.3, θ̇0=0)")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/pid_trajectory.png", dpi=150)
    plt.close()
    print(f"saved {OUT_DIR}/pid_trajectory.png")


def plot_phase_portrait(true_traj: np.ndarray, surrogate_traj: np.ndarray) -> None:
    surrogate_traj = unwrap_theta(surrogate_traj)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, (label, traj) in zip(
        axes, [("真の物理モデル + PID", true_traj), ("NSSサロゲート + PID", surrogate_traj)]
    ):
        ax.plot(traj[:, 0], traj[:, 1], linewidth=1.0)
        ax.scatter([traj[0, 0]], [traj[0, 1]], color="green", marker="o", label="開始", zorder=3)
        ax.scatter([traj[-1, 0]], [traj[-1, 1]], color="red", marker="x", label="終了", zorder=3)
        ax.axvline(np.pi, color="gray", linestyle="--", linewidth=1)
        ax.axvline(-np.pi, color="gray", linestyle="--", linewidth=1)
        ax.set_xlabel("theta [rad]")
        ax.set_ylabel("theta_dot [rad/s]")
        ax.set_title(label, fontsize=10)
        ax.legend(fontsize=8)

    fig.suptitle("PID安定化の位相空間軌道 (IC: θ0=π-0.3, θ̇0=0)")
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/pid_phase_portrait.png", dpi=150)
    plt.close()
    print(f"saved {OUT_DIR}/pid_phase_portrait.png")


def plot_error_scatter(model: NSSModel) -> None:
    true_errors = []
    surrogate_errors = []
    labels = []

    for angle_offset in ANGLE_OFFSETS:
        for velocity_offset in VELOCITY_OFFSETS:
            ic = np.array([np.pi + angle_offset, velocity_offset])
            true_traj, _ = run_pid_true(ic, N_STEPS, c=DAMPING)
            surrogate_traj, _ = run_pid_surrogate(model, ic, N_STEPS)

            true_errors.append(stabilization_error(true_traj))
            surrogate_errors.append(stabilization_error(surrogate_traj))
            labels.append(f"Δθ0={angle_offset:+.1f}, θ̇0={velocity_offset:+.1f}")
            print(f"{labels[-1]}: true={true_errors[-1]:.4f} surrogate={surrogate_errors[-1]:.4f}")

    true_errors = np.array(true_errors)
    surrogate_errors = np.array(surrogate_errors)

    plt.figure(figsize=(6, 6))
    max_val = max(true_errors.max(), surrogate_errors.max()) * 1.1 + 1e-3
    plt.plot([0, max_val], [0, max_val], color="gray", linestyle="--", label="真値=サロゲート")
    plt.scatter(true_errors, surrogate_errors)
    plt.xlabel("真の物理モデル+PIDの安定化誤差 [rad]")
    plt.ylabel("NSSサロゲート+PIDの安定化誤差 [rad]")
    plt.title(f"倒立近傍PID安定化誤差の比較 (n={len(true_errors)} 初期条件)")
    plt.legend(fontsize=8)
    plt.xlim(0, max_val)
    plt.ylim(0, max_val)
    plt.gca().set_aspect("equal")
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/pid_error_scatter.png", dpi=150)
    plt.close()
    print(f"saved {OUT_DIR}/pid_error_scatter.png")

    n_true_success = (true_errors < 0.1).sum()
    n_surrogate_success = (surrogate_errors < 0.1).sum()
    print(
        f"success (final_err<0.1rad): true={n_true_success}/{len(true_errors)} "
        f"surrogate={n_surrogate_success}/{len(surrogate_errors)}"
    )


if __name__ == "__main__":
    m = load_model()
    true_traj, _ = run_pid_true(REPRESENTATIVE_IC, N_STEPS, c=DAMPING)
    surrogate_traj, _ = run_pid_surrogate(m, REPRESENTATIVE_IC, N_STEPS)

    plot_trajectory(true_traj, surrogate_traj)
    plot_phase_portrait(true_traj, surrogate_traj)
    plot_error_scatter(m)
