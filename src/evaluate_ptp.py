"""同一PID(重力フィードフォワード込み)ゲインを真の物理モデル・NSSサロゲートモデルに
閉ループ適用し、倒立近傍でのPTP(点対点)制御を比較する。3つのPNGを outputs/ に生成する。

1. ptp_trajectory.png     : 代表的な複数目標系列でのtheta(t)追従比較
2. ptp_phase_portrait.png : 位相空間軌道の比較
3. ptp_step_response.png  : 倒立からの目標オフセット量を変えた単発ステップ応答の最終誤差比較
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

plt.rcParams["font.family"] = "Noto Sans CJK JP"

from src.model import AutoregressiveModel, GrayBoxNSSModel
from src.pid import angular_error, run_pid_surrogate, run_pid_true

DT = 0.02
DAMPING = 0.15
SEGMENT_STEPS = 150  # 各目標を3秒保持
WINDOW = 30  # 各セグメント最終0.6秒で追従誤差を評価

# 代表的なPTP目標系列(倒立近傍、スイングアップ不要な範囲)
REPRESENTATIVE_TARGETS = [np.pi - 0.2, np.pi + 0.2, np.pi - 0.15]
REPRESENTATIVE_IC = np.array([np.pi, 0.0])

# 単発ステップ応答: 倒立(π)からのオフセット量を変えて追従誤差を見る
STEP_OFFSETS = np.array([-0.4, -0.3, -0.2, -0.1, 0.1, 0.2, 0.3, 0.4])

MODEL_PATH = "outputs/nss_model.pt"
OUT_DIR = "outputs"


def load_model() -> AutoregressiveModel:
    model = GrayBoxNSSModel()
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    model.eval()
    return model


def make_target_seq(targets: list[float]) -> np.ndarray:
    return np.concatenate([np.full(SEGMENT_STEPS, t) for t in targets])


def segment_final_errors(traj: np.ndarray, targets: list[float]) -> list[float]:
    errors = []
    for i, target in enumerate(targets):
        seg = traj[i * SEGMENT_STEPS + 1 : (i + 1) * SEGMENT_STEPS + 1, 0]
        errors.append(float(np.abs(angular_error(seg[-WINDOW:], target)).mean()))
    return errors


def unwrap_theta(traj: np.ndarray) -> np.ndarray:
    """NSSモデルはステップごとにatan2でthetaをデコードするため、軌道が±πを
    跨ぐと見た目上2πジャンプする(物理的には連続な同一角度)。プロット用に
    np.unwrapで連続な表現に直す(角速度など他の量はwrapの影響を受けないため
    theta列だけ置き換える)。
    """
    unwrapped = traj.copy()
    unwrapped[:, 0] = np.unwrap(traj[:, 0])
    return unwrapped


def plot_trajectory(
    true_traj: np.ndarray, surrogate_traj: np.ndarray, target_seq: np.ndarray
) -> None:
    n_steps = len(target_seq)
    t = np.arange(n_steps + 1) * DT
    target_plot = np.concatenate([[target_seq[0]], target_seq])
    surrogate_traj = unwrap_theta(surrogate_traj)

    plt.figure(figsize=(8, 4.5))
    plt.plot(t, target_plot, color="gray", linestyle="--", label="目標角度")
    plt.plot(t, true_traj[:, 0], label="真の物理モデル + PID")
    plt.plot(t, surrogate_traj[:, 0], label="NSSサロゲート + PID(同一ゲイン)")
    plt.xlabel("time [s]")
    plt.ylabel("theta [rad]")
    plt.title("PTP制御のθ(t)推移(倒立近傍、目標切り替え)")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/ptp_trajectory.png", dpi=150)
    plt.close()
    print(f"saved {OUT_DIR}/ptp_trajectory.png")


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
        ax.set_xlabel("theta [rad]")
        ax.set_ylabel("theta_dot [rad/s]")
        ax.set_title(label, fontsize=10)
        ax.legend(fontsize=8)

    fig.suptitle("PTP制御の位相空間軌道(倒立近傍、目標切り替え)")
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/ptp_phase_portrait.png", dpi=150)
    plt.close()
    print(f"saved {OUT_DIR}/ptp_phase_portrait.png")


def plot_step_response(model: AutoregressiveModel) -> None:
    true_errors = []
    surrogate_errors = []

    for offset in STEP_OFFSETS:
        target = np.pi + offset
        true_traj, _ = run_pid_true(REPRESENTATIVE_IC, SEGMENT_STEPS, c=DAMPING, target=target)
        surrogate_traj, _ = run_pid_surrogate(model, REPRESENTATIVE_IC, SEGMENT_STEPS, target=target)

        true_err = float(np.abs(angular_error(true_traj[-WINDOW:, 0], target)).mean())
        surrogate_err = float(np.abs(angular_error(surrogate_traj[-WINDOW:, 0], target)).mean())
        true_errors.append(true_err)
        surrogate_errors.append(surrogate_err)
        print(f"offset={offset:+.1f}: true={true_err:.4f} surrogate={surrogate_err:.4f}")

    plt.figure(figsize=(7, 4.5))
    width = 0.035
    plt.bar(STEP_OFFSETS - width / 2, true_errors, width=width, label="真の物理モデル + PID")
    plt.bar(STEP_OFFSETS + width / 2, surrogate_errors, width=width, label="NSSサロゲート + PID")
    plt.xlabel("目標オフセット (target - π) [rad]")
    plt.ylabel("最終0.6秒間の平均追従誤差 [rad]")
    plt.title("倒立からの単発ステップ応答の最終追従誤差")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/ptp_step_response.png", dpi=150)
    plt.close()
    print(f"saved {OUT_DIR}/ptp_step_response.png")


if __name__ == "__main__":
    m = load_model()
    target_seq = make_target_seq(REPRESENTATIVE_TARGETS)

    true_traj, _ = run_pid_true(REPRESENTATIVE_IC, len(target_seq), c=DAMPING, target=target_seq)
    surrogate_traj, _ = run_pid_surrogate(m, REPRESENTATIVE_IC, len(target_seq), target=target_seq)

    print("true segment errors:", segment_final_errors(true_traj, REPRESENTATIVE_TARGETS))
    print("surrogate segment errors:", segment_final_errors(surrogate_traj, REPRESENTATIVE_TARGETS))

    plot_trajectory(true_traj, surrogate_traj, target_seq)
    plot_phase_portrait(true_traj, surrogate_traj)
    plot_step_response(m)
