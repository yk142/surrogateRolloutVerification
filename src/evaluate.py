"""学習済みNSSモデル(トルク入力あり)のロールアウト誤差検証。

3種類のトルク条件(ゼロ/ランダム/共振スイングアップ的トルク)それぞれについて、
3つのPNGを outputs/ に生成する。

1. rollout_error_curve.png : 条件ごとのhorizonに対するRMSE蓄積カーブ
2. energy_deviation.png    : 条件ごとのエネルギー逸脱(真値 vs サロゲート)
3. phase_portrait.png      : 条件ごとの代表IC位相空間軌道比較(スイングアップの再現度を含む)
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

plt.rcParams["font.family"] = "Noto Sans CJK JP"

from src.dataset import sample_initial_states, sample_torque_sequence
from src.model import NSSModel
from src.physics import G, L, energy
from src.rollout import rmse_curve, true_rollout

DT = 0.02
N_STEPS_EVAL = 300  # 6秒分。学習時ホライズン(50ステップ=1秒)より大幅に長い
N_TEST_TRAJ = 200
SEED_TEST = 1000
DAMPING = 0.15  # train.py と揃える(M2)

# スイングアップ的共振トルク: 振り子の固有角振動数に同期させた正弦波(手動設計・開ループ)
RESONANT_AMPLITUDE = 2.0
RESONANT_OMEGA = np.sqrt(G / L)

MODEL_PATH = "outputs/nss_model.pt"
OUT_DIR = "outputs"


def make_control_conditions() -> dict[str, np.ndarray]:
    t = np.arange(N_STEPS_EVAL) * DT
    rng = np.random.default_rng(SEED_TEST + 1)
    return {
        "ゼロトルク (M2回帰確認)": np.zeros(N_STEPS_EVAL),
        "ランダムトルク列": sample_torque_sequence(N_STEPS_EVAL, rng),
        "共振スイングアップ的トルク": RESONANT_AMPLITUDE * np.sin(RESONANT_OMEGA * t),
    }


def load_model() -> NSSModel:
    model = NSSModel()
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    model.eval()
    return model


def plot_error_curve(model: NSSModel, conditions: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(SEED_TEST)
    ics = sample_initial_states(N_TEST_TRAJ, rng)
    t = np.arange(N_STEPS_EVAL + 1) * DT

    curves = {}
    plt.figure(figsize=(7, 4.5))
    for label, tau_seq in conditions.items():
        true_traj = true_rollout(ics, DT, N_STEPS_EVAL, c=DAMPING, tau_seq=tau_seq)
        pred_traj = model.rollout(ics, N_STEPS_EVAL, tau_seq=tau_seq)
        true_traj_t_first = np.transpose(true_traj, (1, 0, 2))
        curve = rmse_curve(true_traj_t_first, pred_traj)
        curves[label] = curve
        plt.plot(t, curve, label=label)

    plt.axvline(50 * DT, color="gray", linestyle="--", label="学習ホライズン境界")
    plt.xlabel("time [s]")
    plt.ylabel("RMSE (theta[rad], theta_dot[rad/s] 合成)")
    plt.title(f"ロールアウト誤差蓄積カーブ (n={N_TEST_TRAJ} test trajectories)")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/rollout_error_curve.png", dpi=150)
    plt.close()
    print(f"saved {OUT_DIR}/rollout_error_curve.png")
    return curves


def plot_energy_deviation(model: NSSModel, conditions: dict[str, np.ndarray]) -> None:
    rng = np.random.default_rng(SEED_TEST)
    ics = sample_initial_states(N_TEST_TRAJ, rng)
    t = np.arange(N_STEPS_EVAL + 1) * DT

    fig, axes = plt.subplots(1, len(conditions), figsize=(6 * len(conditions), 4.5))
    for ax, (label, tau_seq) in zip(axes, conditions.items()):
        true_traj = true_rollout(ics, DT, N_STEPS_EVAL, c=DAMPING, tau_seq=tau_seq)
        pred_traj = model.rollout(ics, N_STEPS_EVAL, tau_seq=tau_seq)

        true_traj_t_first = np.transpose(true_traj, (1, 0, 2))
        e_true = energy(true_traj_t_first).mean(axis=1)
        e_pred = energy(pred_traj).mean(axis=1)

        ax.plot(t, e_true, label="真値")
        ax.plot(t, e_pred, label="NSSサロゲート")
        ax.axvline(50 * DT, color="gray", linestyle="--", label="学習ホライズン境界")
        ax.set_xlabel("time [s]")
        ax.set_ylabel("平均力学的エネルギー [J]")
        ax.set_title(label, fontsize=10)
        ax.legend(fontsize=8)

    fig.suptitle(f"エネルギー逸脱 (減衰系 c={DAMPING}, トルク条件別)")
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/energy_deviation.png", dpi=150)
    plt.close()
    print(f"saved {OUT_DIR}/energy_deviation.png")


def plot_phase_portrait(model: NSSModel, conditions: dict[str, np.ndarray]) -> None:
    # ゼロトルクは (0,0) が厳密な不動点で真値が1点に潰れて見えなくなるため、
    # 条件ごとに意味のある初期条件を使う(ランダム/共振はトルクで直ちに動き出すので(0,0)のままでよい)。
    default_ic = np.array([0.0, 0.0])
    ic_overrides = {"ゼロトルク (M2回帰確認)": np.array([0.3, 0.0])}

    fig, axes = plt.subplots(1, len(conditions), figsize=(6 * len(conditions), 5))
    for ax, (label, tau_seq) in zip(axes, conditions.items()):
        ic = ic_overrides.get(label, default_ic)
        true_traj = true_rollout(ic[None, :], DT, N_STEPS_EVAL, c=DAMPING, tau_seq=tau_seq)[0]
        pred_traj = model.rollout(ic[None, :], N_STEPS_EVAL, tau_seq=tau_seq)[:, 0, :]

        ax.plot(true_traj[:, 0], true_traj[:, 1], label="真値", linewidth=1.5)
        ax.plot(
            pred_traj[:, 0],
            pred_traj[:, 1],
            label="NSSサロゲート",
            linewidth=1.2,
            linestyle="--",
        )
        ax.set_xlabel("theta [rad]")
        ax.set_ylabel("theta_dot [rad/s]")
        ax.set_title(f"{label}\n(θ0={ic[0]:.1f}, θ̇0={ic[1]:.1f}から開始)", fontsize=10)
        ax.legend(fontsize=8)

    fig.suptitle("位相空間軌道の乖離(トルク条件別)")
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/phase_portrait.png", dpi=150)
    plt.close()
    print(f"saved {OUT_DIR}/phase_portrait.png")


if __name__ == "__main__":
    m = load_model()
    conds = make_control_conditions()
    curves = plot_error_curve(m, conds)
    plot_energy_deviation(m, conds)
    plot_phase_portrait(m, conds)
    for label, curve in curves.items():
        print(f"[{label}] final RMSE at t={N_STEPS_EVAL*DT:.1f}s: {curve[-1]:.4f}")
