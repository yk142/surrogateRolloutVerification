"""学習済みNSSモデルのロールアウト誤差検証。3つのPNGを outputs/ に生成する。

1. rollout_error_curve.png : horizonに対するRMSE蓄積カーブ
2. energy_deviation.png    : エネルギー保存逸脱(真値 vs サロゲート)
3. phase_portrait.png      : 代表ICでの位相空間軌道比較
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

plt.rcParams["font.family"] = "Noto Sans CJK JP"

from src.dataset import sample_initial_states
from src.model import NSSModel
from src.physics import energy
from src.rollout import rmse_curve, true_rollout

DT = 0.02
N_STEPS_EVAL = 300  # 6秒分。学習時ホライズン(50ステップ=1秒)より大幅に長い
N_TEST_TRAJ = 200
SEED_TEST = 1000
DAMPING = 0.15  # train.py と揃える(M2)

MODEL_PATH = "outputs/nss_model.pt"
OUT_DIR = "outputs"


def load_model() -> NSSModel:
    model = NSSModel()
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    model.eval()
    return model


def plot_error_curve(model: NSSModel) -> None:
    rng = np.random.default_rng(SEED_TEST)
    ics = sample_initial_states(N_TEST_TRAJ, rng)

    true_traj = true_rollout(ics, DT, N_STEPS_EVAL, c=DAMPING)  # (n_traj, n_steps+1, 2)
    pred_traj = model.rollout(ics, N_STEPS_EVAL)  # (n_steps+1, n_traj, 2)

    true_traj_t_first = np.transpose(true_traj, (1, 0, 2))
    curve = rmse_curve(true_traj_t_first, pred_traj)

    t = np.arange(N_STEPS_EVAL + 1) * DT
    plt.figure(figsize=(6, 4))
    plt.plot(t, curve)
    plt.axvline(50 * DT, color="gray", linestyle="--", label="学習ホライズン境界")
    plt.xlabel("time [s]")
    plt.ylabel("RMSE (theta[rad], theta_dot[rad/s] 合成)")
    plt.title(f"ロールアウト誤差蓄積カーブ (n={N_TEST_TRAJ} test trajectories)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/rollout_error_curve.png", dpi=150)
    plt.close()
    print(f"saved {OUT_DIR}/rollout_error_curve.png")
    return curve


def plot_energy_deviation(model: NSSModel) -> None:
    rng = np.random.default_rng(SEED_TEST)
    ics = sample_initial_states(N_TEST_TRAJ, rng)

    true_traj = true_rollout(ics, DT, N_STEPS_EVAL, c=DAMPING)
    pred_traj = model.rollout(ics, N_STEPS_EVAL)

    true_traj_t_first = np.transpose(true_traj, (1, 0, 2))
    e_true = energy(true_traj_t_first).mean(axis=1)
    e_pred = energy(pred_traj).mean(axis=1)

    t = np.arange(N_STEPS_EVAL + 1) * DT
    plt.figure(figsize=(6, 4))
    plt.plot(t, e_true, label="真値(減衰により単調減少するはず)")
    plt.plot(t, e_pred, label="NSSサロゲート")
    plt.axvline(50 * DT, color="gray", linestyle="--", label="学習ホライズン境界")
    plt.xlabel("time [s]")
    plt.ylabel("平均力学的エネルギー [J] (m=1kg換算)")
    plt.title(f"エネルギー逸脱 (減衰系 c={DAMPING}, 真値からのズレ)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/energy_deviation.png", dpi=150)
    plt.close()
    print(f"saved {OUT_DIR}/energy_deviation.png")


def plot_phase_portrait(model: NSSModel) -> None:
    representative_ics = {
        "小振幅振動 (θ0=0.3, θ̇0=0)": np.array([0.3, 0.0]),
        "完全回転域 (θ0=0, θ̇0=7.0)": np.array([0.0, 7.0]),
        "倒立近傍 (θ0=π-0.3, θ̇0=0.3)": np.array([np.pi - 0.3, 0.3]),
    }

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, (label, ic) in zip(axes, representative_ics.items()):
        true_traj = true_rollout(ic[None, :], DT, N_STEPS_EVAL, c=DAMPING)[0]
        pred_traj = model.rollout(ic[None, :], N_STEPS_EVAL)[:, 0, :]

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
        ax.set_title(label)
        ax.legend(fontsize=8)

    fig.suptitle("位相空間軌道の乖離(代表初期条件)")
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/phase_portrait.png", dpi=150)
    plt.close()
    print(f"saved {OUT_DIR}/phase_portrait.png")


if __name__ == "__main__":
    m = load_model()
    curve = plot_error_curve(m)
    plot_energy_deviation(m)
    plot_phase_portrait(m)
    print(f"final RMSE at t={N_STEPS_EVAL*DT:.1f}s: {curve[-1]:.4f}")
