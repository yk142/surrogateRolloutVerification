"""M14: ブラックボックス版とグレーボックス版の比較(#29)。

同一の学習レシピ(M7のカリキュラム、DAggerなし)で両アーキテクチャを学習し、
アーキテクチャの違いだけを切り分けて比較する。

比較指標:
1. M3の3トルク条件での開ループロールアウトRMSE(t=6s時点)
2. M12の角速度スイープ(PID回復挙動が真値の失敗パターンと一致するか)
3. M5の18通りグリッドでのPID安定化成功数
4. (M15〜) グレーボックスが学習した残差 vs 真の摩擦特性
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

plt.rcParams["font.family"] = "Noto Sans CJK JP"

from src.dataset import sample_initial_states
from src.diagnose_amplitude_sweep import N_STEPS as SWEEP_N_STEPS
from src.diagnose_amplitude_sweep import THETA0 as SWEEP_THETA0
from src.diagnose_amplitude_sweep import THETA_DOT0_RANGE, WINDOW
from src.evaluate import (
    DT,
    N_STEPS_EVAL,
    N_TEST_TRAJ,
    SEED_TEST,
    make_control_conditions,
)
from src.evaluate_pid import ANGLE_OFFSETS, VELOCITY_OFFSETS
from src.model import AutoregressiveModel, GrayBoxNSSModel, NSSModel
from src.physics import C_COULOMB, friction_acceleration
from src.pid import angular_error_to_inverted, run_pid_surrogate, run_pid_true
from src.rollout import rmse_curve, true_rollout
from src.train import CURRICULUM, DAMPING, SEED, train

SUCCESS_THRESHOLD = 0.1  # M5/M8/M9/M12と同じ
OUT_DIR = "outputs"


def open_loop_rmse(model: AutoregressiveModel) -> dict[str, float]:
    """M3の3トルク条件での、t=6s時点のロールアウトRMSE。"""
    rng = np.random.default_rng(SEED_TEST)
    ics = sample_initial_states(N_TEST_TRAJ, rng)

    results = {}
    for label, tau_seq in make_control_conditions().items():
        true_traj = true_rollout(ics, DT, N_STEPS_EVAL, c=DAMPING, tau_seq=tau_seq)
        pred_traj = model.rollout(ics, N_STEPS_EVAL, tau_seq=tau_seq)
        curve = rmse_curve(np.transpose(true_traj, (1, 0, 2)), pred_traj)
        results[label] = float(curve[-1])
    return results


def sweep_errors(model: AutoregressiveModel | None) -> np.ndarray:
    """M12の角速度スイープ。model=Noneなら真の物理モデル。"""
    errors = []
    for v0 in THETA_DOT0_RANGE:
        ic = np.array([SWEEP_THETA0, -v0])
        if model is None:
            traj, _ = run_pid_true(ic, SWEEP_N_STEPS, c=DAMPING)
        else:
            traj, _ = run_pid_surrogate(model, ic, SWEEP_N_STEPS)
        errors.append(np.abs(angular_error_to_inverted(traj[-WINDOW:, 0])).mean())
    return np.array(errors)


def grid_errors(model: AutoregressiveModel | None) -> np.ndarray:
    """M5の18通りグリッド。model=Noneなら真の物理モデル。"""
    errors = []
    for angle_offset in ANGLE_OFFSETS:
        for velocity_offset in VELOCITY_OFFSETS:
            ic = np.array([np.pi + angle_offset, velocity_offset])
            if model is None:
                traj, _ = run_pid_true(ic, SWEEP_N_STEPS, c=DAMPING)
            else:
                traj, _ = run_pid_surrogate(model, ic, SWEEP_N_STEPS)
            errors.append(np.abs(angular_error_to_inverted(traj[-50:, 0])).mean())
    return np.array(errors)


def plot_comparison(
    true_sweep: np.ndarray,
    blackbox_sweep: np.ndarray,
    graybox_sweep: np.ndarray,
    blackbox_rmse: dict[str, float],
    graybox_rmse: dict[str, float],
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

    axes[0].plot(THETA_DOT0_RANGE, true_sweep, marker="o", markersize=4, label="真の物理モデル")
    axes[0].plot(
        THETA_DOT0_RANGE, blackbox_sweep, marker="o", markersize=4, label="ブラックボックス"
    )
    axes[0].plot(
        THETA_DOT0_RANGE, graybox_sweep, marker="o", markersize=4, label="グレーボックス"
    )
    axes[0].axhline(SUCCESS_THRESHOLD, color="red", linestyle=":", linewidth=1, label="成功しきい値")
    axes[0].set_xlabel("初期角速度の大きさ |θ̇0| [rad/s]")
    axes[0].set_ylabel("最終1秒間の平均角距離 |θ-π| [rad]")
    axes[0].set_title("PID回復挙動の真値との一致度(M12スイープ)", fontsize=10)
    axes[0].legend(fontsize=8)

    labels = list(blackbox_rmse.keys())
    x = np.arange(len(labels))
    width = 0.35
    axes[1].bar(x - width / 2, [blackbox_rmse[k] for k in labels], width, label="ブラックボックス")
    axes[1].bar(x + width / 2, [graybox_rmse[k] for k in labels], width, label="グレーボックス")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([k.replace(" (", "\n(") for k in labels], fontsize=8)
    axes[1].set_ylabel("t=6s時点のRMSE")
    axes[1].set_title("開ループロールアウト誤差(M3の3条件)", fontsize=10)
    axes[1].legend(fontsize=8)

    fig.suptitle("M14: ブラックボックス vs グレーボックス(既知物理+残差学習)")
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/graybox_comparison.png", dpi=150)
    plt.close()
    print(f"saved {OUT_DIR}/graybox_comparison.png")


def plot_learned_residual(graybox: GrayBoxNSSModel) -> None:
    """グレーボックスが学習した残差加速度と、真の摩擦特性を重ねて比較する。

    真の未知項は粘性摩擦 + クーロン摩擦 = -c*theta_dot - c_coulomb*sign(theta_dot) で、
    theta_dot=0 で不連続。NNがこの不連続をどこまで再現できるかを見る。
    """
    theta_dot_grid = np.linspace(-6.0, 6.0, 601)
    truth = friction_acceleration(theta_dot_grid, DAMPING, C_COULOMB)

    plt.figure(figsize=(7, 4.5))
    plt.plot(theta_dot_grid, truth, label="真の摩擦特性", linewidth=2)
    for theta, tau in [(0.0, 0.0), (np.pi, 0.0), (1.5, 3.0)]:
        states = np.stack(
            [np.full_like(theta_dot_grid, theta), theta_dot_grid], axis=-1
        )
        with torch.no_grad():
            learned = graybox.residual_acceleration(
                torch.as_tensor(states, dtype=torch.float32),
                torch.full((len(theta_dot_grid), 1), tau, dtype=torch.float32),
            ).numpy()
        plt.plot(theta_dot_grid, learned, "--", linewidth=1,
                 label=f"学習された残差 (θ={theta:.1f}, τ={tau:.0f})")

    plt.xlabel("theta_dot [rad/s]")
    plt.ylabel("残差加速度 [rad/s²]")
    plt.title("グレーボックスが学習した残差 vs 真の摩擦特性(粘性+クーロン)")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/graybox_residual.png", dpi=150)
    plt.close()
    print(f"saved {OUT_DIR}/graybox_residual.png")


if __name__ == "__main__":
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    blackbox = train(curriculum=CURRICULUM, model_cls=NSSModel)

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    graybox = train(curriculum=CURRICULUM, model_cls=GrayBoxNSSModel)

    blackbox_rmse = open_loop_rmse(blackbox)
    graybox_rmse = open_loop_rmse(graybox)

    true_sweep = sweep_errors(None)
    blackbox_sweep = sweep_errors(blackbox)
    graybox_sweep = sweep_errors(graybox)

    true_grid = grid_errors(None)
    blackbox_grid = grid_errors(blackbox)
    graybox_grid = grid_errors(graybox)

    plot_comparison(true_sweep, blackbox_sweep, graybox_sweep, blackbox_rmse, graybox_rmse)
    plot_learned_residual(graybox)

    print("\n=== 1. 開ループロールアウトRMSE (t=6s) ===")
    for label in blackbox_rmse:
        print(f"{label}: ブラックボックス={blackbox_rmse[label]:.4f} "
              f"グレーボックス={graybox_rmse[label]:.4f}")

    print("\n=== 2. M12角速度スイープ (n=15) ===")
    for name, err in [("真値", true_sweep), ("ブラックボックス", blackbox_sweep),
                      ("グレーボックス", graybox_sweep)]:
        print(f"{name}: 成功数={(err < SUCCESS_THRESHOLD).sum()}/{len(err)}")
    print("真値の成功/失敗パターンとの一致数: "
          f"ブラックボックス={((blackbox_sweep < SUCCESS_THRESHOLD) == (true_sweep < SUCCESS_THRESHOLD)).sum()}/15, "
          f"グレーボックス={((graybox_sweep < SUCCESS_THRESHOLD) == (true_sweep < SUCCESS_THRESHOLD)).sum()}/15")

    print("\n=== 3. M5の18通りグリッド ===")
    for name, err in [("真値", true_grid), ("ブラックボックス", blackbox_grid),
                      ("グレーボックス", graybox_grid)]:
        print(f"{name}: 成功数={(err < SUCCESS_THRESHOLD).sum()}/{len(err)}")
    print("真値の成功/失敗パターンとの一致数: "
          f"ブラックボックス={((blackbox_grid < SUCCESS_THRESHOLD) == (true_grid < SUCCESS_THRESHOLD)).sum()}/18, "
          f"グレーボックス={((graybox_grid < SUCCESS_THRESHOLD) == (true_grid < SUCCESS_THRESHOLD)).sum()}/18")
