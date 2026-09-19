"""大振幅域でのPID回復挙動を1-step/マルチステップで定量比較する(#25)。

M11 (#23) で、M7(マルチステップ損失)の制御タスクでのリグレッションは、単一の
代表IC(theta0=pi-0.1, theta_dot0=-1.0、最下点を通過して落下→再上昇する軌道)で
確認された。当初は無制御(tau=0)の純粋ロールアウト精度で汎化を確認しようとしたが、
8秒間の無制御ロールアウトは真値自体もカオス的に減衰・振動するため、両モデルとも
ノイジーな指標になり境界を特定できなかった。そこでM11と同じPID閉ループ設定
(run_pid_true/run_pid_surrogate)を使い、初期角速度をスイープして
「最下点まで落下した後に倒立へ回復できるか」を定量比較する。
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

plt.rcParams["font.family"] = "Noto Sans CJK JP"

from src.model import NSSModel
from src.pid import angular_error_to_inverted, run_pid_surrogate, run_pid_true
from src.train import CURRICULUM, DAMPING, SEED, train

THETA0 = np.pi - 0.1  # M11の代表ICと同じ開始角度
THETA_DOT0_RANGE = np.linspace(0.2, 3.0, 15)  # 小さな乱れ→最下点通過を伴う大振幅まで
N_STEPS = 400  # 8秒分(M11と同じ)
WINDOW = 50  # 最終1秒間で安定化誤差を評価(M5/M8/M9と同じ)
BASELINE_CURRICULUM = [(1, 200)]  # M3相当

OUT_DIR = "outputs"


def run_sweep(baseline_model: NSSModel, multistep_model: NSSModel) -> dict[str, np.ndarray]:
    results = {"真の物理モデル": [], "1-stepのみ (M3相当)": [], "マルチステップ (M7)": []}

    for v0 in THETA_DOT0_RANGE:
        ic = np.array([THETA0, -v0])
        true_traj, _ = run_pid_true(ic, N_STEPS, c=DAMPING)
        base_traj, _ = run_pid_surrogate(baseline_model, ic, N_STEPS)
        multi_traj, _ = run_pid_surrogate(multistep_model, ic, N_STEPS)

        results["真の物理モデル"].append(
            np.abs(angular_error_to_inverted(true_traj[-WINDOW:, 0])).mean()
        )
        results["1-stepのみ (M3相当)"].append(
            np.abs(angular_error_to_inverted(base_traj[-WINDOW:, 0])).mean()
        )
        results["マルチステップ (M7)"].append(
            np.abs(angular_error_to_inverted(multi_traj[-WINDOW:, 0])).mean()
        )
        print(
            f"theta_dot0=-{v0:.2f}: 真値={results['真の物理モデル'][-1]:.4f} "
            f"1-step={results['1-stepのみ (M3相当)'][-1]:.4f} "
            f"マルチステップ={results['マルチステップ (M7)'][-1]:.4f}"
        )

    return {k: np.array(v) for k, v in results.items()}


def plot_sweep(results: dict[str, np.ndarray]) -> None:
    plt.figure(figsize=(7, 4.5))
    for label, err in results.items():
        plt.plot(THETA_DOT0_RANGE, err, marker="o", label=label, markersize=4)
    plt.axvline(1.0, color="gray", linestyle="--", linewidth=1, label="M11の代表IC (θ̇0=1.0)")
    plt.axhline(0.1, color="red", linestyle=":", linewidth=1, label="成功しきい値 (M5/M8/M9と同じ0.1rad)")
    plt.xlabel("初期角速度の大きさ |θ̇0| [rad/s] (θ0=π-0.1から下向きに開始)")
    plt.ylabel("最終1秒間の平均角距離 |θ-π| [rad]")
    plt.title("大振幅域でのPID回復挙動: 1-step vs マルチステップ")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/amplitude_sweep_rmse.png", dpi=150)
    plt.close()
    print(f"saved {OUT_DIR}/amplitude_sweep_rmse.png")


if __name__ == "__main__":
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    baseline_model = train(curriculum=BASELINE_CURRICULUM, model_cls=NSSModel)

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    multistep_model = train(curriculum=CURRICULUM, model_cls=NSSModel)

    results = run_sweep(baseline_model, multistep_model)
    plot_sweep(results)

    for label in ["真の物理モデル", "1-stepのみ (M3相当)", "マルチステップ (M7)"]:
        n_success = (results[label] < 0.1).sum()
        print(f"[{label}] 成功数: {n_success}/{len(THETA_DOT0_RANGE)}")
