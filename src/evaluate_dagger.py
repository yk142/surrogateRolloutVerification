"""M13: DAgger風再学習のBefore/After比較(#27)。

M12の角速度スイープとM5の18通りグリッドで、DAgger適用前(outputs/nss_model_
before_dagger.pt)・適用後(outputs/nss_model.pt)・真の物理モデルを比較する。
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

plt.rcParams["font.family"] = "Noto Sans CJK JP"

from src.diagnose_amplitude_sweep import DAMPING, N_STEPS, THETA0, THETA_DOT0_RANGE, WINDOW
from src.model import NSSModel
from src.pid import angular_error_to_inverted, run_pid_surrogate, run_pid_true

BEFORE_PATH = "outputs/nss_model_before_dagger.pt"
AFTER_PATH = "outputs/nss_model.pt"
OUT_DIR = "outputs"


def load(path: str) -> NSSModel:
    model = NSSModel()
    model.load_state_dict(torch.load(path, map_location="cpu"))
    model.eval()
    return model


def sweep_errors(model: NSSModel | None) -> np.ndarray:
    errors = []
    for v0 in THETA_DOT0_RANGE:
        ic = np.array([THETA0, -v0])
        if model is None:
            traj, _ = run_pid_true(ic, N_STEPS, c=DAMPING)
        else:
            traj, _ = run_pid_surrogate(model, ic, N_STEPS)
        errors.append(np.abs(angular_error_to_inverted(traj[-WINDOW:, 0])).mean())
    return np.array(errors)


if __name__ == "__main__":
    before_model = load(BEFORE_PATH)
    after_model = load(AFTER_PATH)

    true_err = sweep_errors(None)
    before_err = sweep_errors(before_model)
    after_err = sweep_errors(after_model)

    plt.figure(figsize=(7, 4.5))
    plt.plot(THETA_DOT0_RANGE, true_err, marker="o", label="真の物理モデル", markersize=4)
    plt.plot(THETA_DOT0_RANGE, before_err, marker="o", label="Before (M7, DAgger前)", markersize=4)
    plt.plot(THETA_DOT0_RANGE, after_err, marker="o", label="After (M13, DAgger後)", markersize=4)
    plt.axhline(0.1, color="red", linestyle=":", linewidth=1, label="成功しきい値 (0.1rad)")
    plt.xlabel("初期角速度の大きさ |θ̇0| [rad/s] (θ0=π-0.1から下向きに開始)")
    plt.ylabel("最終1秒間の平均角距離 |θ-π| [rad]")
    plt.title("DAgger再学習のBefore/After比較(M12の角速度スイープ)")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/dagger_before_after_sweep.png", dpi=150)
    plt.close()
    print(f"saved {OUT_DIR}/dagger_before_after_sweep.png")

    n_true = (true_err < 0.1).sum()
    n_before = (before_err < 0.1).sum()
    n_after = (after_err < 0.1).sum()
    print(f"成功数: 真値={n_true}/15, Before={n_before}/15, After={n_after}/15")
