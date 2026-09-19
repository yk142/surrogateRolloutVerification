"""M7リグレッションの根本原因調査(#23)。

M8/M9で、マルチステップ損失モデル(M7)は自由振動ロールアウトの平均指標では
一部改善したにもかかわらず、PID/MPC/PTPといった閉ループ制御タスクでは悪化・
不安定化した(特にM4ではばらつきが激増)。仮説: マルチステップ損失は多くの
ステップにわたる累積誤差を最小化しようとするため、倒立平衡点まわりの局所的な
ダイナミクス(実効的なヤコビアン=線形化)を真の物理系とは異なる方向に歪めており、
これが局所線形化に敏感な閉ループ制御の不安定化を説明しうる。

この仮説を検証するため、倒立平衡点(theta=pi, theta_dot=0, tau=0)における
離散時間ヤコビアン(A行列: 状態に対する感度、B行列: トルクに対する感度)を
- 真の物理モデル(有限差分)
- 1-stepのみで学習したベースラインモデル(M3相当)
- M7と同じレシピ(カリキュラム学習)のマルチステップ損失モデル
の3種類で計算し、固有値(離散時間の極)を比較する。
"""
import numpy as np
import torch

from src.model import NSSModel
from src.physics import rk4_step
from src.pid import angular_error_to_inverted, run_pid_surrogate, run_pid_true
from src.train import CURRICULUM, DAMPING, DT, SEED, train

EQUILIBRIUM_STATE = np.array([np.pi, 0.0])
EQUILIBRIUM_TAU = 0.0

BASELINE_CURRICULUM = [(1, 200)]  # M3相当: 1-stepのみ、M3と同じエポック数

# M8 (#17) で新規リグレッションが見つかったIC。平衡点そのものではなく、
# 平衡点から大きく外れた状態からの「回復挙動」を比較するために使う。
REGRESSION_IC = np.array([np.pi - 0.1, -1.0])
REGRESSION_N_STEPS = 400
REGRESSION_WINDOW = 50


def true_jacobian() -> tuple[np.ndarray, np.ndarray]:
    """真の物理モデル(rk4_step)を有限差分で線形化する。"""
    eps = 1e-6

    def f(state: np.ndarray, tau: float) -> np.ndarray:
        return rk4_step(state, DT, c=DAMPING, tau=tau)

    f0 = f(EQUILIBRIUM_STATE, EQUILIBRIUM_TAU)
    A = np.zeros((2, 2))
    for i in range(2):
        perturbed = EQUILIBRIUM_STATE.copy()
        perturbed[i] += eps
        A[:, i] = (f(perturbed, EQUILIBRIUM_TAU) - f0) / eps

    B = (f(EQUILIBRIUM_STATE, EQUILIBRIUM_TAU + eps) - f0) / eps
    return A, B


def model_jacobian(model: NSSModel) -> tuple[np.ndarray, np.ndarray]:
    """NSSモデル(model.step)を自動微分で線形化する。"""
    state = torch.tensor(EQUILIBRIUM_STATE, dtype=torch.float32, requires_grad=True)
    tau = torch.tensor([EQUILIBRIUM_TAU], dtype=torch.float32, requires_grad=True)

    next_state = model.step(state, tau)

    A = np.zeros((2, 2))
    B = np.zeros(2)
    for i in range(2):
        grad_state, grad_tau = torch.autograd.grad(
            next_state[i], [state, tau], retain_graph=True
        )
        A[i, :] = grad_state.detach().numpy()
        B[i] = grad_tau.detach().numpy()[0]
    return A, B


def report(label: str, A: np.ndarray, B: np.ndarray, A_true: np.ndarray) -> None:
    eigvals = np.linalg.eigvals(A)
    diff_norm = np.linalg.norm(A - A_true)
    print(f"--- {label} ---")
    print(f"A =\n{A}")
    print(f"B = {B}")
    print(f"eigenvalues(A) = {eigvals}")
    print(f"||A - A_true||_F = {diff_norm:.6f}")
    print()


def compare_regression_ic(baseline_model: NSSModel, multistep_model: NSSModel) -> None:
    """平衡点から大きく外れたIC(M8のリグレッション事例)での回復挙動を比較する。

    局所線形化(倒立平衡点そのもの)が真値とほぼ一致していても、大きく外れた
    状態からの「落下→再上昇」の過程で挙動が乖離しうることを確認する。
    """
    true_traj, _ = run_pid_true(REGRESSION_IC, REGRESSION_N_STEPS, c=DAMPING)
    base_traj, _ = run_pid_surrogate(baseline_model, REGRESSION_IC, REGRESSION_N_STEPS)
    multi_traj, _ = run_pid_surrogate(multistep_model, REGRESSION_IC, REGRESSION_N_STEPS)

    print(f"--- 平衡点から外れたIC (theta0={REGRESSION_IC[0]:.3f}, theta_dot0={REGRESSION_IC[1]:.1f}) ---")
    for label, traj in [("真値", true_traj), ("1-stepのみ", base_traj), ("マルチステップ", multi_traj)]:
        final_err = np.abs(angular_error_to_inverted(traj[-REGRESSION_WINDOW:, 0])).mean()
        checkpoints = traj[[0, 50, 100, 200, 400], 0]
        print(f"{label}: 最終誤差={final_err:.4f}, theta(t=0,1,2,4,8s)={checkpoints}")
    print()


if __name__ == "__main__":
    A_true, B_true = true_jacobian()

    print("真の物理モデル(倒立平衡点での離散時間線形化):")
    report("真の物理モデル", A_true, B_true, A_true)

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    baseline_model = train(curriculum=BASELINE_CURRICULUM, model_cls=NSSModel)
    A_baseline, B_baseline = model_jacobian(baseline_model)
    report("1-stepのみ(M3相当ベースライン)", A_baseline, B_baseline, A_true)

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    multistep_model = train(curriculum=CURRICULUM, model_cls=NSSModel)
    A_multistep, B_multistep = model_jacobian(multistep_model)
    report("マルチステップ損失(M7)", A_multistep, B_multistep, A_true)

    print("真値の固有値:", np.linalg.eigvals(A_true))
    print(
        f"||A_baseline - A_true||_F = {np.linalg.norm(A_baseline - A_true):.6f} vs "
        f"||A_multistep - A_true||_F = {np.linalg.norm(A_multistep - A_true):.6f}"
    )
    print()

    compare_regression_ic(baseline_model, multistep_model)
