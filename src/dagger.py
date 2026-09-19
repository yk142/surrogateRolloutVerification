"""オンポリシーデータ収集(DAgger風)によるサロゲート改善(#27)。

M8/M9/M11/M12で、サロゲートの誤差は「訓練時のランダム軌道分布」と「実際の
制御ループ(PID)で訪れる状態分布」のズレ(distributional shift)に起因することが
繰り返し示唆された。ここでは、現在のサロゲート+PIDを実際に閉ループさせて訪れた
状態・トルク系列を収集し、その各時点から真の物理モデルでKステップ先まで
ロールアウトした正解ラベルを付けることで、通常の学習ウィンドウと同じ形式
(x0, u_seq, targets)の訓練データを作る。
"""
import numpy as np

from src.model import NSSModel
from src.physics import simulate
from src.pid import run_pid_surrogate

DT = 0.02


def collect_dagger_windows(
    model: NSSModel,
    ics: np.ndarray,
    n_steps: int,
    k: int,
    c: float,
    stride: int = 5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """サロゲート+PIDの閉ループ軌道を訪れ、真の物理モデルでラベル付けする。

    ics: shape (n_ic, 2)。各ICについてサロゲート+PIDをn_stepsステップ実行し、
    訪れた状態・適用したトルクから stride ごとにKステップの学習ウィンドウを切り出す。
    正解ラベル(targets)は、訪れた状態からサロゲートが実際に適用したトルク系列を
    そのまま使って真の物理モデルでKステップ先までロールアウトしたもの。

    Returns: make_rollout_windows と同じ形式
        x0:      shape (n_windows, 2)
        u_seq:   shape (k, n_windows)
        targets: shape (k, n_windows, 2)
    """
    x0_parts, u_parts, target_parts = [], [], []

    for ic in ics:
        traj, taus = run_pid_surrogate(model, ic, n_steps)  # traj:(n_steps+1,2), taus:(n_steps,)
        for start in range(0, n_steps - k + 1, stride):
            x0 = traj[start]
            u_seq = taus[start : start + k]
            true_continuation = simulate(x0, DT, k, c=c, tau=u_seq)  # (k+1, 2)
            x0_parts.append(x0)
            u_parts.append(u_seq)
            target_parts.append(true_continuation[1:])

    x0 = np.stack(x0_parts, axis=0)
    u_seq = np.stack(u_parts, axis=0).T
    targets = np.stack(target_parts, axis=0).transpose(1, 0, 2)
    return x0, u_seq, targets
