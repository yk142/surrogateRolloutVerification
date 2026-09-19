"""NSSモデルの学習スクリプト。

M1-M6は1-step教師強制MSEのみで学習していたが、実際の利用(ロールアウト評価・
MPC・PID)は自己回帰的な多段階予測であり、1-step誤差が学習時に見えない形で
蓄積することが繰り返し観察された。M7ではKステップ分自己回帰展開してから
損失を取る「マルチステップ(ロールアウト)損失」に切り替える。

Kをいきなり大きくして最初から学習すると(from-scratch)、勾配が不安定になり
1-step版より悪化することが実験で判明した。そのため K=1 から徐々に増やす
カリキュラム学習を採用する(K=1で1-step相当の学習を先に安定させてから、
段階的にロールアウトホライズンを伸ばして微調整する)。
"""
import numpy as np
import torch
import torch.nn as nn

from src.dataset import generate_controlled_trajectories, make_rollout_windows
from src.model import NSSModel, encode_state

DT = 0.02
N_STEPS_PER_TRAJ = 50
N_TRAIN_TRAJ = 600
N_VAL_TRAJ = 80
SEED = 0
BATCH_SIZE = 512
LR = 1e-3
GRAD_CLIP_NORM = 1.0
DAMPING = 0.15  # M2: 減衰項を追加(M1は0.0)

K_MAX = 30  # 学習時の最終ロールアウトホライズン(0.6秒)
WINDOW_STRIDE = 2

# M7: カリキュラム学習のスケジュール (k, そのkでのエポック数)。
# K=1で1-step相当の学習を先に安定させてから段階的にKを伸ばす。
CURRICULUM = [(1, 40), (3, 30), (5, 30), (10, 30), (20, 30), (K_MAX, 40)]


def train(device: str = "cpu") -> NSSModel:
    train_traj, train_tau = generate_controlled_trajectories(
        N_TRAIN_TRAJ, DT, N_STEPS_PER_TRAJ, seed=SEED, c=DAMPING
    )
    val_traj, val_tau = generate_controlled_trajectories(
        N_VAL_TRAJ, DT, N_STEPS_PER_TRAJ, seed=SEED + 1, c=DAMPING
    )

    x0_train, u_train, targets_train = make_rollout_windows(
        train_traj, train_tau, K_MAX, stride=WINDOW_STRIDE
    )
    x0_val, u_val, targets_val = make_rollout_windows(
        val_traj, val_tau, K_MAX, stride=WINDOW_STRIDE
    )

    x0_train_t = torch.as_tensor(x0_train, dtype=torch.float32, device=device)
    u_train_t = torch.as_tensor(u_train, dtype=torch.float32, device=device)
    target_enc_train = encode_state(
        torch.as_tensor(targets_train, dtype=torch.float32, device=device)
    )

    x0_val_t = torch.as_tensor(x0_val, dtype=torch.float32, device=device)
    u_val_t = torch.as_tensor(u_val, dtype=torch.float32, device=device)
    target_enc_val = encode_state(
        torch.as_tensor(targets_val, dtype=torch.float32, device=device)
    )

    model = NSSModel().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.MSELoss()

    n_samples = x0_train_t.shape[0]
    global_epoch = 0

    for k, n_epochs in CURRICULUM:
        for local_epoch in range(n_epochs):
            perm = torch.randperm(n_samples)
            epoch_loss = 0.0
            for i in range(0, n_samples, BATCH_SIZE):
                idx = perm[i : i + BATCH_SIZE]
                pred_traj = model.rollout_diff(x0_train_t[idx], u_train_t[:k, idx], k)
                pred_enc = encode_state(pred_traj[1:])
                loss = loss_fn(pred_enc, target_enc_train[:k, idx, :])

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
                optimizer.step()
                epoch_loss += loss.item() * idx.shape[0]

            global_epoch += 1
            if local_epoch % 10 == 0 or local_epoch == n_epochs - 1:
                with torch.no_grad():
                    val_pred_traj = model.rollout_diff(x0_val_t, u_val_t[:k], k)
                    val_pred_enc = encode_state(val_pred_traj[1:])
                    val_loss = loss_fn(val_pred_enc, target_enc_val[:k]).item()
                print(
                    f"k={k:2d} epoch {global_epoch:4d} "
                    f"train_loss={epoch_loss / n_samples:.6f} val_loss={val_loss:.6f}"
                )

    return model


if __name__ == "__main__":
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    trained_model = train()
    torch.save(trained_model.state_dict(), "outputs/nss_model.pt")
    print("saved to outputs/nss_model.pt")
