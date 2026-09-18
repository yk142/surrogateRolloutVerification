"""NSSモデルの学習スクリプト(1-step teacher forcing MSE)。"""
import numpy as np
import torch
import torch.nn as nn

from src.dataset import generate_trajectories, make_transition_pairs
from src.model import NSSModel, encode_state

DT = 0.02
N_STEPS_PER_TRAJ = 50
N_TRAIN_TRAJ = 400
N_VAL_TRAJ = 50
SEED = 0
N_EPOCHS = 200
BATCH_SIZE = 512
LR = 1e-3


def train(device: str = "cpu") -> NSSModel:
    train_traj = generate_trajectories(N_TRAIN_TRAJ, DT, N_STEPS_PER_TRAJ, seed=SEED)
    val_traj = generate_trajectories(N_VAL_TRAJ, DT, N_STEPS_PER_TRAJ, seed=SEED + 1)

    x_train, x_next_train = make_transition_pairs(train_traj)
    x_val, x_next_val = make_transition_pairs(val_traj)

    x_train_t = torch.as_tensor(x_train, dtype=torch.float32, device=device)
    x_next_train_t = torch.as_tensor(x_next_train, dtype=torch.float32, device=device)
    x_val_t = torch.as_tensor(x_val, dtype=torch.float32, device=device)
    x_next_val_t = torch.as_tensor(x_next_val, dtype=torch.float32, device=device)

    target_train = encode_state(x_next_train_t)
    target_val = encode_state(x_next_val_t)

    model = NSSModel().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.MSELoss()

    n_samples = x_train_t.shape[0]

    for epoch in range(N_EPOCHS):
        perm = torch.randperm(n_samples)
        epoch_loss = 0.0
        for i in range(0, n_samples, BATCH_SIZE):
            idx = perm[i : i + BATCH_SIZE]
            pred = model(x_train_t[idx])
            loss = loss_fn(pred, target_train[idx])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * idx.shape[0]

        if epoch % 20 == 0 or epoch == N_EPOCHS - 1:
            with torch.no_grad():
                val_pred = model(x_val_t)
                val_loss = loss_fn(val_pred, target_val).item()
            print(
                f"epoch {epoch:4d} train_loss={epoch_loss / n_samples:.6f} "
                f"val_loss={val_loss:.6f}"
            )

    return model


if __name__ == "__main__":
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    trained_model = train()
    torch.save(trained_model.state_dict(), "outputs/nss_model.pt")
    print("saved to outputs/nss_model.pt")
