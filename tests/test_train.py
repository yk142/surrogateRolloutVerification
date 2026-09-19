import numpy as np
import torch

from src.dataset import generate_controlled_trajectories, make_rollout_windows
from src.model import NSSModel


def test_make_rollout_windows_shapes():
    trajectories, tau_seqs = generate_controlled_trajectories(5, dt=0.02, n_steps=20, seed=0)
    k = 4
    x0, u_seq, targets = make_rollout_windows(trajectories, tau_seqs, k, stride=2)

    n_starts = (20 - k) // 2 + 1
    n_windows = n_starts * 5
    assert x0.shape == (n_windows, 2)
    assert u_seq.shape == (k, n_windows)
    assert targets.shape == (k, n_windows, 2)


def test_rollout_diff_is_differentiable():
    """マルチステップ損失で使うrollout_diffが勾配を保持していること。"""
    model = NSSModel()
    state = torch.zeros(3, 2, requires_grad=False)
    tau_seq = torch.zeros(5, 3, requires_grad=True)

    traj = model.rollout_diff(state, tau_seq, n_steps=5)
    assert traj.shape == (6, 3, 2)

    loss = traj[1:].pow(2).sum()
    loss.backward()
    assert tau_seq.grad is not None
    assert torch.isfinite(tau_seq.grad).all()
