import numpy as np

from src.dagger import collect_dagger_windows
from src.model import NSSModel


def test_collect_dagger_windows_shapes():
    model = NSSModel()
    ics = np.array([[np.pi - 0.1, -1.0], [np.pi + 0.2, 0.5]])
    k = 5
    n_steps = 20
    stride = 5

    x0, u_seq, targets = collect_dagger_windows(model, ics, n_steps=n_steps, k=k, c=0.15, stride=stride)

    n_starts = (n_steps - k) // stride + 1
    n_windows = n_starts * len(ics)
    assert x0.shape == (n_windows, 2)
    assert u_seq.shape == (k, n_windows)
    assert targets.shape == (k, n_windows, 2)
