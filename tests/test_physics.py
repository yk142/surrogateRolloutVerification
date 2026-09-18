import numpy as np

from src.physics import energy, simulate


def test_energy_conservation_undamped():
    """無減衰系ではRK4積分中のエネルギー変動が十分小さいこと。"""
    initial_state = np.array([1.0, 0.0])  # theta=1rad, theta_dot=0
    dt = 0.01
    n_steps = 2000  # 20秒分
    traj = simulate(initial_state, dt, n_steps)

    e = energy(traj)
    e0 = e[0]
    max_rel_drift = np.max(np.abs(e - e0)) / e0

    assert max_rel_drift < 1e-3


def test_stable_equilibrium_is_fixed_point():
    """theta=0, theta_dot=0 は不動点であること。"""
    initial_state = np.array([0.0, 0.0])
    traj = simulate(initial_state, dt=0.01, n_steps=100)
    assert np.allclose(traj, 0.0)
