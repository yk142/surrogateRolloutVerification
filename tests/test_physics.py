import numpy as np

from src.physics import energy, simulate

C_DAMPED = 0.2


def test_energy_conservation_undamped():
    """無減衰系ではRK4積分中のエネルギー変動が十分小さいこと。"""
    initial_state = np.array([1.0, 0.0])  # theta=1rad, theta_dot=0
    dt = 0.01
    n_steps = 2000  # 20秒分
    traj = simulate(initial_state, dt, n_steps, c=0.0, c_coulomb=0.0)

    e = energy(traj)
    e0 = e[0]
    max_rel_drift = np.max(np.abs(e - e0)) / e0

    assert max_rel_drift < 1e-3


def test_stable_equilibrium_is_fixed_point():
    """theta=0, theta_dot=0 は不動点であること。"""
    initial_state = np.array([0.0, 0.0])
    traj = simulate(initial_state, dt=0.01, n_steps=100, c=0.0, c_coulomb=0.0)
    assert np.allclose(traj, 0.0)


def test_energy_monotonically_decreases_when_damped():
    """減衰系(c>0)ではエネルギーが単調に(非増加に)減少すること。"""
    initial_state = np.array([2.0, 1.0])
    dt = 0.01
    n_steps = 2000
    traj = simulate(initial_state, dt, n_steps, c=C_DAMPED, c_coulomb=0.0)

    e = energy(traj)
    assert np.all(np.diff(e) <= 1e-9)
    assert e[-1] < e[0]


def test_constant_torque_matches_work_energy_theorem():
    """無減衰系で一定トルクtauを与えたとき、エネルギー変化量が仕事 tau*Δtheta と一致すること。

    (m=L=1のとき dE/dt = tau*theta_dot なので、tauが一定なら ΔE = tau*Δtheta)
    """
    initial_state = np.array([0.3, 0.0])
    dt = 0.005
    n_steps = 400
    tau = 1.5
    traj = simulate(initial_state, dt, n_steps, c=0.0, tau=tau, c_coulomb=0.0)

    e = energy(traj)
    delta_e = e[-1] - e[0]
    delta_theta = traj[-1, 0] - traj[0, 0]
    expected_delta_e = tau * delta_theta

    assert np.isclose(delta_e, expected_delta_e, rtol=1e-2)
