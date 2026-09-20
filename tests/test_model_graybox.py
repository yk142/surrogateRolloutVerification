import numpy as np
import torch

from src.model import GrayBoxNSSModel
from src.physics import L_NOMINAL, M_NOMINAL, rk4_step


def _zero_residual_model() -> GrayBoxNSSModel:
    """残差NNの出力を恒等的に0にしたグレーボックスモデル。"""
    model = GrayBoxNSSModel()
    final_layer = model.net[-1]
    torch.nn.init.zeros_(final_layer.weight)
    torch.nn.init.zeros_(final_layer.bias)
    return model


def test_zero_residual_matches_known_physics():
    """残差が0なら、グレーボックスの1ステップは既知物理(無摩擦RK4)と一致すること。

    既知構造(重力項・トルク入力項・RK4積分)がハードコードされていることの検証。
    グレーボックスが信じている公称パラメータ(L_NOMINAL, M_NOMINAL)で比較する
    (真の系のL, Mとはキャリブレーション誤差がある: M16 #33)。
    """
    model = _zero_residual_model()
    dt = model.dt

    for theta, theta_dot, tau in [(0.3, 0.0, 0.0), (np.pi - 0.2, -1.0, 2.5), (2.0, 4.0, -3.0)]:
        state = torch.tensor([theta, theta_dot], dtype=torch.float32)
        u = torch.tensor([tau], dtype=torch.float32)

        with torch.no_grad():
            predicted = model.step(state, u).numpy()
        expected = rk4_step(
            np.array([theta, theta_dot]), dt, c=0.0, tau=tau,
            m=M_NOMINAL, c_coulomb=0.0, length=L_NOMINAL,
        )

        assert np.allclose(predicted, expected, atol=1e-5)


def test_theta_is_not_wrapped():
    """グレーボックスはatan2で畳み込まないため、thetaが連続に伸びること。"""
    model = _zero_residual_model()
    traj = model.rollout(np.array([np.pi - 0.05, 5.0]), n_steps=50)
    assert traj[:, 0].max() > np.pi
