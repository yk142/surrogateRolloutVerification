"""Neural State Space (NSS) サロゲートモデル。

2種類のアーキテクチャを提供する:

- `NSSModel` (ブラックボックス、M1-M13): 離散時間の状態遷移 x_{t+1} = f_theta(x_t, u_t)
  をMLPで丸ごと学習する。theta の周期性を扱うため (sin theta, cos theta, theta_dot) に
  エンコードし、出力も同じ表現で予測してから theta = atan2(sin, cos) にデコードする。
- `GrayBoxNSSModel` (グレーボックス、M14〜): 既知の物理構造(重力項・トルク入力項・
  RK4積分)をハードコードし、NNは未知の残差加速度(摩擦など)のみを学習する。

どちらも step / rollout / rollout_diff の同じインターフェースを持ち、下流の制御・評価
コードからは差し替えて使える。
"""
import numpy as np
import torch
import torch.nn as nn

from src.physics import G, L, M

STATE_ENC_DIM = 3  # sin(theta), cos(theta), theta_dot
CONTROL_DIM = 1  # u (M1では常に0)
DT = 0.02  # 学習データの離散化幅と一致させること


def encode_state(state: torch.Tensor) -> torch.Tensor:
    """(..., 2) [theta, theta_dot] -> (..., 3) [sin, cos, theta_dot]"""
    theta, theta_dot = state[..., 0], state[..., 1]
    return torch.stack([torch.sin(theta), torch.cos(theta), theta_dot], dim=-1)


def decode_state(enc: torch.Tensor) -> torch.Tensor:
    """(..., 3) [sin, cos, theta_dot] -> (..., 2) [theta, theta_dot]"""
    sin_t, cos_t, theta_dot = enc[..., 0], enc[..., 1], enc[..., 2]
    theta = torch.atan2(sin_t, cos_t)
    return torch.stack([theta, theta_dot], dim=-1)


def _make_mlp(out_dim: int, hidden_dim: int, n_hidden_layers: int) -> nn.Sequential:
    in_dim = STATE_ENC_DIM + CONTROL_DIM
    layers: list[nn.Module] = [nn.Linear(in_dim, hidden_dim), nn.Tanh()]
    for _ in range(n_hidden_layers - 1):
        layers += [nn.Linear(hidden_dim, hidden_dim), nn.Tanh()]
    layers.append(nn.Linear(hidden_dim, out_dim))
    return nn.Sequential(*layers)


class AutoregressiveModel(nn.Module):
    """1ステップ遷移 `step` を持つモデルに、自己回帰ロールアウトを与える基底クラス。"""

    def step(self, state: torch.Tensor, u: torch.Tensor | None = None) -> torch.Tensor:
        raise NotImplementedError

    @torch.no_grad()
    def rollout(
        self,
        initial_state: np.ndarray,
        n_steps: int,
        tau_seq: np.ndarray | None = None,
    ) -> np.ndarray:
        """numpy initial_state (..., 2) から自己回帰的にn_steps展開する(評価用、勾配なし)。

        tau_seq: shape (n_steps,) または (n_steps, ...)。Noneならu=0(無入力)。

        Returns: shape (n_steps + 1, ..., 2)
        """
        device = next(self.parameters()).device
        state = torch.as_tensor(initial_state, dtype=torch.float32, device=device)
        traj = [state]
        for t in range(n_steps):
            if tau_seq is None:
                u = None
            else:
                u_t = torch.as_tensor(tau_seq[t], dtype=torch.float32, device=device)
                target_shape = state.shape[:-1] + (CONTROL_DIM,)
                u = u_t.expand(target_shape) if u_t.numel() == 1 else u_t.reshape(target_shape)
            state = self.step(state, u)
            traj.append(state)
        return torch.stack(traj, dim=0).cpu().numpy()

    def rollout_diff(self, state: torch.Tensor, tau_seq: torch.Tensor, n_steps: int) -> torch.Tensor:
        """torch state (..., 2) から勾配を保持したまま自己回帰的にn_steps展開する(学習用)。

        M1-M6の1-step教師強制損失は、実際のロールアウト時に生じる誤差の蓄積を
        学習時に一切見ないというミスマッチがあった。この関数は評価用rolloutと
        同じ自己回帰計算を勾配ありで行い、マルチステップ・ロールアウト損失の
        逆伝播に使う。

        tau_seq: shape (n_steps, ..., 1) または (n_steps, ...)。

        Returns: shape (n_steps + 1, ..., 2) (初期状態を含む)
        """
        traj = [state]
        for t in range(n_steps):
            u_t = tau_seq[t]
            target_shape = state.shape[:-1] + (CONTROL_DIM,)
            u = u_t.reshape(target_shape)
            state = self.step(state, u)
            traj.append(state)
        return torch.stack(traj, dim=0)

    @staticmethod
    def _default_u(state: torch.Tensor, u: torch.Tensor | None) -> torch.Tensor:
        if u is not None:
            return u
        return torch.zeros(*state.shape[:-1], CONTROL_DIM, device=state.device)


class NSSModel(AutoregressiveModel):
    """ブラックボックス版(M1-M13): 状態遷移そのものをMLPで学習する。"""

    def __init__(self, hidden_dim: int = 64, n_hidden_layers: int = 2):
        super().__init__()
        self.net = _make_mlp(STATE_ENC_DIM, hidden_dim, n_hidden_layers)

    def forward(self, state: torch.Tensor, u: torch.Tensor | None = None) -> torch.Tensor:
        """state: (..., 2) -> next encoded state (..., 3)"""
        enc = encode_state(state)
        net_in = torch.cat([enc, self._default_u(state, u)], dim=-1)
        return self.net(net_in)

    def step(self, state: torch.Tensor, u: torch.Tensor | None = None) -> torch.Tensor:
        """state: (..., 2) -> next decoded state (..., 2)"""
        return decode_state(self.forward(state, u))


class GrayBoxNSSModel(AutoregressiveModel):
    """グレーボックス版(M14〜): 既知の物理構造 + 残差加速度のみをNNで学習する。

    加速度を
        theta_ddot = -(g/L) sin(theta) + tau/(m L^2) + NN_residual(sin, cos, theta_dot, tau)
    と分解し、既知項(重力・トルク入力)と積分構造(RK4)はハードコードする。
    NNが担うのは未知成分のみで、真の系ではこれは減衰項 -c*theta_dot に相当する。

    ブラックボックス版と違い theta を atan2 で毎ステップ畳み込まないため、出力の
    theta は連続(unwrap済み相当)になる。角度差を扱う下流コードは全てwrapに対して
    不変な計算(atan2ベース)をしているため、そのまま差し替えて使える。
    """

    def __init__(self, hidden_dim: int = 64, n_hidden_layers: int = 2, dt: float = DT):
        super().__init__()
        self.net = _make_mlp(1, hidden_dim, n_hidden_layers)
        self.dt = dt

    def residual_acceleration(self, state: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        """未知成分(摩擦など)による残差加速度。 (..., 2), (..., 1) -> (...,)"""
        net_in = torch.cat([encode_state(state), u], dim=-1)
        return self.net(net_in).squeeze(-1)

    def dynamics(self, state: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        """dx/dt = [theta_dot, theta_ddot]。 (..., 2) -> (..., 2)"""
        theta, theta_dot = state[..., 0], state[..., 1]
        known_acc = -(G / L) * torch.sin(theta) + u[..., 0] / (M * L**2)
        theta_ddot = known_acc + self.residual_acceleration(state, u)
        return torch.stack([theta_dot, theta_ddot], dim=-1)

    def step(self, state: torch.Tensor, u: torch.Tensor | None = None) -> torch.Tensor:
        """既知構造のRK4で1ステップ積分する。 (..., 2) -> (..., 2)"""
        u = self._default_u(state, u)
        dt = self.dt
        k1 = self.dynamics(state, u)
        k2 = self.dynamics(state + 0.5 * dt * k1, u)
        k3 = self.dynamics(state + 0.5 * dt * k2, u)
        k4 = self.dynamics(state + dt * k3, u)
        return state + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
