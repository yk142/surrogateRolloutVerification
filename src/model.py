"""Neural State Space (NSS) サロゲートモデル。

離散時間の状態遷移 x_{t+1} = f_theta(x_t, u_t) をMLPで学習する。
theta の周期性を扱うため (sin theta, cos theta, theta_dot) にエンコードし、
出力も同じ表現で予測してから theta = atan2(sin, cos) にデコードする。

u_t (制御入力=トルク) はM1/M2では常に0だったが、M3からは実際のトルク値を渡して学習・
ロールアウトに使用する。
"""
import numpy as np
import torch
import torch.nn as nn

STATE_ENC_DIM = 3  # sin(theta), cos(theta), theta_dot
CONTROL_DIM = 1  # u (M1では常に0)


def encode_state(state: torch.Tensor) -> torch.Tensor:
    """(..., 2) [theta, theta_dot] -> (..., 3) [sin, cos, theta_dot]"""
    theta, theta_dot = state[..., 0], state[..., 1]
    return torch.stack([torch.sin(theta), torch.cos(theta), theta_dot], dim=-1)


def decode_state(enc: torch.Tensor) -> torch.Tensor:
    """(..., 3) [sin, cos, theta_dot] -> (..., 2) [theta, theta_dot]"""
    sin_t, cos_t, theta_dot = enc[..., 0], enc[..., 1], enc[..., 2]
    theta = torch.atan2(sin_t, cos_t)
    return torch.stack([theta, theta_dot], dim=-1)


class NSSModel(nn.Module):
    def __init__(self, hidden_dim: int = 64, n_hidden_layers: int = 2):
        super().__init__()
        in_dim = STATE_ENC_DIM + CONTROL_DIM
        layers: list[nn.Module] = [nn.Linear(in_dim, hidden_dim), nn.Tanh()]
        for _ in range(n_hidden_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.Tanh()]
        layers.append(nn.Linear(hidden_dim, STATE_ENC_DIM))
        self.net = nn.Sequential(*layers)

    def forward(self, state: torch.Tensor, u: torch.Tensor | None = None) -> torch.Tensor:
        """state: (..., 2) -> next encoded state (..., 3)"""
        enc = encode_state(state)
        if u is None:
            u = torch.zeros(*state.shape[:-1], CONTROL_DIM, device=state.device)
        net_in = torch.cat([enc, u], dim=-1)
        return self.net(net_in)

    def step(self, state: torch.Tensor, u: torch.Tensor | None = None) -> torch.Tensor:
        """state: (..., 2) -> next decoded state (..., 2)"""
        return decode_state(self.forward(state, u))

    @torch.no_grad()
    def rollout(
        self,
        initial_state: np.ndarray,
        n_steps: int,
        tau_seq: np.ndarray | None = None,
    ) -> np.ndarray:
        """numpy initial_state (..., 2) から自己回帰的にn_steps展開する。

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
