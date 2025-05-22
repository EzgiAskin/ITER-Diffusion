import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


def compute_success(ag_seq, goal, tol=0.05):
    dists = np.linalg.norm(ag_seq[1:] - goal.reshape(1, -1), axis=-1)
    return (dists < tol).astype(np.float32).reshape(-1, 1)

class TemporalUNet(nn.Module):
    def __init__(self, feature_dim, time_steps, cond_dim, hidden_dim=256):
        super().__init__()
        self.time_embed = nn.Embedding(time_steps, hidden_dim)
        self.enc1 = nn.Conv1d(feature_dim + cond_dim + hidden_dim, hidden_dim, 3, padding=1)
        self.enc2 = nn.Conv1d(hidden_dim, hidden_dim * 2, 3, padding=1)
        self.enc3 = nn.Conv1d(hidden_dim * 2, hidden_dim * 4, 3, padding=1)
        self.mid = nn.Conv1d(hidden_dim * 4, hidden_dim * 4, 3, padding=1)
        self.dec3 = nn.ConvTranspose1d(hidden_dim * 4, hidden_dim * 2, 3, padding=1)
        self.dec2 = nn.ConvTranspose1d(hidden_dim * 2, hidden_dim, 3, padding=1)
        self.dec1 = nn.ConvTranspose1d(hidden_dim, feature_dim, 3, padding=1)
        self.act = nn.ReLU()

    def forward(self, x, t, cond):
        # x: [B, feature_dim, T]
        # t: [B]
        # cond: [B, cond_dim]
        B, C, L = x.shape
        t_emb = self.time_embed(t).unsqueeze(-1).expand(-1, -1, L)
        cond_rep = cond.unsqueeze(-1).expand(-1, -1, L)
        h = torch.cat([x, t_emb, cond_rep], dim=1)
        e1 = self.act(self.enc1(h))
        e2 = self.act(self.enc2(e1))
        e3 = self.act(self.enc3(e2))
        m = self.act(self.mid(e3))
        d3 = self.act(self.dec3(m) + e2)
        d2 = self.act(self.dec2(d3) + e1)
        return self.dec1(d2)


class DDPM_Temporal(nn.Module):
    def __init__(self, replay_buffer, time_steps=50,
                 beta_start=1e-4, beta_end=2e-2, device='cpu', tol=0.05):
        super().__init__()
        self.device = device
        self.time_steps = time_steps
        self.tol = tol

        dims = replay_buffer.buffer_shapes
        # Feature per step s_t, a_t, ag_t, g_t
        self.feature_dim = dims['o'][1] + dims['u'][1] + dims['ag'][1] + dims['g'][1]
        #conditioning on initial state + goal
        self.cond_dim = dims['o'][1] + dims['g'][1]

        betas = torch.linspace(beta_start, beta_end, time_steps, device=device)
        alphas = 1 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)
        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alpha_bars', alpha_bars)

        self.unet = TemporalUNet(self.feature_dim, time_steps, self.cond_dim).to(device)
        self.replay_buffer = replay_buffer

    def q_sample(self, x0, t, noise=None):
        noise = noise if noise is not None else torch.randn_like(x0)
        sa = torch.sqrt(self.alpha_bars[t]).view(-1, 1, 1)
        sb = torch.sqrt(1 - self.alpha_bars[t]).view(-1, 1, 1)
        return sa * x0 + sb * noise, noise

    def p_losses(self, x0, cond, t):
        noise = torch.randn_like(x0)
        xt, noise = self.q_sample(x0, t, noise)
        pred = self.unet(xt, t, cond)
        return nn.functional.mse_loss(pred, noise)

    def train_ddpm(self, episodes, batch_size=64, epochs=10, lr=1e-4, return_losses=False):
        """
        Train the DDPM on real episodes from the replay buffer. todo Possibly only successful real episodes.
        Each episode provides T steps and features per step are [s_t, a_t, ag_t, g_t].
        """
        self.to(self.device)
        opt = optim.Adam(self.parameters(), lr=lr)
        feats, conds = [], []
        for ep in episodes:
            # s: length T+1, u: length T, ag: T+1, g: T
            # we want per-step features for t=0..T-1
            o = ep['o'][:-1]  # shape [T, state_dim]
            u = ep['u']  # shape [T, action_dim]
            ag = ep['ag'][:-1]  # shape [T, goal_dim]
            g = ep['g']  # shape [T, goal_dim]

            f = np.concatenate([o, u, ag, g], axis=-1)  # [T, feature_dim]
            feats.append(f.T)

            # conditioning
            conds.append(np.concatenate([ep['g'][0], ep['o'][0]]))
        X = torch.tensor(np.stack(feats), dtype=torch.float32, device=self.device)
        C = torch.tensor(np.stack(conds), dtype=torch.float32, device=self.device)
        loader = DataLoader(TensorDataset(X, C), batch_size=batch_size, shuffle=True)

        losses_list = [] if return_losses else None
        for e in range(epochs):
            total_loss = 0.0
            for x0, cond in loader:
                B = x0.size(0)
                t = torch.randint(0, self.time_steps, (B,), device=self.device)
                loss = self.p_losses(x0, cond, t)
                opt.zero_grad()
                loss.backward()
                opt.step()
                total_loss += loss.item() * B

            #print(f"Epoch {e + 1}/{epochs}, loss={total / len(loader.dataset):.6f}")
            avg_loss = total_loss / len(loader.dataset)

            if return_losses:
                losses_list.append(avg_loss)

        if return_losses:
            return losses_list