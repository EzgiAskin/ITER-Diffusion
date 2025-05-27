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
        self.replay_buffer = replay_buffer

        dims = replay_buffer.buffer_shapes
        self.feature_dim = dims['o'][1] + dims['u'][1] + dims['ag'][1] + dims['g'][1]
        self.cond_dim = dims['o'][1] + dims['g'][1]
        self.obs_dim = dims['o'][1]
        self.goal_dim = dims['g'][1]
        self.act_dim = dims['u'][1]
        self.ag_dim = dims['ag'][1]

        betas = torch.linspace(beta_start, beta_end, time_steps, device=device)
        alphas = 1 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)
        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alpha_bars', alpha_bars)

        self.unet = TemporalUNet(self.feature_dim, time_steps, self.cond_dim).to(device)
        self.stats = {}

    def compute_normalization_stats(self):
        buf = self.replay_buffer
        current = buf.current_size

        def get_stats(key):
            flat = buf.buffers[key][:current].reshape(-1, buf.buffer_shapes[key][1])
            return flat.mean(0), flat.std(0) + 1e-6

        self.stats['o_mean'], self.stats['o_std'] = get_stats('o')
        self.stats['u_mean'], self.stats['u_std'] = get_stats('u')
        self.stats['ag_mean'], self.stats['ag_std'] = get_stats('ag')
        self.stats['g_mean'], self.stats['g_std'] = get_stats('g')

    def normalize(self, key, x):
        mean = self.stats[f'{key}_mean']
        std = self.stats[f'{key}_std']
        return (x - mean) / std

    def denormalize(self, key, x):
        mean = self.stats[f'{key}_mean']
        std = self.stats[f'{key}_std']
        return x * std + mean

    def p_losses(self, x0, cond, t):
        noise = torch.randn_like(x0)
        ab = self.alpha_bars[t].view(-1, 1, 1)
        noisy_x = ab.sqrt() * x0 + (1 - ab).sqrt() * noise
        eps_pred = self.unet(noisy_x, t, cond)
        return ((noise - eps_pred) ** 2).mean()

    def train_ddpm(self, episodes, batch_size=64, epochs=10, lr=1e-4, return_losses=False):
        self.to(self.device)
        self.compute_normalization_stats()
        opt = optim.Adam(self.parameters(), lr=lr)
        feats, conds = [], []
        for ep in episodes:
            o = self.normalize('o', ep['o'][:-1])
            u = self.normalize('u', ep['u'])
            ag = self.normalize('ag', ep['ag'][:-1])
            g = self.normalize('g', ep['g'])
            f = np.concatenate([o, u, ag, g], axis=-1)
            feats.append(f.T)
            conds.append(np.concatenate([self.normalize('g', ep['g'][0]), self.normalize('o', ep['o'][0])]))
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
            avg_loss = total_loss / len(loader.dataset)
            if return_losses:
                losses_list.append(avg_loss)
        if return_losses:
            return losses_list

    @torch.no_grad()
    def sample_episode(self, s0_raw, g0_raw):
        dims = self.replay_buffer.buffer_shapes
        trans_T = dims['u'][0]
        obs_dim = dims['o'][1]
        act_dim = dims['u'][1]
        ag_dim = dims['ag'][1]
        goal_dim = dims['g'][1]

        # Normalize s0 and g0
        s0 = self.normalize('o', s0_raw)
        g0 = self.normalize('g', g0_raw)

        # Condition
        cond = torch.tensor(
            np.concatenate([g0, s0]),
            dtype=torch.float32, device=self.device
        ).unsqueeze(0)

        #Generate normalized trajectory [1, feature_dim, T]
        x = torch.randn(1, self.feature_dim, trans_T, device=self.device)
        for t in reversed(range(self.time_steps)):
            t_tensor = torch.full((1,), t, dtype=torch.long, device=self.device)
            eps = self.unet(x, t_tensor, cond)
            a_t, ab_t = self.alphas[t], self.alpha_bars[t]
            x = (1 / a_t.sqrt()) * (x - (1 - a_t) / (1 - ab_t).sqrt() * eps)
            if t > 0:
                noise = torch.randn_like(x)
                x = x + torch.sqrt((1 - a_t) / (1 - ab_t)) * noise

        # Denormalize before splitting
        traj = x.squeeze(0).cpu().numpy().T  # [T, feature_dim]
        idx = 0
        o_seq_norm = traj[:, idx:idx + obs_dim];
        idx += obs_dim
        u_seq_norm = traj[:, idx:idx + act_dim];
        idx += act_dim
        ag_seq_norm = traj[:, idx:idx + ag_dim];
        idx += ag_dim
        g_seq_norm = traj[:, idx:idx + goal_dim];
        idx += goal_dim

        o_seq = self.denormalize('o', o_seq_norm)
        u_seq = self.denormalize('u', u_seq_norm)
        ag_seq = self.denormalize('ag', ag_seq_norm)
        g_seq = self.denormalize('g', g_seq_norm)

        # Build s sequence starting from real s0
        s_seq = np.vstack([s0_raw, o_seq])  # (T+1, obs_dim)

        # Recompute achieved goals from denormalized o_seq
        # Assuming ag = s[8:11] (HER convention) wrong!!!! from prev!!!!
        ##############
        ag_seq_from_o = [s0_raw[3:6]]
        for s_flat in o_seq:
            ag_seq_from_o.append(s_flat[3:6])
        ag_seq_from_o = np.stack(ag_seq_from_o, axis=0)
        ##############

        # Add some noise to g_seq to avoid singular covariance for evaluation ?
        #g_seq = np.tile(g0_raw, (trans_T, 1)) + np.random.normal(0, 1e-5, size=(trans_T, len(g0_raw))) dont do it fixed this is not optimal
        real_g_std = np.std(self.replay_buffer.buffers['g'][:self.replay_buffer.current_size], axis=(0, 1))
        g_seq = np.tile(g0_raw, (trans_T, 1)) + np.random.normal(0, real_g_std, size=(trans_T, len(g0_raw)))

        # todo is this the best approach? Because we add some noise to g_seq
        # Making sure that the noise is half as small as the tol?
        info = compute_success(ag_seq_from_o, g0_raw, tol=self.tol)

        return {
            'o': s_seq,  # (T+1, obs_dim)
            'u': u_seq,  # (T, act_dim)
            'ag': ag_seq_from_o,  # (T+1, ag_dim)
            'g': g_seq,  # (T, goal_dim)
            'info_is_success': info  # (T, 1)
        }

    def generate_and_store_synthetic_data(self, num_synthetic_episodes):
        contexts = self.replay_buffer.sample_successful_contexts(num_synthetic_episodes)
        for s0, g0 in contexts:
            ep = self.sample_episode(s0, g0)
            batch = {k: v[None] for k, v in ep.items()}
            self.replay_buffer.store_episode(batch, synthetic=True)
        #print(f"Added {len(contexts)} synthetic eps from successful contexts.")


