import torch
import torch.nn as nn


# the time-invariant vector-field
class ODEFunc(nn.Module):
    """Time-invariant drift: dz/dt = f(z)."""

    def __init__(self, latent_dim, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, latent_dim),
        )

    def forward(self, z):
        return self.net(z)


class Decoder(nn.Module):
    """MLP decoder: (z, I, T) -> V_pred."""

    def __init__(self, latent_dim, input_dim, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim + input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, z, inputs):
        """z: (B, T, latent_dim), inputs: (B, T, input_dim) -> (B, T)"""
        return self.net(torch.cat([z, inputs], dim=-1)).squeeze(-1)


class VanillaNeuralODE(nn.Module):
    def __init__(self, latent_dim=16, hidden_dim=64):
        super().__init__()
        # Single linear projection of first observation -> z0
        # Intentionally shallow: this is the "no encoder" baseline
        self.initial_projector = nn.Linear(3, latent_dim)
        self.ode_func = ODEFunc(latent_dim=latent_dim, hidden_dim=hidden_dim)
        self.decoder = Decoder(
            latent_dim=latent_dim, input_dim=2, hidden_dim=hidden_dim
        )

    def integrate(self, z0, dt):
        """Forward Euler integration."""
        B, T = dt.shape
        z = z0
        z_traj = []

        for i in range(T):
            dt_i = dt[:, i].unsqueeze(-1)
            z = z + self.ode_func(z) * dt_i
            z_traj.append(z.unsqueeze(1))

        return torch.cat(z_traj, dim=1)

    def forward(self, batch):
        V, I, T, dt = batch["V"], batch["I"], batch["T"], batch["dt"]

        x0 = torch.stack([V[:, 0], I[:, 0], T[:, 0]], dim=-1)
        z0 = self.initial_projector(x0)

        inputs = torch.stack([I, T], dim=-1)  # (B, T, 2) — no normalisation

        z_traj = self.integrate(z0, dt)
        V_pred = self.decoder(z_traj, inputs)
        return V_pred

    def loss(self, batch):
        V = batch["V"]
        mask = batch["mask"].squeeze(-1)
        V_pred = self.forward(batch)
        return ((V_pred - V) ** 2 * mask).sum() / mask.sum()
