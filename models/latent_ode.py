import torch
import torch.nn as nn


class ODEFunc(nn.Module):
    """Time-invariant drift: dz/dt = f(z). No input conditioning."""

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


class ODERNNEncoder(nn.Module):
    """ODE-RNN recognition network. Runs backwards through observations."""

    def __init__(self, input_dim, latent_dim, hidden_dim=64):
        super().__init__()
        self.hidden_dim = hidden_dim

        # ODE transition between observations (backward Euler)
        self.ode_func = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.gru_cell = nn.GRUCell(input_dim, hidden_dim)
        self.to_mean = nn.Linear(hidden_dim, latent_dim)
        self.to_logvar = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x, dt, mask):
        """
        x:    (B, T, input_dim)
        dt:   (B, T)
        mask: (B, T, 1)
        """
        B, T, _ = x.shape
        h = torch.zeros(B, self.hidden_dim, device=x.device)

        for i in range(T - 1, -1, -1):
            x_i = x[:, i, :]
            dt_i = dt[:, i].unsqueeze(-1)  # (B, 1)
            mask_i = mask[:, i, 0]

            if i < T - 1:
                h = h - dt_i * self.ode_func(h)  # backward Euler ODE transition

            h_new = self.gru_cell(x_i, h)
            h = torch.where(mask_i.unsqueeze(-1).bool(), h_new, h)

        return self.to_mean(h), self.to_logvar(h)


# decoder: maps latent state + inputs to observed voltage
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


# full latent ode model
class LatentODE(nn.Module):
    def __init__(self, latent_dim=16, hidden_dim=64):
        super().__init__()
        self.latent_dim = latent_dim

        self.encoder = ODERNNEncoder(
            input_dim=3, latent_dim=latent_dim, hidden_dim=hidden_dim
        )
        self.ode_func = ODEFunc(latent_dim=latent_dim, hidden_dim=hidden_dim)
        self.decoder = Decoder(
            latent_dim=latent_dim, input_dim=2, hidden_dim=hidden_dim
        )

    def reparameterise(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def integrate(self, z0, dt):
        """Forward Euler integration of the generative ODE."""
        B, T = dt.shape
        z = z0
        z_traj = []

        for i in range(T):
            dt_i = dt[:, i].unsqueeze(-1)
            z = z + self.ode_func(z) * dt_i
            z_traj.append(z.unsqueeze(1))

        return torch.cat(z_traj, dim=1)

    def forward(self, batch):
        V, I, T, dt, mask = (
            batch["V"],
            batch["I"],
            batch["T"],
            batch["dt"],
            batch["mask"],
        )

        x = torch.stack([V, I, T], dim=-1)
        mu, logvar = self.encoder(x, dt, mask)
        z0 = self.reparameterise(mu, logvar)

        inputs = torch.stack([I, T], dim=-1)  # (B, T, 2) - no normalisation

        z_traj = self.integrate(z0, dt)
        V_pred = self.decoder(z_traj, inputs)

        return V_pred, mu, logvar

    def loss(self, batch, kl_weight=1.0):
        V = batch["V"]
        mask = batch["mask"].squeeze(-1)

        V_pred, mu, logvar = self.forward(batch)

        # MSE reconstruction (fixed isotropic Gaussian likelihood)
        recon = ((V_pred - V) ** 2 * mask).sum() / mask.sum()

        # Closed-form KL to standard Gaussian prior
        kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(dim=-1).mean()

        return recon + kl_weight * kl, recon, kl
