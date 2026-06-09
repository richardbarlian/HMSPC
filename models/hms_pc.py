import torch
import torch.nn as nn
import torch.nn.functional as F


class ODEFunc(nn.Module):
    def __init__(
        self, latent_dim, hidden_dim=64, condition_on_inputs=True, use_gate=True
    ):
        super().__init__()
        self.condition_on_inputs = condition_on_inputs
        self.use_gate = use_gate

        # base drift — always active
        self.base = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, latent_dim),
        )

        if condition_on_inputs:
            # input-conditioned correction
            self.delta = nn.Sequential(
                nn.Linear(latent_dim + 2, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, latent_dim),
            )
            # gate: scalar per sample, learns when inputs are informative
            self.gate = nn.Sequential(
                nn.Linear(latent_dim + 2, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, 1),
                nn.Sigmoid(),
            )

    def f(self, z, inputs):
        dz = self.base(z)
        if self.condition_on_inputs:
            zi = torch.cat([z, inputs], dim=-1)
            delta = self.delta(zi)
            if self.use_gate:
                gate = self.gate(zi)
                dz = dz + gate * delta
            else:
                dz = dz + delta  # hard conditioning, no gate
        return dz


class ODERNNEncoder(nn.Module):
    def __init__(self, input_dim, latent_dim, hidden_dim=64):
        super().__init__()
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim

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
        B, T, _ = x.shape  # (B, T, 3)
        h = torch.zeros(B, self.hidden_dim, device=x.device)

        for i in range(T - 1, -1, -1):
            x_i = x[:, i, :]
            dt_i = dt[:, i].unsqueeze(-1)
            mask_i = mask[:, i, 0]

            if i < T - 1:
                h = h - dt_i * self.ode_func(h)  # backward euler

            h_new = self.gru_cell(x_i, h)
            h = torch.where(mask_i.unsqueeze(-1).bool(), h_new, h)

        return self.to_mean(h), self.to_logvar(h)


class ObservationModel(nn.Module):
    def __init__(self, latent_dim, hidden_dim=64):
        super().__init__()
        self.mean_net = nn.Sequential(
            nn.Linear(latent_dim + 2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )
        # noise net now also sees inputs — noise varies with charging regime
        self.noise_net = nn.Sequential(
            nn.Linear(latent_dim + 2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
            nn.Softplus(),
        )

    def forward(self, z, inputs):
        """
        z:      (B, T, latent_dim)
        inputs: (B, T, 2)
        """
        zi = torch.cat([z, inputs], dim=-1)
        V_mean = self.mean_net(zi).squeeze(-1)
        V_var = self.noise_net(zi).squeeze(-1) + 1e-4
        return V_mean, V_var


class HMSPC(nn.Module):
    def __init__(
        self,
        latent_dim=16,
        hidden_dim=64,
        use_state_dep_noise=True,
        condition_drift_on_inputs=True,
        use_gate=True,
    ):
        super().__init__()

        self.latent_dim = latent_dim
        self.use_state_dep_noise = use_state_dep_noise

        self.encoder = ODERNNEncoder(
            input_dim=3, latent_dim=latent_dim, hidden_dim=hidden_dim
        )

        # separate norms for encoder inputs and decoder inputs
        # so LayerNorm doesn't destroy relative scale across timesteps
        self.input_norm = nn.LayerNorm(2)

        self.ode_func = ODEFunc(
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            condition_on_inputs=condition_drift_on_inputs,
            use_gate=use_gate,
        )
        self.obs_model = ObservationModel(latent_dim=latent_dim, hidden_dim=hidden_dim)

        if not use_state_dep_noise:
            # init to log(0.01) — sensible starting variance for voltage residuals
            self.register_parameter("log_obs_var", nn.Parameter(torch.tensor([-4.6])))

    def reparameterise(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def integrate(self, z0, inputs, dt):
        """Forward Euler integration."""
        B, T, _ = inputs.shape
        z = z0
        z_traj = []

        for i in range(T):
            inp_i = inputs[:, i, :]
            dt_i = dt[:, i]
            dz = self.ode_func.f(z, inp_i) * dt_i.unsqueeze(-1)
            z = z + dz
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

        inputs = torch.stack([I, T], dim=-1)  # (B, T, 2)
        inputs_normed = self.input_norm(inputs)  # (B, T, 2)

        z_traj = self.integrate(z0, inputs_normed, dt)
        V_mean, V_var = self.obs_model(z_traj, inputs_normed)

        if not self.use_state_dep_noise:
            V_var = torch.exp(self.log_obs_var).expand_as(V_mean)

        # soh = self.soh_head(z_traj[:, -1, :])

        return V_mean, V_var, mu, logvar, z_traj

    def loss(self, batch, beta=0.1, kl_weight=0.01, noise_warmup=False):
        V = batch["V"]
        mask = batch["mask"].squeeze(-1)
        V_mean, V_var, mu, logvar, z_traj = self.forward(batch)

        # noise_warmup: detach Sigma_theta only for state-dep path
        # scalar log_obs_var needs NLL signal from epoch 1
        if noise_warmup and self.use_state_dep_noise:
            V_var_for_recon = V_var.detach()
        else:
            V_var_for_recon = V_var

        recon = (
            0.5 * torch.log(V_var_for_recon)
            + 0.5 * ((V - V_mean) ** 2) / V_var_for_recon
        )
        recon = (recon * mask).sum() / mask.sum()

        kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(dim=-1).mean()

        l_uncertainty = torch.tensor(0.0, device=V.device)
        if self.use_state_dep_noise:
            residuals = (V - V_mean.detach()) ** 2
            l_uncertainty = ((V_var - residuals).pow(2) * mask).sum() / mask.sum()

        total = recon + kl_weight * kl + beta * l_uncertainty
        return total, recon, kl, l_uncertainty
