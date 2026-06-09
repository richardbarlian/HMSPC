import argparse
import os
import matplotlib.pyplot as plt
import numpy as np
import torch
from models import HMSPC, LatentODE, VanillaNeuralODE
from mpl_toolkits.axes_grid1 import make_axes_locatable
from utils import init_style, infer_model_type, extract_2d_dynamics

parser = argparse.ArgumentParser(
    description="Generate a 2D latent vector field from a given checkpoint."
)
parser.add_argument("checkpoint_path", type=str, help="Path to the target *.pt file.")
args = parser.parse_args()

init_style()
device = torch.device("cpu")
graphs_dir = "graphs"
os.makedirs(graphs_dir, exist_ok=True)

if not os.path.exists(args.checkpoint_path):
    raise FileNotFoundError(f"Checkpoint not found at: {args.checkpoint_path}")

filename = os.path.basename(args.checkpoint_path)
print(f"Loading checkpoint: {args.checkpoint_path}")
checkpoint = torch.load(args.checkpoint_path, map_location=device)

config = checkpoint.get("config", {})
n_latent_dim = config.get("latent_dim", 32)
n_hidden_dim = config.get("hidden_dim", 64)
print(f"Loaded config -> latent_dim: {n_latent_dim}, hidden_dim: {n_hidden_dim}")

model_type = infer_model_type(filename)

if model_type == "hmspc":
    model = HMSPC(
        latent_dim=n_latent_dim,
        hidden_dim=n_hidden_dim,
        use_state_dep_noise=True,
        # use_stress_clock=True,
        condition_drift_on_inputs=True,
    )
    title_name = "Proposed HMSPC"
elif model_type == "latent_ode":
    model = LatentODE(latent_dim=n_latent_dim, hidden_dim=n_hidden_dim)
    title_name = "Latent ODE"
else:
    model = VanillaNeuralODE(latent_dim=n_latent_dim, hidden_dim=n_hidden_dim)
    title_name = "Vanilla Neural ODE"

model.load_state_dict(checkpoint["model"])

print(f"Extracting 2D dynamics for inferred architecture: {title_name}...")
X, Y, U, V = extract_2d_dynamics(
    model, model_type, latent_dim=n_latent_dim, grid_size=25, device=device
)
magnitude = np.sqrt(U**2 + V**2)

# base figure layout configuration
fig, ax = plt.subplots(figsize=(5.5, 5.2), dpi=300)
strm = ax.streamplot(
    X, Y, U, V, color=magnitude, cmap="plasma", linewidth=0.8, density=1.2
)

# standard ax title centers exactly to the plot box frame boundaries
ax.set_title(f"Vector Field: {title_name}", fontsize=12, pad=12)

ax.set_xlabel(r"Latent Dimension $h_1$", fontsize=10)
ax.set_ylabel(r"Latent Dimension $h_2$", fontsize=10)
ax.set_xlim(-3.0, 3.0)
ax.set_ylim(-3.0, 3.0)
ax.grid(True, linestyle=":", linewidth=0.5, color="#d0d0d0")
ax.tick_params(axis="both", direction="in", labelsize=9)

# split colorbar axis directly from the plot box layout geometry to prevent distortion
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.15)

cbar = fig.colorbar(strm.lines, cax=cax)
cbar.ax.tick_params(labelsize=8)
cbar.set_label(r"Velocity Magnitude $\||dh/dt|\|_2$", fontsize=9)

plt.tight_layout()

run_id = os.path.splitext(filename)[0]
save_output_path = os.path.join(graphs_dir, f"{run_id}_vector_field.png")
plt.savefig(save_output_path, bbox_inches="tight")
plt.close()

print(f"Vector field visualization successfully saved to: {save_output_path}")
