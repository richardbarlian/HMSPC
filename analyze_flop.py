import torch
import numpy as np
from thop import profile, clever_format

from models import HMSPC, LatentODE, VanillaNeuralODE

LATENT_DIM = 32
HIDDEN_DIM = 64
WINDOW_SIZE = 120

MODELS = {
    "HMS-PC (full)": HMSPC(
        LATENT_DIM,
        HIDDEN_DIM,
        use_state_dep_noise=True,
        condition_drift_on_inputs=True,
        use_gate=True,
    ),
    "HMS-PC hard cond.": HMSPC(
        LATENT_DIM,
        HIDDEN_DIM,
        use_state_dep_noise=True,
        condition_drift_on_inputs=True,
        use_gate=False,
    ),
    "HMS-PC w/o Σ_θ": HMSPC(
        LATENT_DIM,
        HIDDEN_DIM,
        use_state_dep_noise=False,
        condition_drift_on_inputs=True,
        use_gate=True,
    ),
    "HMS-PC w/o cond.": HMSPC(
        LATENT_DIM,
        HIDDEN_DIM,
        use_state_dep_noise=True,
        condition_drift_on_inputs=False,
        use_gate=True,
    ),
    "Latent ODE": LatentODE(LATENT_DIM, HIDDEN_DIM),
    "Vanilla NODE": VanillaNeuralODE(LATENT_DIM, HIDDEN_DIM),
    # "GRU Baseline": GRU(LATENT_DIM, HIDDEN_DIM),
}

# synthetic batch (batch size 1, single window)
B, T = 1, WINDOW_SIZE
batch = {
    "V": torch.randn(B, T),
    "I": torch.randn(B, T),
    "T": torch.randn(B, T),
    "dt": torch.ones(B, T) * 0.1,
    "mask": torch.ones(B, T, 1),
}

print(f"\n{'Model':<22} {'Params (K)':>10}  {'FLOPs (M)':>10}")
print("-" * 46)

baseline_flops = None
rows = {}

for name, model in MODELS.items():
    model.eval()
    flops, params = profile(model, inputs=(batch,), verbose=False)
    rows[name] = (flops, params)
    if name == "Latent ODE":
        baseline_flops = flops
    f_str, p_str = clever_format([flops, params], "%.2f")
    print(f"{name:<22} {p_str:>10}  {f_str:>10}")

if baseline_flops:
    print(f"\nRelative FLOPs vs Latent ODE:")
    for name, (flops, _) in rows.items():
        print(f"  {name:<22} {flops / baseline_flops:.2f}x")
