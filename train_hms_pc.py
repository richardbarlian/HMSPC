import copy
import json
import os
import random
import torch
from models import HMSPC
from tqdm import tqdm
import numpy as np
from utils import (
    HMSBatteryDataset,
    evaluate_model_performance,
    hms_collate_fn,
    load_dataset,
    seed_worker,
    set_seed,
    log_results,
    init_weights,
)
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--use_state_dep_noise", type=int, default=1)
parser.add_argument("--condition_drift_on_inputs", type=int, default=1)
# parser.add_argument("--use_mse", type=int, default=0)
parser.add_argument("--use_gate", type=int, default=1)
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()

use_state_dep_noise = bool(args.use_state_dep_noise)
condition_drift_on_inputs = bool(args.condition_drift_on_inputs)
use_gate = bool(args.use_gate)

set_seed(args.seed)  # set deterministic env

# configuration
num_epochs = 50
patience = 10
target_kl_weight = 0.001
warmup_epochs = 15  # KL weight scales linearly from 0 to target over 15 epochs
alpha = 0.05
target_beta = 0.3
beta_warmup_epochs = 15  # beta scales linearly from 0 to target_beta over 10 epochs
device = torch.device("cpu")

n_latent_dim = 32
n_hidden_dim = 64

# run name - used for save paths
run_name = (
    f"hmspc"
    f"_sdn{int(use_state_dep_noise)}"
    f"_cdi{int(condition_drift_on_inputs)}"
    f"_ug{int(use_gate)}"
    f"_s{args.seed}"
)

os.makedirs("saves", exist_ok=True)

print(f"Device: {device}")
print(f"Run: {run_name}\n")
print(f"Ablation config:")
# print(f"  use MSE criterion:      {use_mse}")
print(f"  state-dep noise:        {use_state_dep_noise}")
print(f"  condition drift inputs: {condition_drift_on_inputs}")
print(f"  use gate:               {use_gate}\n")

path = "battery_all_batches.pkl"
data_dict = load_dataset(path)

# stratified split by cycle life at the CELL level
# ensures train/val/test all contain short-, medium-, and long-life batteries
cell_ids = list(data_dict.keys())

valid_cells = []
nan_cells = []

for cid in cell_ids:
    cycle_life = float(data_dict[cid]["cycle_life"])
    if np.isnan(cycle_life):
        nan_cells.append(cid)
    else:
        valid_cells.append((cid, cycle_life))

# sort by cycle life
valid_cells.sort(key=lambda x: x[1])

lives = [life for _, life in valid_cells]
print(f"Valid cycle lives: {len(lives)}, range: {min(lives):.0f} - {max(lives):.0f}")
print(f"NaN cells (assigned to train): {len(nan_cells)}\n")

# reproducible RNG
rng = random.Random(args.seed)

train_cells, val_cells, test_cells = [], [], []

# split ranked cells into quantile bins
n_bins = 20
bins = np.array_split(valid_cells, n_bins)

for bin_cells in bins:
    bin_cells = list(bin_cells)
    rng.shuffle(bin_cells)

    n_bin = len(bin_cells)

    n_train = int(round(0.70 * n_bin))
    n_val = int(round(0.15 * n_bin))

    train_cells.extend(cid for cid, _ in bin_cells[:n_train])
    val_cells.extend(cid for cid, _ in bin_cells[n_train : n_train + n_val])
    test_cells.extend(cid for cid, _ in bin_cells[n_train + n_val :])

# put NaN-life cells into train only
train_cells.extend(nan_cells)

# final shuffle within splits
rng.shuffle(train_cells)
rng.shuffle(val_cells)
rng.shuffle(test_cells)

train_dict = {cid: data_dict[cid] for cid in train_cells}
val_dict = {cid: data_dict[cid] for cid in val_cells}
test_dict = {cid: data_dict[cid] for cid in test_cells}

train_dataset = HMSBatteryDataset(train_dict, window_size=120, num_windows_per_cell=50)
val_dataset = HMSBatteryDataset(val_dict, window_size=120, num_windows_per_cell=50)
test_dataset = HMSBatteryDataset(test_dict, window_size=120, num_windows_per_cell=50)

g = torch.Generator()
g.manual_seed(args.seed)

train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=16,
    shuffle=True,
    collate_fn=hms_collate_fn,
    worker_init_fn=seed_worker,
    generator=g,
)
val_loader = torch.utils.data.DataLoader(
    val_dataset, batch_size=16, shuffle=False, collate_fn=hms_collate_fn
)
test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=16, shuffle=False, collate_fn=hms_collate_fn
)

print(f"Train dataset ready: {len(train_dataset)} windows ({len(train_cells)} cells)")
print(f"Val dataset ready:   {len(val_dataset)} windows ({len(val_cells)} cells)")
print(f"Test dataset ready:  {len(test_dataset)} windows ({len(test_cells)} cells)\n")

model = HMSPC(
    latent_dim=n_latent_dim,
    hidden_dim=n_hidden_dim,
    use_state_dep_noise=use_state_dep_noise,
    condition_drift_on_inputs=condition_drift_on_inputs,
    use_gate=use_gate,
).to(device)
model.apply(init_weights)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", factor=0.5, patience=3
)

best_val_loss = float("inf")
best_model_state = None
patience_counter = 0

history = {
    "train_loss": [],
    "train_recon": [],
    "train_kl": [],
    "train_unc": [],
    "val_loss": [],
    "val_recon": [],
    "val_kl": [],
}

for epoch in range(num_epochs):
    model.train()
    current_kl_weight = target_kl_weight * min(1.0, epoch / warmup_epochs)
    current_beta = target_beta * min(1.0, epoch / beta_warmup_epochs)
    noise_warmup = epoch < beta_warmup_epochs

    epoch_loss = epoch_recon = epoch_kl = epoch_unc = 0.0

    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1:02d} [Train]", leave=True)
    for batch in pbar:
        batch = {k: v.to(device) for k, v in batch.items()}

        optimizer.zero_grad()
        loss, recon, kl, l_unc = model.loss(
            batch,
            beta=current_beta,
            kl_weight=current_kl_weight,
            # use_mse=use_mse,
            noise_warmup=noise_warmup,
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        epoch_loss += loss.item()
        epoch_recon += recon.item()
        epoch_kl += kl.item()
        epoch_unc += l_unc.item()

        pbar.set_postfix(
            {
                "Loss": f"{loss.item():.4f}",
                "Recon": f"{recon.item():.4f}",
                "KL": f"{kl.item():.4f}",
                "Unc": f"{l_unc.item():.4f}",
            }
        )

    n = len(train_loader)
    t_loss = epoch_loss / n
    t_recon = epoch_recon / n
    t_kl = epoch_kl / n
    t_unc = epoch_unc / n

    history["train_loss"].append(t_loss)
    history["train_recon"].append(t_recon)
    history["train_kl"].append(t_kl)
    history["train_unc"].append(t_unc)

    print(
        f"Epoch {epoch+1:02d} | Train Loss: {t_loss:.4f} | "
        f"Recon: {t_recon:.4f} | KL: {t_kl:.4f} | "
        f"Unc: {t_unc:.4f} | Beta: {current_beta:.3f}"
    )

    model.eval()
    val_loss = val_recon = val_kl = 0.0

    with torch.no_grad():
        for batch in val_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            loss, recon, kl, _ = model.loss(
                batch,
                beta=current_beta,
                kl_weight=current_kl_weight,
                # use_mse=use_mse,
            )
            val_loss += loss.item()
            val_recon += recon.item()
            val_kl += kl.item()

    n_val = len(val_loader)
    avg_val_loss = val_loss / n_val
    scheduler.step(avg_val_loss)

    history["val_loss"].append(avg_val_loss)
    history["val_recon"].append(val_recon / n_val)
    history["val_kl"].append(val_kl / n_val)

    print(
        f"         | Val Loss:   {avg_val_loss:.4f} | "
        f"Recon: {val_recon/n_val:.4f} | KL: {val_kl/n_val:.4f}"
    )

    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        best_model_state = copy.deepcopy(model.state_dict())
        patience_counter = 0
        torch.save(
            {
                "epoch": epoch + 1,
                "model": best_model_state,
                "val_loss": best_val_loss,
                "config": {
                    "latent_dim": n_latent_dim,
                    "hidden_dim": n_hidden_dim,
                    # "use_mse": use_mse,
                    "use_state_dep_noise": use_state_dep_noise,
                    "condition_drift_on_inputs": condition_drift_on_inputs,
                    "use_gate": use_gate,
                    "target_beta": target_beta,
                    "beta_warmup_epochs": beta_warmup_epochs,
                },
            },
            f"saves/{run_name}_best.pt",
        )
        print("         | --> New best model saved")
    else:
        patience_counter += 1
        print(
            f"         | --> No improvement. Early stopping: {patience_counter}/{patience}"
        )

    print()

    if patience_counter >= patience:
        print(f"Early stopping triggered at epoch {epoch+1}")
        break

# save training history
with open(f"saves/{run_name}_history.json", "w") as f:
    json.dump(history, f, indent=2)

# test evaluation
if best_model_state is not None:
    print("Loading best validation model for testing...")
    model.load_state_dict(best_model_state)
else:
    print("Warning: no cached state, using current weights.")

print("\nRunning final evaluation on test set...")
metrics = evaluate_model_performance(model, test_loader, device, deterministic=False)

config = {
    "seed": args.seed,
    # "use_mse": use_mse,
    "kl_weight": target_kl_weight,
    "warmup_epochs": warmup_epochs,
    "target_beta": target_beta,
    "beta_warmup_epochs": beta_warmup_epochs,
    "latent_dim": n_latent_dim,
    "hidden_dim": n_hidden_dim,
    "use_state_dep_noise": use_state_dep_noise,
    "condition_drift_on_inputs": condition_drift_on_inputs,
    "use_gate": use_gate,
}

log_results(metrics, run_name, config, best_val_loss, deterministic=False)
