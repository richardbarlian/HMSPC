import copy
import json
import os
import random
import torch
from models import VanillaNeuralODE
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

# set deterministic environment
parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()
set_seed(args.seed)

# configuration
num_epochs = 50
patience = 10  # early stopping patience

n_latent_dim = 32
n_hidden_dim = 64

device = torch.device("cpu")
run_name = f"vanilla_neural_ode_s{args.seed}"

os.makedirs("saves", exist_ok=True)

print(f"Device: {device}")
print(f"Run: {run_name}\n")

path = "battery_all_batches.pkl"
data_dict = load_dataset(path)

# stratified split by cycle life at the cell level
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

# create datasets
train_dataset = HMSBatteryDataset(train_dict, window_size=120, num_windows_per_cell=50)
val_dataset = HMSBatteryDataset(val_dict, window_size=120, num_windows_per_cell=50)
test_dataset = HMSBatteryDataset(test_dict, window_size=120, num_windows_per_cell=50)

# pass generator and worker_init_fn to secure the shuffling indices reproducibility
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

model = VanillaNeuralODE(latent_dim=n_latent_dim, hidden_dim=n_hidden_dim).to(device)
model.apply(init_weights)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# add LR Scheduler tracking average validation loss
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", factor=0.5, patience=3
)

# dict initialization to track metrics identically
history = {"train_loss": [], "val_loss": []}

# early stopping tracking variables
best_val_loss = float("inf")
best_model_state = None
patience_counter = 0

for epoch in range(num_epochs):
    # TRAINING
    model.train()
    epoch_loss = 0.0

    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1:02d} [Train]", leave=True)
    for batch in pbar:
        batch = {k: v.to(device) for k, v in batch.items()}
        optimizer.zero_grad()

        loss = model.loss(batch)

        loss.backward()
        # bound trajectory gradient steps identical to main models
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        epoch_loss += loss.item()

        pbar.set_postfix({"Loss": f"{loss.item():.4f}"})

    num_train_batches = len(train_loader)
    avg_train_loss = epoch_loss / num_train_batches
    history["train_loss"].append(avg_train_loss)
    print(f"Epoch {epoch+1:02d} | Train Loss: {avg_train_loss:.4f}")

    # VALIDATION
    model.eval()
    val_loss = 0.0

    with torch.no_grad():
        for batch in val_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            loss = model.loss(batch)

            val_loss += loss.item()

    num_val_batches = len(val_loader)
    avg_val_loss = val_loss / num_val_batches

    # update learning rate tracker using validation metrics
    scheduler.step(avg_val_loss)

    history["val_loss"].append(avg_val_loss)
    print(f"         | Val Loss:   {avg_val_loss:.4f}")

    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        best_model_state = copy.deepcopy(model.state_dict())
        patience_counter = 0

        # save structural metadata format matching HMS-PC checkpointing
        torch.save(
            {
                "epoch": epoch + 1,
                "model": best_model_state,
                "val_loss": best_val_loss,
                "config": {
                    "latent_dim": n_latent_dim,
                    "hidden_dim": n_hidden_dim,
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

if best_model_state is not None:
    print("Loading best validation model for testing...")
    model.load_state_dict(best_model_state)
else:
    print("Warning: no cached state, using current weights.")

# save training history dictionary
with open(f"saves/{run_name}_history.json", "w") as f:
    json.dump(history, f, indent=2)

# TEST EVALUATION
print("\nRunning final evaluation on test set...")
metrics = evaluate_model_performance(model, test_loader, device, deterministic=True)

config = {
    "seed": args.seed,
    "latent_dim": n_latent_dim,
    "hidden_dim": n_hidden_dim,
}

log_results(metrics, run_name, config, best_val_loss, deterministic=True)
