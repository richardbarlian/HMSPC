from torch.utils.data import Dataset
import torch
import numpy as np
from .data_loading import clean_temperature
from collections import defaultdict


# dataset
class HMSBatteryDataset(Dataset):
    def __init__(self, bat_dict, window_size=120, num_windows_per_cell=50):
        self.samples = []
        self.window_size = window_size

        for cell_id, cell_info in bat_dict.items():

            cycles = cell_info["cycles"]

            # weight cycle selection by length for uniform temporal coverage
            cycle_lengths = np.array([len(c["t"]) for c in cycles], dtype=np.float64)
            cycle_weights = cycle_lengths / cycle_lengths.sum()

            valid_windows = 0
            attempts = 0
            max_attempts = num_windows_per_cell * 30

            while valid_windows < num_windows_per_cell and attempts < max_attempts:
                attempts += 1

                cycle_idx = np.random.choice(len(cycles), p=cycle_weights)
                cycle = cycles[cycle_idx]

                t_full = np.asarray(cycle["t"], dtype=np.float32)
                V_full = np.asarray(cycle["V"], dtype=np.float32)
                I_full = np.asarray(cycle["I"], dtype=np.float32)
                T_full = clean_temperature(cycle["T"])

                if len(t_full) <= self.window_size:
                    continue

                start = np.random.randint(0, len(t_full) - self.window_size)

                t = t_full[start : start + self.window_size]
                V = V_full[start : start + self.window_size]
                I = I_full[start : start + self.window_size]
                T = T_full[start : start + self.window_size]

                if np.isnan(T).mean() > 0.1:
                    continue

                if np.isnan(T).any():
                    idx = np.arange(len(T))
                    mask = ~np.isnan(T)

                    if mask.sum() < 2:
                        continue

                    T = np.interp(idx, idx[mask], T[mask]).astype(np.float32)

                # reject cv taper windows - voltage pinned at upper cutoff, no dynamics for drift
                if np.std(V) < 1e-4:
                    continue

                dt = np.diff(t)

                if np.any(dt <= 0):
                    continue

                if np.max(dt) > 1e3:
                    continue

                # prepend 0.0 as a neutral dt placeholder for t[0] (no prior step)
                dt = np.concatenate([[0.0], dt])

                if not (
                    np.isfinite(t).all()
                    and np.isfinite(V).all()
                    and np.isfinite(I).all()
                    and np.isfinite(T).all()
                    and np.isfinite(dt).all()
                ):
                    continue

                self.samples.append(
                    {
                        "cell_id": cell_id,
                        "cycle_idx": cycle_idx,
                        "t": torch.tensor(t, dtype=torch.float32),
                        "dt": torch.tensor(dt, dtype=torch.float32),
                        "V": torch.tensor(V, dtype=torch.float32),
                        "I": torch.tensor(I, dtype=torch.float32),
                        "T": torch.tensor(T, dtype=torch.float32),
                    }
                )

                valid_windows += 1

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

    def get_monotonicity_pairs(self, n_pairs=None):
        """
        returns list of (sample_a, sample_b) where sample_b is from a later
        cycle than sample_a, both from the same cell.
        """

        cell_map = defaultdict(list)
        for s in self.samples:
            cell_map[s["cell_id"]].append(s)

        # sort each cell's samples by cycle index
        for cid in cell_map:
            cell_map[cid].sort(key=lambda x: x["cycle_idx"])

        pairs = []
        for cid, samples in cell_map.items():
            if len(samples) < 2:
                continue
            for i in range(len(samples) - 1):
                pairs.append((samples[i], samples[i + 1]))

        if n_pairs is not None:
            idx = np.random.choice(len(pairs), min(n_pairs, len(pairs)), replace=False)
            pairs = [pairs[i] for i in idx]

        return pairs


# collate
def hms_collate_fn(batch):
    def pad(key):
        return torch.nn.utils.rnn.pad_sequence(
            [x[key] for x in batch], batch_first=True, padding_value=0.0
        )

    lengths = torch.tensor([len(x["t"]) for x in batch])

    return {
        "t": pad("t"),
        "dt": pad("dt"),
        "V": pad("V"),
        "I": pad("I"),
        "T": pad("T"),
        "cycle_idx": torch.stack([torch.tensor(x["cycle_idx"]) for x in batch]),
        "mask": torch.stack(
            [torch.cat([torch.ones(l), torch.zeros(max(lengths) - l)]) for l in lengths]
        ).unsqueeze(-1),
        "lengths": lengths,
    }
