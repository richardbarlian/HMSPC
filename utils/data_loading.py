import pickle
from tqdm import tqdm
import numpy as np


# data loading
def load_dataset(filepath):
    print(f"Loading {filepath}...")

    with open(filepath, "rb") as fp:
        bat_dict = pickle.load(fp)

    # required_cell_keys = {"cycle_life", "charge_policy", "cycles", "batch"}
    # required_cycle_keys = {"I", "Qc", "Qd", "Qdlin", "T", "Tdlin", "V", "dQdV", "t"}

    # for cell_id, cell_info in tqdm(bat_dict.items(), desc="Verifying cells"):
    #     assert required_cell_keys.issubset(cell_info.keys())
    #     assert len(cell_info["cycles"]) > 0

    #     for cycle in cell_info["cycles"]:
    #         assert required_cycle_keys.issubset(cycle.keys())

    # print(f"Sanity check passed: {len(bat_dict)} cells verified")
    return bat_dict


# temperature cleaning
def clean_temperature(T):
    T = np.asarray(T, dtype=np.float32)
    T[T == -270] = np.nan
    T[T == 400] = np.nan
    T[(T < 15) | (T > 60)] = np.nan  # chamber at 30°C, fast-charge rise allowed
    return T
