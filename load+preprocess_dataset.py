import h5py
import numpy as np
from tqdm import tqdm
import pickle
import subprocess
import os
import shutil

# download and extract dataset
print("Downloading dataset...")
subprocess.run(
    [
        "kaggle",
        "datasets",
        "download",
        "itshpark/data-driven-prediction-of-battery-cycle",
    ],
    check=True,
)

print("Unzipping dataset...")
subprocess.run(
    ["unzip", "-o", "data-driven-prediction-of-battery-cycle.zip", "-d", "dataset"],
    check=True,
)

files = [
    "./dataset/2017-05-12_batchdata_updated_struct_errorcorrect.mat",
    "./dataset/2018-02-20_batchdata_updated_struct_errorcorrect.mat",
    "./dataset/2018-04-03_varcharge_batchdata_updated_struct_errorcorrect.mat",
    "./dataset/2018-04-12_batchdata_updated_struct_errorcorrect.mat",
]

bat_dict = {}

print("\nMerging all batches...")

for file_idx, matFilename in enumerate(files):
    if not os.path.exists(matFilename):
        print(f"Skipping missing file: {matFilename}")
        continue

    print(f"Processing: {matFilename}")

    f = h5py.File(matFilename, "r")
    batch = f["batch"]
    num_cells = batch["summary"].shape[0]
    batch_prefix = f"b{file_idx + 1}"

    for i in tqdm(range(num_cells), desc=f"Cells {batch_prefix}"):
        cycles_ref = batch["cycles"][i, 0]
        cycles = f[cycles_ref]

        cl = np.array(f[batch["cycle_life"][i, 0]][()]).squeeze().item()
        policy = f[batch["policy_readable"][i, 0]][()].tobytes()[::2].decode()

        num_cycles = cycles["I"].shape[0]
        cell_cycles = []

        for j in range(num_cycles):

            def G(key):
                return f[cycles[key][j, 0]][()]

            cycle_data = {
                "I": np.hstack(G("I")).astype(np.float32),
                "Qc": np.hstack(G("Qc")).astype(np.float32),
                "Qd": np.hstack(G("Qd")).astype(np.float32),
                "Qdlin": np.hstack(G("Qdlin")).astype(np.float32),
                "T": np.hstack(G("T")).astype(np.float32),
                "Tdlin": np.hstack(G("Tdlin")).astype(np.float32),
                "V": np.hstack(G("V")).astype(np.float32),
                "dQdV": np.hstack(G("discharge_dQdV")).astype(np.float32),
                "t": np.hstack(G("t")).astype(np.float32),
            }
            cell_cycles.append(cycle_data)

        bat_dict[f"{batch_prefix}c{i}"] = {
            "cycle_life": cl,
            "charge_policy": policy,
            "cycles": cell_cycles,
            "batch": batch_prefix,
        }

    f.close()

# cleanup
shutil.rmtree("dataset")
os.remove("data-driven-prediction-of-battery-cycle.zip")
print("\nCleaned up dataset files")

with open("battery_all_batches.pkl", "wb") as fp:
    pickle.dump(bat_dict, fp)

print(f"\nSaved -> battery_all_batches.pkl ({len(bat_dict)} cells)")
