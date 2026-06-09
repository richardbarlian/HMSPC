import json
from pathlib import Path

saves = Path("saves")

configs = {
    "hmspc_sdn1_cdi1_ug1": "Full",
    "hmspc_sdn0_cdi1_ug1": "w/o Σ_θ",
    "hmspc_sdn1_cdi0_ug1": "w/o drift cond.",
    "hmspc_sdn1_cdi1_ug0": "Hard cond.",
    "hmspc_sdn0_cdi0_ug1": "w/o everything",
    "latent_ode": "Latent ODE",
    "vanilla_ode": "Vanilla Neural ODE",
}


def config_key(run_str):
    run_str = run_str.lower()

    if "latent_ode" in run_str:
        return "latent_ode"
    if "vanilla" in run_str:
        return "vanilla_ode"
    if "hmspc" in run_str:
        parts = run_str.split("_")
        kept = [
            p
            for p in parts
            if any(p.startswith(x) for x in ("hmspc", "sdn", "cdi", "ug"))
        ]
        key = "_".join(kept)
        if "ug" not in key:
            key += "_ug1"
        return key

    return run_str


results = []
for path in saves.glob("*_results.json"):
    with open(path) as f:
        results.append(json.load(f))

groups = {}
for r in results:
    k = config_key(r["run"])
    groups.setdefault(k, []).append(r)

# IMPORTANT: iterate over ALL observed configs, not just configs dict
all_keys = sorted(groups.keys())

for cfg_key in all_keys:
    runs = groups[cfg_key]
    if not runs:
        continue

    label = configs.get(cfg_key, cfg_key)

    print(f"\n{label}")
    print(f"  {'Seed':<8} {'RMSE':>8} {'NLL':>8} {'ECE':>8}")

    runs_sorted = sorted(runs, key=lambda r: r["run"])

    for r in runs_sorted:
        seed = r["config"].get("seed", "?")
        m = r["metrics"]

        rmse = f"{m['rmse_mv']:.1f}" if m.get("rmse_mv") is not None else "n/a"
        nll = f"{m['gaussian_nll']:.3f}" if m.get("gaussian_nll") is not None else "n/a"
        ece = (
            f"{m['regression_ece']:.4f}"
            if m.get("regression_ece") is not None
            else "n/a"
        )

        print(f"  {seed:<8} {rmse:>8} {nll:>8} {ece:>8}")
