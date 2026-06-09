import json
from pathlib import Path
import numpy as np

LABELS = {
    "hmspc_sdn1_cdi1_ug1": "HMS-PC (full)",
    "hmspc_sdn0_cdi1_ug1": "HMS-PC w/o Σ_θ",
    "hmspc_sdn1_cdi0_ug1": "HMS-PC w/o drift cond.",
    "hmspc_sdn1_cdi1_ug0": "HMS-PC hard cond. (no gate)",
    "hmspc_sdn0_cdi0_ug1": "HMS-PC w/o everything w/ NLL",
    "latent_ode": "Latent ODE baseline",
    "vanilla_ode": "Vanilla Neural ODE",
}

ORDER = [
    "hmspc_sdn1_cdi1_ug1",
    "hmspc_sdn0_cdi1_ug1",
    "hmspc_sdn1_cdi0_ug1",
    "hmspc_sdn1_cdi1_ug0",
    "hmspc_sdn0_cdi0_ug1",
    "latent_ode",
    "vanilla_ode",
]

TOP_DOWN = {
    "hmspc_sdn1_cdi1_ug1",
    "hmspc_sdn0_cdi1_ug1",
    "hmspc_sdn1_cdi0_ug1",
    "hmspc_sdn1_cdi1_ug0",
    "hmspc_sdn0_cdi0_ug1",
}
BASELINES = {"latent_ode", "vanilla_ode"}


def load_results(saves_dir="saves"):
    results = []
    for path in Path(saves_dir).glob("*_results.json"):
        with open(path) as f:
            results.append(json.load(f))
    return results


def config_key_from_run(run_str):
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
        # if ug token is missing (old runs without the flag), default to ug1
        key = "_".join(kept)
        if "ug" not in key:
            key += "_ug1"
        return key
    tokens = run_str.split("_")
    if tokens[-1].startswith("s") and tokens[-1][1:].isdigit():
        return "_".join(tokens[:-1])
    return run_str


def group_by_config(results):
    groups = {}
    for r in results:
        key = config_key_from_run(r["run"])
        groups.setdefault(key, []).append(r)
    return groups


def summarise(groups):
    rows = []
    for config_key, runs in groups.items():
        metrics_list = [r["metrics"] for r in runs]
        all_keys = metrics_list[0].keys()
        row = {"config": config_key, "n_seeds": len(runs)}
        for k in all_keys:
            vals = [m[k] for m in metrics_list if m.get(k) is not None]
            if vals:
                row[f"{k}_mean"] = np.mean(vals)
                row[f"{k}_std"] = np.std(vals)
        rows.append(row)
    return rows


def fmt(row, key_mean, key_std, decimals=2):
    mean = row.get(key_mean, float("nan"))
    std = row.get(key_std, float("nan"))
    f = f"{{:.{decimals}f}}"
    return f"{f.format(mean)} ± {f.format(std)}"


def print_table(rows):
    def sort_key(r):
        cfg = r["config"]
        return (
            ORDER.index(cfg) if cfg in ORDER else len(ORDER),
            r.get("rmse_mv_mean", float("inf")),
        )

    rows = sorted(rows, key=sort_key)
    label_w = 36

    print(
        f"\n{'Model':<{label_w}} {'N':>2}  {'RMSE (mV)':>15}  {'R2':>16}  {'NLL':>16}  {'ECE':>16}"
    )
    print("(± values are std across seeds)")
    print("-" * 110)

    prev_section = None
    for row in rows:
        cfg = row["config"]
        label = LABELS.get(cfg, cfg)
        n = row["n_seeds"]

        rmse = fmt(row, "rmse_mv_mean", "rmse_mv_std", 2)
        r2 = fmt(row, "r2_score_mean", "r2_score_std", 4)
        nll = (
            fmt(row, "gaussian_nll_mean", "gaussian_nll_std", 3)
            if "gaussian_nll_mean" in row
            else "n/a"
        )
        ece = (
            fmt(row, "regression_ece_mean", "regression_ece_std", 4)
            if "regression_ece_mean" in row
            else "n/a"
        )

        section = "top_down" if cfg in TOP_DOWN else "baseline"
        if section != prev_section and prev_section is not None:
            print()
        prev_section = section

        print(f"{label:<{label_w}} {n:>2}  {rmse:>15}  {r2:>16}  {nll:>16}  {ece:>16}")
    print()


def main():
    results = load_results("saves")
    if not results:
        print("No results files found in saves/")
        return

    groups = group_by_config(results)
    rows = summarise(groups)
    print_table(rows)

    with open("saves/ablation_summary.json", "w") as f:
        json.dump(rows, f, indent=2, default=str)
    print("Summary saved to saves/ablation_summary.json")


if __name__ == "__main__":
    main()
