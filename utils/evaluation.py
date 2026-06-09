import json
import os
import numpy as np
import scipy.stats as stats
import torch


def calculate_battery_metrics(preds, targets, variances=None):
    """
    Deterministic and probabilistic metrics for battery state estimation.
    Variances optional - if None, probabilistic metrics are omitted.
    """
    metrics = {}
    residuals = preds - targets

    # deterministic
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((targets - np.mean(targets)) ** 2)
    metrics["RMSE_mV"] = np.sqrt(np.mean(residuals**2)) * 1000.0
    metrics["R2_Score"] = 1.0 - (ss_res / (ss_tot + 1e-9))

    # probabilistic
    if variances is not None:
        stds = np.sqrt(np.clip(variances, 1e-9, None))

        nll = 0.5 * np.log(2 * np.pi * variances) + (residuals**2) / (2 * variances)
        metrics["Gaussian_NLL"] = np.mean(nll)

        quantiles = np.linspace(0.1, 0.9, 9)
        ece = 0.0
        for q in quantiles:
            z_score = stats.norm.ppf(1.0 - (1.0 - q) / 2.0)
            lb = preds - z_score * stds
            ub = preds + z_score * stds
            ece += np.abs(np.mean((targets >= lb) & (targets <= ub)) - q)
        metrics["Regression_ECE"] = ece / len(quantiles)

    return metrics


def evaluate_model_performance(model, data_loader, device, deterministic=False):
    """
    Loops through a DataLoader and computes metrics.
    Set deterministic=True for baselines without learned variance.
    """
    model.eval()
    all_preds = []
    all_targets = []
    all_vars = []

    with torch.no_grad():
        for batch in data_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            V_true = batch["V"]
            mask = batch["mask"].squeeze(-1)

            outputs = model(batch)
            V_pred = outputs[0] if isinstance(outputs, tuple) else outputs
            V_var = (
                outputs[1]
                if (
                    not deterministic
                    and isinstance(outputs, tuple)
                    and len(outputs) >= 2
                )
                else None
            )

            for i in range(V_true.size(0)):
                valid = mask[i].bool()
                all_preds.append(V_pred[i][valid].cpu())
                all_targets.append(V_true[i][valid].cpu())
                if V_var is not None:
                    all_vars.append(V_var[i][valid].cpu())

    flat_preds = torch.cat(all_preds).numpy()
    flat_targets = torch.cat(all_targets).numpy()
    flat_vars = torch.cat(all_vars).numpy() if all_vars else None

    return calculate_battery_metrics(flat_preds, flat_targets, flat_vars)


def log_results(metrics, run_name, config, val_loss, deterministic=False):
    """
    Print metrics and saves results json to saves/.
    Probabilistic metrics only printed and saved for non-deterministic models.
    """
    print(f"\nResults: {run_name}")

    print("\nDeterministic")
    print(f"  RMSE:     {metrics['RMSE_mV']:.4f} mV")
    print(f"  R2 Score: {metrics['R2_Score']:.4f}")

    results = {
        "run": run_name,
        "val_loss": val_loss,
        "metrics": {
            "rmse_mv": float(metrics["RMSE_mV"]),
            "r2_score": float(metrics["R2_Score"]),
        },
        "config": config,
    }

    if not deterministic:
        print("\nProbabilistic")
        print(f"  Gaussian NLL:   {metrics['Gaussian_NLL']:.4f}")
        print(f"  Regression ECE: {metrics['Regression_ECE']:.4f}")

        results["metrics"]["gaussian_nll"] = float(metrics["Gaussian_NLL"])
        results["metrics"]["regression_ece"] = float(metrics["Regression_ECE"])

    os.makedirs("saves", exist_ok=True)
    with open(f"saves/{run_name}_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved to saves/{run_name}_results.json")
