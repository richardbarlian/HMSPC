import json
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from utils import (
    init_style,
    styled_ax,
    smooth,
    save_fig,
    load_history,
    load_seed_histories,
    mean_across_seeds,
    load_results_metric,
    set_ylim_to_ignore_outliers,
)

# config
SAVES_DIR = "saves"
GRAPHS_DIR = "graphs"
SEEDS = [42, 100, 7, 6, 10]
os.makedirs(GRAPHS_DIR, exist_ok=True)

init_style()

COLORS = {
    "full": "#e05a47",
    "no_sdn": "#e67e22",
    "no_cdi": "#9b59b6",
    "no_nll": "#2ecc71",
    "hard_cond": "#3498db",
    "latent_ode": "#1abc9c",
    "vanilla": "#95a5a6",
}

ABLATION_RUNS = {
    "HMSPC (full)": ("hmspc_sdn1_cdi1_ug1_s{seed}", COLORS["full"]),
    "w/o Σ_θ": ("hmspc_sdn0_cdi1_ug1_s{seed}", COLORS["no_sdn"]),
    "w/o drift cond.": ("hmspc_sdn1_cdi0_ug1_s{seed}", COLORS["no_cdi"]),
    "w/o everything w/ NLL": ("hmspc_sdn0_cdi0_ug1_s{seed}", COLORS["no_nll"]),
}

GATE_RUNS = {
    "HMSPC (gated, full)": ("hmspc_sdn1_cdi1_ug1_s{seed}", COLORS["full"]),
    "Hard cond. (no gate)": ("hmspc_sdn1_cdi1_ug0_s{seed}", COLORS["hard_cond"]),
    "w/o drift cond.": ("hmspc_sdn1_cdi0_ug1_s{seed}", COLORS["no_cdi"]),
}

FULL_SINGLE = load_history(
    os.path.join(SAVES_DIR, "hmspc_sdn1_cdi1_ug1_s42_history.json")
)

# fig 1 - ablation val total loss
print("\n[Fig 1] Ablation val total loss...")
fig, ax = plt.subplots(figsize=(6.5, 4.5))
any_plotted = False
for label, (pattern, color) in ABLATION_RUNS.items():
    hists = load_seed_histories(pattern, SAVES_DIR, SEEDS)
    if not hists:
        continue
    epochs, mean = mean_across_seeds(hists, "val_loss")
    if epochs is None:
        continue
    ax.plot(epochs, mean, color=color, linewidth=1.8, label=label)
    any_plotted = True
if any_plotted:
    styled_ax(ax, "Validation Total Loss (Ablations)", "Epoch", "Loss")
    set_ylim_to_ignore_outliers(ax, percentile=99)
    ax.legend(
        frameon=True, edgecolor="black", facecolor="white", fontsize=8, framealpha=1
    )
    save_fig(fig, "fig1_ablations_val_loss.png", GRAPHS_DIR)
else:
    plt.close(fig)
    print("  Skipped - no data")

# fig 2 - full model train vs val diagnostics (seed 42)
print("[Fig 2] Full model diagnostics (seed 42)...")
if FULL_SINGLE is not None:
    metrics = [
        ("loss", "Total Loss", "Loss", (-4, 4)),
        ("recon", "Reconstruction Loss (NLL)", "NLL", (-4, 4)),
        ("kl", "KL Divergence", "KL", (0, 300)),
        ("unc", "Uncertainty Calibration Loss", "L_unc", None),
    ]
    available = [
        (k, t, y, lim)
        for k, t, y, lim in metrics
        if f"train_{k}" in FULL_SINGLE and f"val_{k}" in FULL_SINGLE
    ]
    if available:
        ncols = min(3, len(available))
        fig, axes = plt.subplots(1, ncols, figsize=(11, 4))
        axes = np.array(axes).flatten()
        epochs = range(1, len(FULL_SINGLE["train_loss"]) + 1)
        for i, (key, title, ylabel, ylim) in enumerate(available):
            ax = axes[i]
            ax.plot(
                epochs,
                FULL_SINGLE[f"train_{key}"],
                color=COLORS["full"],
                linestyle="--",
                linewidth=1.2,
                label="Train",
            )
            ax.plot(
                epochs,
                FULL_SINGLE[f"val_{key}"],
                color=COLORS["full"],
                linestyle="-",
                linewidth=1.8,
                label="Val",
            )
            styled_ax(ax, title, "Epoch", ylabel)
            ax.relim()
            ax.autoscale_view(scaley=True)
            ax.legend(
                frameon=True,
                edgecolor="black",
                facecolor="white",
                fontsize=8,
                framealpha=1,
            )
        for j in range(len(available), len(axes)):
            axes[j].set_visible(False)
        plt.suptitle(
            "HMSPC Full Model — Training Dynamics (seed 42)",
            fontsize=12,
            fontweight="bold",
            y=1.01,
        )
        plt.tight_layout()
        save_fig(fig, "fig2_full_model_diagnostics.png", GRAPHS_DIR)
else:
    print("  Skipped - seed 42 history not found")

# fig 3 - kl divergence across ablations
print("[Fig 3] KL divergence across ablations...")
fig, ax = plt.subplots(figsize=(6.5, 4.5))
any_plotted = False
for label, (pattern, color) in ABLATION_RUNS.items():
    hists = load_seed_histories(pattern, SAVES_DIR, SEEDS)
    if not hists:
        continue
    epochs, mean = mean_across_seeds(hists, "val_kl")
    if epochs is None:
        continue
    ax.plot(epochs, mean, color=color, linewidth=1.8, label=label)
    any_plotted = True
if any_plotted:
    styled_ax(ax, "KL Divergence Across Ablations", "Epoch", "KL")
    ax.set_ylim(bottom=0)
    set_ylim_to_ignore_outliers(ax)
    ax.legend(
        frameon=True, edgecolor="black", facecolor="white", fontsize=8, framealpha=1
    )
    save_fig(fig, "fig3_kl_divergence_ablations.png", GRAPHS_DIR)
else:
    plt.close(fig)
    print("  Skipped - no data")


# fig 4 - seed stability, one line per seed
print("[Fig 4] Full model seed stability...")
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
seed_colors = ["#e05a47", "#c0392b", "#922b21", "#7b241c", "#641e16"]
any_plotted = False
for i, s in enumerate(SEEDS):
    h = load_history(os.path.join(SAVES_DIR, f"hmspc_sdn1_cdi1_ug1_s{s}_history.json"))
    if h is None:
        continue
    vr = np.array(h.get("val_recon", h["val_loss"]))
    ep = range(1, len(h["train_loss"]) + 1)
    axes[0].plot(
        ep, smooth(vr.tolist()), color=seed_colors[i], linewidth=1.5, label=f"Seed {s}"
    )
    axes[1].plot(
        ep, smooth(h["val_kl"]), color=seed_colors[i], linewidth=1.5, label=f"Seed {s}"
    )
    any_plotted = True
if any_plotted:
    styled_ax(axes[0], "Val Reconstruction (per seed)", "Epoch", "NLL")
    styled_ax(axes[1], "Val KL Divergence (per seed)", "Epoch", "KL")
    axes[1].set_ylim(bottom=0)
    set_ylim_to_ignore_outliers(axes[0])
    set_ylim_to_ignore_outliers(axes[1])
    for ax in axes:
        ax.legend(
            frameon=True, edgecolor="black", facecolor="white", fontsize=8, framealpha=1
        )
    plt.suptitle("HMSPC Full Model — Seed Stability", fontsize=12, fontweight="bold")
    plt.tight_layout()
    save_fig(fig, "fig4_full_model_seed_stability.png", GRAPHS_DIR)
else:
    plt.close(fig)
    print("  Skipped - no data")

# fig 5 - gate ablation per-seed + mean
print("[Fig 5] Gate ablation stability...")
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
any_plotted = False
for label, (pattern, color) in GATE_RUNS.items():
    hists = load_seed_histories(pattern, SAVES_DIR, SEEDS)
    if not hists:
        print(f"  Skipping {label} - no histories found")
        continue
    for j, h in enumerate(hists):
        ep = range(1, len(h["val_loss"]) + 1)
        axes[0].plot(
            ep,
            smooth(h["val_loss"]),
            color=color,
            linewidth=1.0,
            alpha=0.7,
            label=label if j == 0 else None,
        )
    epochs, mean = mean_across_seeds(hists, "val_loss")
    if epochs is not None:
        axes[1].plot(epochs, mean, color=color, linewidth=1.8, label=label)
    any_plotted = True
if any_plotted:
    styled_ax(axes[0], "Gate Ablation — Val Loss (per seed)", "Epoch", "Loss")
    styled_ax(axes[1], "Gate Ablation — Val Loss (mean)", "Epoch", "Loss")
    set_ylim_to_ignore_outliers(axes[0])
    set_ylim_to_ignore_outliers(axes[1])
    for ax in axes:
        ax.legend(
            frameon=True, edgecolor="black", facecolor="white", fontsize=8, framealpha=1
        )
    plt.suptitle(
        "Gated vs Hard Conditioning — Stability Comparison",
        fontsize=12,
        fontweight="bold",
    )
    plt.tight_layout()
    save_fig(fig, "fig5_gate_ablation_stability.png", GRAPHS_DIR)
else:
    plt.close(fig)
    print("  Skipped - no data")


# fig 6 - rmse bar chart
print("[Fig 6] RMSE bar chart...")
bar_configs = [
    ("Vanilla NODE", "vanilla_neural_ode_s{seed}_results.json", COLORS["vanilla"]),
    ("Latent ODE", "latent_ode_s{seed}_results.json", COLORS["latent_ode"]),
    ("w/o everything", "hmspc_sdn0_cdi0_ug1_s{seed}_results.json", COLORS["no_nll"]),
    ("w/o Σ_θ", "hmspc_sdn0_cdi1_ug1_s{seed}_results.json", COLORS["no_sdn"]),
    ("w/o drift cond.", "hmspc_sdn1_cdi0_ug1_s{seed}_results.json", COLORS["no_cdi"]),
    (
        "Hard cond. (no gate)",
        "hmspc_sdn1_cdi1_ug0_s{seed}_results.json",
        COLORS["hard_cond"],
    ),
    ("HMSPC (full)", "hmspc_sdn1_cdi1_ug1_s{seed}_results.json", COLORS["full"]),
]
bar_labels, bar_means, bar_stds, bar_colors = [], [], [], []
for label, pattern, color in bar_configs:
    vals = load_results_metric(pattern, "rmse_mv", SEEDS, SAVES_DIR)
    if vals:
        bar_labels.append(label)
        bar_means.append(np.mean(vals))
        bar_stds.append(np.std(vals))
        bar_colors.append(color)
if bar_labels:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(bar_labels))
    ax.bar(
        x,
        bar_means,
        yerr=bar_stds,
        color=bar_colors,
        width=0.55,
        capsize=4,
        error_kw={"linewidth": 1.2},
        edgecolor="black",
        linewidth=0.7,
    )

    # dashed reference line at hmspc full mean
    full_idx = bar_labels.index("HMSPC (full)")
    full_mean = bar_means[full_idx]
    ax.axhline(
        full_mean,
        color=COLORS["full"],
        linewidth=1.2,
        linestyle="--",
        alpha=0.7,
        label=f"HMSPC (full) mean ({full_mean:.1f} mV)",
    )

    # shaded regions for baselines vs ablations
    n_baselines = 2
    n_ablations = len(bar_labels) - n_baselines - 1
    ax.axvspan(-0.5, n_baselines - 0.5, color="#f0f0f0", zorder=0, label="Baselines")
    ax.axvspan(
        n_baselines - 0.5,
        len(bar_labels) - 1.5,
        color="#fff8f0",
        zorder=0,
        label="Ablations",
    )

    ax.text(
        (n_baselines - 1) / 2,
        ax.get_ylim()[1] * 0.97,
        "Baselines",
        ha="center",
        va="top",
        fontsize=8,
        color="#666666",
        style="italic",
    )
    ax.text(
        n_baselines + (n_ablations - 1) / 2,
        ax.get_ylim()[1] * 0.97,
        "Ablations",
        ha="center",
        va="top",
        fontsize=8,
        color="#666666",
        style="italic",
    )

    ax.set_xticks(x)
    ax.set_xticklabels(bar_labels, fontsize=9, rotation=20, ha="right")
    ax.set_ylabel("RMSE (mV)", fontsize=10)
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.legend(
        frameon=True, edgecolor="black", facecolor="white", fontsize=8, framealpha=1
    )
    styled_ax(ax, "RMSE Across Model Configurations (± std)", "Model", "RMSE (mV)")
    plt.tight_layout()
    save_fig(fig, "fig6_rmse_bar_chart.png", GRAPHS_DIR)
else:
    plt.close(fig)
    print("  Skipped - no results JSONs found")

print("\nDone. Figures saved to", GRAPHS_DIR)
