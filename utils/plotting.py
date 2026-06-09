import os
import matplotlib.pyplot as plt
import numpy as np
import torch
import json


def init_style():
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["axes.edgecolor"] = "black"
    plt.rcParams["axes.linewidth"] = 0.8


def styled_ax(ax, title, xlabel, ylabel):
    ax.set_title(title, fontsize=11, pad=8, fontweight="bold")
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.grid(True, linestyle="--", linewidth=0.5, color="#e0e0e0")
    ax.tick_params(axis="both", labelsize=9, direction="in")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def smooth(scalars, weight=0.85):
    last = scalars[0]
    out = []
    for v in scalars:
        s = last * weight + (1 - weight) * v
        out.append(s)
        last = s
    return out


def save_fig(fig, name, output_dir="graphs"):
    path = os.path.join(output_dir, name)
    fig.savefig(path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"  Saved: {path}")


def load_history(path):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def load_seed_histories(pattern, saves_dir, seeds):
    hists = []
    for s in seeds:
        h = load_history(
            os.path.join(saves_dir, pattern.format(seed=s) + "_history.json")
        )
        if h is not None:
            hists.append(h)
    return hists


def mean_across_seeds(hists, key):
    arrays = [np.array(h[key]) for h in hists if key in h]
    if not arrays:
        return None, None
    min_len = min(len(a) for a in arrays)
    arrays = [a[:min_len] for a in arrays]
    arr = np.stack(arrays)
    return np.arange(1, min_len + 1), arr.mean(0)


def load_results_metric(pattern, metric_key, seeds, saves_dir):
    vals = []
    for s in seeds:
        path = os.path.join(saves_dir, pattern.format(seed=s))
        if not os.path.exists(path):
            continue
        with open(path) as f:
            r = json.load(f)
        v = r.get("metrics", {}).get(metric_key)
        if v is not None:
            vals.append(v)
    return vals


def set_ylim_to_ignore_outliers(ax, percentile=95, margin_factor=0.1):
    """clip upper y-limit to percentile of all line data"""
    y_vals = []
    for line in ax.get_lines():
        y_vals.extend(line.get_ydata())
    if not y_vals:
        return
    upper = np.percentile(y_vals, percentile) * (1 + margin_factor)
    if ax.get_ylim()[1] > upper:
        ax.set_ylim(top=upper)


def extract_2d_dynamics(model, model_type, latent_dim, grid_size=25, device="cpu"):
    """Evaluates the continuous vector field derivative over a 2D state space mesh grid."""
    model.eval()

    x = np.linspace(-3.0, 3.0, grid_size)
    y = np.linspace(-3.0, 3.0, grid_size)
    X, Y = np.meshgrid(x, y)

    h1_flat = X.ravel()
    h2_flat = Y.ravel()
    num_points = len(h1_flat)

    h_input = torch.zeros((num_points, latent_dim), device=device)
    h_input[:, 0] = torch.tensor(h1_flat, dtype=torch.float32)
    h_input[:, 1] = torch.tensor(h2_flat, dtype=torch.float32)

    with torch.no_grad():
        if model_type in ["vanilla", "latent_ode"]:
            dh_dt = model.ode_func(0, h_input)
        elif model_type == "hmspc":
            static_inputs = torch.zeros((num_points, 2), device=device)
            dh_dt = model.ode_func.f(h_input, static_inputs)

    U = dh_dt[:, 0].cpu().numpy().reshape(X.shape)
    V = dh_dt[:, 1].cpu().numpy().reshape(Y.shape)

    return X, Y, U, V
