# HMSPC: Hybrid Multi-Scale Physical-Continuous Model for Battery Dynamics

Code for my article: [HMSPC: Hybrid Multi-Scale Physical-Continuous Model for Battery Dynamics Modeling](https://richardbarlian.github.io/research_blog/time-series/energy-systems/2026/03/02/HMSPC.html)

## Structure

- `scripts` folder - bash scripts for automated multi-seed training and ablations
- `models` folder - HMSPC, Latent ODE, and Vanilla NODE model definitions
- `utils` - data loading, evaluation, and plotting utilities

## Reproducing Results

```bash
uv sync
uv run load+preprocess_dataset.py # download Severson dataset and convert to .pkl
bash scripts/run_everything.sh
uv run process_ablations.py # process training runs
uv run analyze_flop.py # calculate params and FLOPs
```

or manually

```bash
uv sync
uv run load+preprocess_dataset.py # download Severson dataset and convert to .pkl
bash scripts/run_ode.sh
bash scripts/run_latent_ode.sh
bash scripts/run_ablation.sh
uv run process_ablations.py # process training runs
uv run analyze_flop.py # calculate params and FLOPs
```

## Known Issues

- `num_workers > 1` in DataLoader causes unintended effects so use `num_workers=0` for now (fix incoming)
