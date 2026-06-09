#!/bin/bash

for seed in 42 100 7 6 10; do
    echo -e "Run: Latent ODE (seed $seed)"
    uv run train_latent_ode.py --seed $seed
    echo
done
