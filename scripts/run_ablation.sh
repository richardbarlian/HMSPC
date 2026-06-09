#!/bin/bash
set -e

for seed in 42 100 7 6 10; do
    echo "Run: HMS-PC Full Model (seed $seed)"
    uv run train_hms_pc.py --seed $seed --use_state_dep_noise 1 --condition_drift_on_inputs 1 --use_gate 1
    echo

    echo "Run: HMS-PC w/o Σ_θ (seed $seed)"
    uv run train_hms_pc.py --seed $seed --use_state_dep_noise 0 --condition_drift_on_inputs 1 --use_gate 1
    echo

    echo "Run: HMS-PC w/o drift cond. (seed $seed)"
    uv run train_hms_pc.py --seed $seed --use_state_dep_noise 1 --condition_drift_on_inputs 0 --use_gate 1
    echo

    echo "Run: HMS-PC w/o everything w/ NLL (seed $seed)"
    uv run train_hms_pc.py --seed $seed --use_state_dep_noise 0 --condition_drift_on_inputs 0 --use_gate 1
    echo

    echo "Run: HMS-PC hard conditioning no gate (seed $seed)"
    uv run train_hms_pc.py --seed $seed --use_state_dep_noise 1 --condition_drift_on_inputs 1 --use_gate 0
    echo
done