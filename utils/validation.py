import numpy as np
from tqdm import tqdm


def hms_dataset_sanity_check(dataset, num_samples=500):
    print("\nHMS-PC Data Validation")
    print("----------------------")

    assert len(dataset) > 0, "FATAL: Dataset is empty."

    indices = np.random.choice(
        len(dataset), min(num_samples, len(dataset)), replace=False
    )

    # hard fail counters
    nan_count = 0
    inf_count = 0
    inverted_time = 0
    window_length_mismatch = 0

    # soft fail counters
    dt_nonpositive = 0
    dt_mismatch = 0
    dt_spike = 0
    voltage_violation = 0
    low_variance_voltage = 0
    current_violation = 0
    temp_outlier = 0

    # global ranges
    v_min, v_max = float("inf"), float("-inf")
    dt_min, dt_max = float("inf"), float("-inf")
    t_min, t_max = float("inf"), float("-inf")
    i_min, i_max = float("inf"), float("-inf")
    temp_min, temp_max = float("inf"), float("-inf")

    sample_temp_outliers = []

    sentinel_flags = {
        "neg270": 0,
        "clipped_high_400": 0,
        "extreme_range_spread": 0,
    }

    for idx in tqdm(indices, desc="Validating"):

        sample = dataset[idx]

        # key presence - chem_id removed (single chemistry dataset)
        required_keys = ["cell_id", "t", "dt", "V", "I", "T"]
        for k in required_keys:
            assert k in sample, f"Missing key: {k}"

        t = sample["t"]
        dt = sample["dt"]
        V = sample["V"]
        I = sample["I"]
        T = sample["T"]

        t_np = t.detach().cpu().numpy()
        dt_np = dt.detach().cpu().numpy()
        V_np = V.detach().cpu().numpy()
        I_np = I.detach().cpu().numpy()
        T_np = T.detach().cpu().numpy()

        # shape consistency
        lengths = [len(t_np), len(dt_np), len(V_np), len(I_np), len(T_np)]
        if len(set(lengths)) != 1:
            window_length_mismatch += 1

        # hard fails
        if (
            np.isnan(t_np).any()
            or np.isnan(V_np).any()
            or np.isnan(T_np).any()
            or np.isnan(I_np).any()
            or np.isnan(dt_np).any()
        ):
            nan_count += 1

        if (
            np.isinf(t_np).any()
            or np.isinf(V_np).any()
            or np.isinf(T_np).any()
            or np.isinf(I_np).any()
            or np.isinf(dt_np).any()
        ):
            inf_count += 1

        if len(t_np) > 1 and np.any(np.diff(t_np) < 0):
            inverted_time += 1

        # dt[0] must be exactly 0.0 (neutral placeholder)
        if dt_np[0] != 0.0:
            dt_nonpositive += 1

        # dt[1:] must all be positive
        if len(dt_np) > 1 and np.any(dt_np[1:] <= 0):
            dt_nonpositive += 1

        # dt consistency with t
        reconstructed_dt = np.diff(t_np, prepend=t_np[0])
        reconstructed_dt[0] = 0.0
        if len(dt_np) > 2 and not np.allclose(
            dt_np[1:], reconstructed_dt[1:], atol=1e-3
        ):
            dt_mismatch += 1

        # dt spike - large jumps destabilise sde integration
        if len(dt_np) > 1 and np.max(dt_np[1:]) > 1e3:
            dt_spike += 1

        # voltage physical range (lfp safe envelope)
        if np.any(V_np < 1.5) or np.any(V_np > 4.8):
            voltage_violation += 1

        # voltage variance - flat windows are uninformative for training
        if np.std(V_np) < 1e-4:
            low_variance_voltage += 1

        # current physical range
        if np.any(np.abs(I_np) > 25.0):
            current_violation += 1

        # temperature physical range (tightened: chamber at 30°C, fast-charge rise allowed)
        T_min_s, T_max_s = float(T_np.min()), float(T_np.max())

        if T_min_s < 15 or T_max_s > 60:
            temp_outlier += 1
            sample_temp_outliers.append((idx, T_min_s, T_max_s))

        # sentinel detection
        if np.any(T_np == -270.0):
            sentinel_flags["neg270"] += 1

        if np.any(T_np >= 399.0):
            sentinel_flags["clipped_high_400"] += 1

        if (T_max_s - T_min_s) > 250:
            sentinel_flags["extreme_range_spread"] += 1

        # global ranges
        v_min = min(v_min, np.min(V_np))
        v_max = max(v_max, np.max(V_np))
        dt_min = min(dt_min, np.min(dt_np))
        dt_max = max(dt_max, np.max(dt_np))
        t_min = min(t_min, np.min(t_np))
        t_max = max(t_max, np.max(t_np))
        i_min = min(i_min, np.min(I_np))
        i_max = max(i_max, np.max(I_np))
        temp_min = min(temp_min, T_min_s)
        temp_max = max(temp_max, T_max_s)

    # report
    print(f"\nSamples checked: {len(indices)}")

    print("\nHard fails")
    print(f"  NaNs:              {nan_count}")
    print(f"  Infs:              {inf_count}")
    print(f"  Inverted time:     {inverted_time}")
    print(f"  Shape mismatch:    {window_length_mismatch}")

    print("\nSoft fails")
    print(f"  Non-positive dt:   {dt_nonpositive}")
    print(f"  dt mismatch:       {dt_mismatch}")
    print(f"  dt spikes:         {dt_spike}")
    print(f"  Voltage range:     {voltage_violation}")
    print(f"  Voltage flat:      {low_variance_voltage}")
    print(f"  Current range:     {current_violation}")
    print(f"  Temperature range: {temp_outlier}")

    print("\nSentinel detection")
    for k, v in sentinel_flags.items():
        print(f"  {k}: {v}")

    print("\nGlobal ranges")
    print(f"  Voltage:     {v_min:.3f} -> {v_max:.3f} V")
    print(f"  Current:     {i_min:.3f} -> {i_max:.3f} A")
    print(f"  dt:          {dt_min:.6f} -> {dt_max:.6f} s")
    print(f"  Temperature: {temp_min:.2f} -> {temp_max:.2f} °C")

    print("\nFlagged temperature outliers")
    for item in sample_temp_outliers[:10]:
        print(f"  sample {item[0]}: {item[1]:.2f} → {item[2]:.2f} °C")
    if len(sample_temp_outliers) == 0:
        print(f"  nil")

    # final asserts
    assert nan_count == 0, "NaNs found in dataset"
    assert inf_count == 0, "Infs found in dataset"
    assert inverted_time == 0, "Inverted time detected"
    assert window_length_mismatch == 0, "Signal length mismatch within samples"
    assert dt_spike == 0, "dt spikes exceed sde integration threshold"
    assert sentinel_flags["neg270"] == 0, "Sentinel -270 values survived cleaning"
    assert (
        sentinel_flags["clipped_high_400"] == 0
    ), "Sentinel 400 values survived cleaning"

    print("\nDataset passed HMS-PC validation")
