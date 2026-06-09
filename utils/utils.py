import torch
import random
import numpy as np
import os


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def set_seed(seed: int = 42):
    # enforces absolute determinism across python, numpy, and pytorch framework backends
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # multi-GPU safety

    # enforce deterministic algorithms in PyTorch operations
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # set python environment seed for hashing reproducibility
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Global seed fixed at: {seed}")


def seed_worker(worker_id):
    # ensures numpy and random arrays within dataloader workers maintain the global seed
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def smooth(values, window=3):
    if len(values) < window:
        return values
    return [
        sum(values[max(0, i - window) : i + 1])
        / len(values[max(0, i - window) : i + 1])
        for i in range(len(values))
    ]


def infer_model_type(filename):
    """
    Infers the architecture key string based on filename string flags or
    saved configuration metadata dictionaries.
    """
    fn_lower = filename.lower()
    if "hmspc" in fn_lower:
        return "hmspc"
    if "latent_ode" in fn_lower:
        return "latent_ode"
    if "vanilla" in fn_lower:
        return "vanilla"

    raise ValueError(
        f"Could not automatically infer model type for {filename}. "
        "Ensure filename contains 'hmspc', 'latent_ode', or 'vanilla'."
    )
