"""Random seeding without a dependency on a training framework."""

import random

import torch


def manual_seed(seed: int) -> None:
    """Seed Python and PyTorch, and NumPy when it is installed.

    This sets RNG seeds; it does not force deterministic accelerator kernels.
    """
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    try:
        import numpy as np
    except ImportError:
        return
    np.random.seed(seed)
