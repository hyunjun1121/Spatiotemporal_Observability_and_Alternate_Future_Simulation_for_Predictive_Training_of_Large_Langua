from __future__ import annotations

import numpy as np
import torch

from src.utils.seed import set_all_seeds


def test_set_all_seeds_reproducible() -> None:
    set_all_seeds(123)
    a1 = np.random.rand(5)
    t1 = torch.randn(3)

    set_all_seeds(123)
    a2 = np.random.rand(5)
    t2 = torch.randn(3)

    assert np.allclose(a1, a2)
    assert torch.allclose(t1, t2)

