from __future__ import annotations
import random
from typing import Optional

def set_all_seeds(seed: int, *, numpy: Optional[object] = None, torch: Optional[object] = None) -> None:
    """Python/numpy/torch 모든 RNG 시드를 고정한다."""
    random.seed(seed)
    try:
        import numpy as _np  # type: ignore
        (_np if numpy is None else numpy).random.seed(seed)
    except Exception:
        pass
    try:
        import torch as _torch  # type: ignore
        torch_mod = _torch if torch is None else torch
        torch_mod.manual_seed(seed)
        if hasattr(torch_mod, "cuda"):
            torch_mod.cuda.manual_seed_all(seed)
    except Exception:
        pass

