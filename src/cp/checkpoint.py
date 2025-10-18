from typing import Dict, Any
import io
import torch

try:
    import zstandard as zstd
except ImportError:  # pragma: no cover
    zstd = None

def save_state(path: str, state: Dict[str, Any]) -> None:
    """Save full training universe state with zstd and optional 8-bit moments."""
    if zstd is None:
        torch.save(state, path)
        return
    buf = io.BytesIO()
    torch.save(state, buf)
    cctx = zstd.ZstdCompressor(level=9)
    with open(path, "wb") as f:
        f.write(cctx.compress(buf.getvalue()))

def load_state(path: str) -> Dict[str, Any]:
    """Load full training universe state from zstd-compressed binary."""
    if zstd is None:
        return torch.load(path, map_location="cpu", weights_only=False)
    with open(path, "rb") as f:
        data = zstd.ZstdDecompressor().decompress(f.read())
    return torch.load(io.BytesIO(data), map_location="cpu", weights_only=False)
