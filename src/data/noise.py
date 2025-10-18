from __future__ import annotations
from typing import Dict
import numpy as np

def apply_noise(tokens: np.ndarray, noise_cfg: Dict, step: int, vocab_size: int) -> np.ndarray:
    """토큰 시퀀스(batch, seq_len)에 다양한 synthetic 노이즈를 주입한다."""
    x = tokens.copy()
    bs, L = x.shape

    # rare_token_burst: 상위 구간의 토큰으로 치환
    p = float(noise_cfg.get("rare_token_burst_pct", 0.0))
    if p > 0:
        mask = np.random.rand(bs) < p
        if mask.any():
            # 상위 1% 영역에서 샘플
            rare_min = int(0.99 * vocab_size)
            x[mask] = np.random.randint(rare_min, vocab_size, size=(mask.sum(), L), dtype=x.dtype)

    # very_long_seq: 길이 증가 시뮬레이션(간단 근사: 내부 블록 반복)
    p = float(noise_cfg.get("very_long_seq_pct", 0.0))
    if p > 0:
        mask = np.random.rand(bs) < p
        for i in np.where(mask)[0]:
            half = L // 2
            x[i, half:] = x[i, :L-half]

    # unicode_artifact: 특수 토큰 영역으로 치환
    p = float(noise_cfg.get("unicode_artifact_pct", 0.0))
    if p > 0:
        mask = np.random.rand(bs) < p
        if mask.any():
            art_min = int(0.95 * vocab_size)
            x[mask, ::16] = np.random.randint(art_min, vocab_size, size=(mask.sum(), L//16 + (L%16>0)), dtype=x.dtype)

    # block_shuffle: intra-seq 블록 순서를 섞음
    ratio = float(noise_cfg.get("block_shuffle_ratio", 0.0))
    if ratio > 0:
        blk = max(1, int(L * ratio))
        for i in range(bs):
            # 블록 단위 섞기
            idxs = list(range(0, L, blk))
            np.random.shuffle(idxs)
            chunks = [x[i, j:j+blk] for j in idxs]
            x[i] = np.concatenate(chunks, axis=0)[:L]

    # warmup_spike: 학습 루프에서 LR warmup으로 반영(여기서는 pass)
    return x

