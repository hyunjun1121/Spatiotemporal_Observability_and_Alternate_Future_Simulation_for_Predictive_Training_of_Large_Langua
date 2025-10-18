from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Callable, Dict, Iterable, Iterator, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, IterableDataset

from .noise import apply_noise


class ToyBPETokenizer:
    """Minimal byte-level tokenizer used when no external tokenizer is available."""

    def __init__(self, vocab_size: int = 50257) -> None:
        self.vocab_size = vocab_size

    def encode(self, text: str) -> list[int]:
        byte_values = text.encode("utf-8", errors="ignore")
        if not byte_values:
            return [0]
        return [int(b) % (self.vocab_size - 1) + 1 for b in byte_values]


toy_tokenizer = ToyBPETokenizer()


class SyntheticLM(IterableDataset):
    """Synthetic LM 데이터셋: 무작위 토큰 시퀀스를 생성한다."""

    def __init__(self, noise_cfg: Dict, batch_size: int, seq_len: int, vocab_size: int = 50257):
        self.noise_cfg = noise_cfg
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        self._step = 0

    def __iter__(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        while True:
            toks = np.random.randint(0, self.vocab_size, size=(self.batch_size, self.seq_len + 1), dtype=np.int64)
            toks = apply_noise(toks, self.noise_cfg, self._step, self.vocab_size)
            x = torch.from_numpy(toks[:, :-1])
            y = torch.from_numpy(toks[:, 1:])
            self._step += 1
            yield x, y


class StreamingLM(IterableDataset):
    """Tokenizes an incoming text stream into LM training batches."""

    def __init__(
        self,
        text_iter_factory: Callable[[], Iterable[str]],
        batch_size: int,
        seq_len: int,
        vocab_size: int = 50257,
    ) -> None:
        self.text_iter_factory = text_iter_factory
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.vocab_size = vocab_size

    def __iter__(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        buffer: list[int] = []
        batch_x: list[torch.Tensor] = []
        batch_y: list[torch.Tensor] = []
        while True:
            for text in self.text_iter_factory():
                buffer.extend(toy_tokenizer.encode(text))
                while len(buffer) >= self.seq_len + 1:
                    chunk = buffer[: self.seq_len + 1]
                    buffer = buffer[self.seq_len + 1 :]
                    arr = torch.tensor(chunk, dtype=torch.long)
                    batch_x.append(arr[:-1])
                    batch_y.append(arr[1:])
                    if len(batch_x) == self.batch_size:
                        yield torch.stack(batch_x), torch.stack(batch_y)
                        batch_x.clear()
                        batch_y.clear()
            # restart stream if it exhausts


def _hf_text_factory(name: str, real_data: str) -> Callable[[], Iterable[str]]:
    try:
        from datasets import load_dataset  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("datasets 패키지가 필요합니다. 설치 또는 offline 모드를 사용하세요.") from exc

    split = "train"
    if real_data == "wikitext":
        dataset = load_dataset(name, split=split, streaming=True)
        return lambda: (ex.get("text", "") or "" for ex in dataset)
    if real_data == "c4":
        dataset = load_dataset(name, "en", split=split, streaming=True)
        return lambda: (ex.get("text", "") or "" for ex in dataset)
    dataset = load_dataset(name, split=split, streaming=True)
    return lambda: (ex.get("text", "") or "" for ex in dataset)


def _iter_offline_jsonl(root: Path, split: str) -> Iterator[str]:
    for path in sorted(root.glob(f"{split}*.jsonl")):
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    continue
                text = data.get("text") or data.get("content")
                if text:
                    yield text


def _iter_offline_txt(root: Path, split: str) -> Iterator[str]:
    for path in sorted(root.glob(f"{split}*.txt")):
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    yield line


def build_dataloader(
    mode: str,
    noise_cfg: Dict,
    batch_size: int,
    seq_len: int,
    real_data: str = "off",
    hf_name: Optional[str] = None,
    offline_data_dir: Optional[str] = None,
) -> DataLoader:
    """모드와 설정에 맞는 DataLoader를 생성한다."""

    if mode.lower() in ("synthetic", "syn") or real_data == "off":
        ds = SyntheticLM(noise_cfg, batch_size=batch_size, seq_len=seq_len)
        return DataLoader(ds, batch_size=None)

    if mode.lower() == "hf":
        if offline_data_dir:
            root = Path(offline_data_dir)
            if not root.exists():
                raise FileNotFoundError(f"offline_data_dir not found: {root}")

            def factory() -> Iterable[str]:
                iterator = _iter_offline_jsonl(root, "train")
                first = next(iterator, None)
                if first is not None:
                    yield first
                    yield from iterator
                    return
                iterator = _iter_offline_txt(root, "train")
                for text in iterator:
                    yield text

            ds = StreamingLM(factory, batch_size=batch_size, seq_len=seq_len)
            return DataLoader(ds, batch_size=None)

        if hf_name is None:
            hf_name = "wikitext-103" if real_data == "wikitext" else "c4"
        text_factory = _hf_text_factory(hf_name, real_data)
        ds = StreamingLM(text_factory, batch_size=batch_size, seq_len=seq_len)
        return DataLoader(ds, batch_size=None)

    raise ValueError(f"Unsupported dataset mode: {mode}")
