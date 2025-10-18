from __future__ import annotations
from typing import Tuple, Dict, Any
import math
import torch
import torch.nn as nn


class TinyGPTBlock(nn.Module):
    """간단한 GPT-like Block(MultiheadAttention + MLP)"""

    def __init__(self, d_model: int, n_head: int, mlp_ratio: int = 4, dropout: float = 0.0):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_head, dropout=dropout, batch_first=True)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * mlp_ratio), nn.GELU(), nn.Linear(d_model * mlp_ratio, d_model)
        )

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor, need_weights: bool = True):
        h = self.ln1(x)
        attn_out, attn_w = self.attn(h, h, h, attn_mask=attn_mask, average_attn_weights=False, need_weights=need_weights)
        x = x + attn_out
        h = self.ln2(x)
        x = x + self.mlp(h)
        return x, attn_w  # (B, T, C), (B, num_heads, T, S)


class TinyGPTLM(nn.Module):
    """소형 GPT 언어모델(Attention/Activation entropy 수집을 지원)."""

    def __init__(self, vocab_size: int = 50257, d_model: int = 256, n_head: int = 4, depth: int = 4, max_len: int = 1024):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos = nn.Embedding(max_len, d_model)
        self.blocks = nn.ModuleList([TinyGPTBlock(d_model, n_head) for _ in range(depth)])
        self.ln = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        self.max_len = max_len

    def forward(self, x: torch.Tensor, return_attentions: bool = False) -> Tuple[torch.Tensor, Dict[str, Any]]:
        B, T = x.shape
        device = x.device
        pos_ids = torch.arange(T, device=device).unsqueeze(0).expand(B, T)
        h = self.embed(x) + self.pos(pos_ids)
        # causal mask: (T, T) with -inf above diagonal; MultiheadAttention expects float mask add
        mask = torch.full((T, T), float("-inf"), device=device)
        mask = torch.triu(mask, diagonal=1)
        attns = []
        for blk in self.blocks:
            h, attn_w = blk(h, attn_mask=mask, need_weights=return_attentions)
            if return_attentions and attn_w is not None:
                # attn_w: (B, num_heads, T, T)
                attns.append(attn_w)
        h = self.ln(h)
        logits = self.head(h)
        aux: Dict[str, Any] = {}
        if return_attentions:
            aux["attentions"] = attns  # List[(B, H, T, T)]
            aux["hidden_last"] = h.detach()
        return logits, aux


def build_model(name: str = "gpt2-medium", precision: str = "bf16", output_attentions: bool = True) -> nn.Module:
    """모델을 생성한다. HF 미사용: TinyGPTLM으로 대체(SMOKE용)."""
    # 이름은 로깅용으로만 사용, 실제는 경량 모델 사용
    model = TinyGPTLM()
    if precision.lower() == "bf16" and torch.cuda.is_available():
        model = model.to(dtype=torch.bfloat16)
    return model


def build_optimizer(model: nn.Module, weight_decay: float = 0.01, lr: float = 3e-4) -> torch.optim.Optimizer:
    """AdamW Optimizer 구성."""
    return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay, betas=(0.9, 0.95), eps=1e-8)


class CosineWarmup(torch.optim.lr_scheduler._LRScheduler):
    """Cosine + Warmup 스케줄러."""

    def __init__(self, optimizer, warmup_steps: int, total_steps: int, last_epoch: int = -1):
        self.warmup_steps = warmup_steps
        self.total_steps = max(total_steps, warmup_steps + 1)
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        step = self.last_epoch + 1
        if step < self.warmup_steps:
            scale = step / max(1, self.warmup_steps)
        else:
            prog = (step - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)
            scale = 0.5 * (1 + math.cos(math.pi * prog))
        return [base_lr * scale for base_lr in self.base_lrs]


def build_scheduler(optimizer: torch.optim.Optimizer, cfg: Dict[str, Any], total_steps: int) -> torch.optim.lr_scheduler._LRScheduler:
    """CosineWarmup 스케줄러 생성."""
    warmup = int(cfg.get("warmup_steps", 500))
    return CosineWarmup(optimizer, warmup_steps=warmup, total_steps=total_steps)

