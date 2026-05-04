"""
gpt_model.py — Modern LLaMA-style transformer implementation.

Architecture upgrades over GPT-2:
  - RoPE positional encoding     (no learned position embeddings, better long-context)
  - RMSNorm                      (faster & simpler than LayerNorm)
  - SwiGLU feed-forward          (more expressive than GELU-FFN)
  - Grouped Query Attention (GQA)(fewer KV parameters, lower memory bandwidth)
  - Flash Attention via SDPA     (uses hardware-accelerated kernels on RTX 5090)

Compatible with GPT-2 tokenizer (vocab_size=50257).
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import tiktoken
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


# ---------------------------------------------------------------------------
# Model configurations
# ---------------------------------------------------------------------------

# Legacy GPT-2 style (updated key names)
GPT_CONFIG_124M: Dict = {
    "vocab_size": 50257,
    "context_length": 256,
    "emb_dim": 768,
    "n_heads": 12,
    "n_kv_heads": 12,           # == n_heads → standard MHA, no GQA
    "n_layers": 12,
    "intermediate_size": 3072,  # 4 × emb_dim
    "drop_rate": 0.1,
    "rope_theta": 10000.0,
    "norm_eps": 1e-5,
}

# Modern LLaMA-style configs — RoPE + RMSNorm + SwiGLU + GQA
# All three fit on RTX 5090 (32 GB) for training with mixed precision.

MODERN_CONFIG_370M: Dict = {    # LLaMA-style replacement for GPT-2 355M
    "vocab_size": 50257,
    "context_length": 4096,
    "emb_dim": 1024,
    "n_heads": 16,
    "n_kv_heads": 4,            # GQA: 4 KV heads → 4× KV memory reduction
    "n_layers": 28,
    "intermediate_size": 2816,  # ≈ 2.75 × emb_dim  (SwiGLU convention)
    "drop_rate": 0.0,
    "rope_theta": 10000.0,
    "norm_eps": 1e-5,
}

MODERN_CONFIG_1B: Dict = {
    "vocab_size": 50257,
    "context_length": 4096,
    "emb_dim": 2048,
    "n_heads": 16,
    "n_kv_heads": 4,
    "n_layers": 24,
    "intermediate_size": 5632,  # ≈ 2.75 × emb_dim
    "drop_rate": 0.0,
    "rope_theta": 10000.0,
    "norm_eps": 1e-5,
}

MODERN_CONFIG_3B: Dict = {      # ~3B params — comfortable on 5090 for training
    "vocab_size": 50257,
    "context_length": 4096,
    "emb_dim": 3072,
    "n_heads": 24,
    "n_kv_heads": 8,
    "n_layers": 28,
    "intermediate_size": 8192,
    "drop_rate": 0.0,
    "rope_theta": 10000.0,
    "norm_eps": 1e-5,
}


# ---------------------------------------------------------------------------
# RMSNorm
# ---------------------------------------------------------------------------

class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalisation (no mean-centering, no bias).

    Used in LLaMA, Mistral, Gemma. Faster and simpler than LayerNorm.
    """

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Upcast to float32 for numerical stability during mixed-precision training.
        # Casting back to the original dtype preserves downstream precision.
        x_f32 = x.float()
        norm = x_f32 * torch.rsqrt(x_f32.pow(2).mean(-1, keepdim=True) + self.eps)
        return (norm * self.weight).to(x.dtype)


# ---------------------------------------------------------------------------
# Rotary Position Embeddings (RoPE)
# ---------------------------------------------------------------------------

def precompute_rope_freqs(
    head_dim: int,
    max_seq_len: int,
    theta: float = 10000.0,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Precompute complex RoPE frequency tensor.

    Returns:
        Complex tensor of shape (max_seq_len, head_dim // 2).
    """
    freqs = 1.0 / (
        theta ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim)
    )
    t = torch.arange(max_seq_len, device=device)
    freqs = torch.outer(t, freqs)
    freqs_complex = torch.polar(torch.ones_like(freqs), freqs)  # e^{i * freq * t}
    # Return as real float32 (shape: max_seq_len, head_dim//2, 2) to avoid
    # dtype-cast issues when the model is moved to bf16/fp16.
    return torch.view_as_real(freqs_complex)


def _apply_rope_to_tensor(t: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    """Rotate a single query or key tensor by RoPE frequencies.

    Args:
        t:         Float tensor of shape (batch, heads, seq_len, head_dim).
        freqs_cis: Real float32 tensor of shape (seq_len, head_dim // 2, 2)
                   produced by :func:`precompute_rope_freqs`.

    Returns:
        Rotated tensor with the same shape and dtype as *t*.
    """
    # Reconstruct complex view from stored real representation
    freqs_complex = torch.view_as_complex(freqs_cis.float())  # (seq_len, head_dim//2)
    t_c = torch.view_as_complex(t.float().reshape(*t.shape[:-1], -1, 2))
    rotated = t_c * freqs_complex[None, None]   # (1, 1, seq_len, head_dim//2)
    return torch.view_as_real(rotated).flatten(-2).to(t.dtype)


def apply_rope(
    query: torch.Tensor,
    key: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply RoPE to query and key tensors.

    Args:
        query:     (batch, n_heads,    seq_len, head_dim)
        key:       (batch, n_kv_heads, seq_len, head_dim)
        freqs_cis: Real float32 tensor of shape (seq_len, head_dim // 2, 2).

    Returns:
        Rotated query and key with the same dtype as inputs.
    """
    return _apply_rope_to_tensor(query, freqs_cis), _apply_rope_to_tensor(key, freqs_cis)


# ---------------------------------------------------------------------------
# SwiGLU Feed-Forward
# ---------------------------------------------------------------------------

class FeedForward(nn.Module):
    """SwiGLU feed-forward network: down(SiLU(gate(x)) ⊙ up(x)).

    More expressive than GELU-FFN with similar parameter count.
    Used in LLaMA, Mistral, Gemma, PaLM.
    """

    def __init__(self, cfg: Dict) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(cfg["emb_dim"], cfg["intermediate_size"], bias=False)
        self.up_proj   = nn.Linear(cfg["emb_dim"], cfg["intermediate_size"], bias=False)
        self.down_proj = nn.Linear(cfg["intermediate_size"], cfg["emb_dim"],  bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


# ---------------------------------------------------------------------------
# Grouped Query Attention (GQA) with Flash Attention
# ---------------------------------------------------------------------------

class MultiHeadAttention(nn.Module):
    """Causal multi-head attention with GQA and RoPE.

    When n_kv_heads == n_heads this is standard MHA.
    When n_kv_heads < n_heads, key/value heads are shared across groups of
    query heads (GQA), reducing KV-cache memory and bandwidth.

    Uses torch.nn.functional.scaled_dot_product_attention for Flash Attention
    on supported hardware (RTX 3090+, A100, H100, RTX 5090).
    """

    def __init__(self, cfg: Dict) -> None:
        super().__init__()
        if cfg["emb_dim"] % cfg["n_heads"] != 0:
            raise ValueError(
                f"emb_dim ({cfg['emb_dim']}) must be divisible by "
                f"n_heads ({cfg['n_heads']})."
            )
        if cfg["n_heads"] % cfg["n_kv_heads"] != 0:
            raise ValueError(
                f"n_heads ({cfg['n_heads']}) must be divisible by "
                f"n_kv_heads ({cfg['n_kv_heads']})."
            )

        self.n_heads    = cfg["n_heads"]
        self.n_kv_heads = cfg["n_kv_heads"]
        self.head_dim   = cfg["emb_dim"] // cfg["n_heads"]

        if self.head_dim % 2 != 0:
            raise ValueError(
                f"head_dim ({self.head_dim}) must be even for RoPE "
                f"(emb_dim={cfg['emb_dim']}, n_heads={cfg['n_heads']})."
            )
        self.n_rep      = self.n_heads // self.n_kv_heads   # GQA repeat factor
        self.dropout_p  = cfg["drop_rate"]

        self.q_proj = nn.Linear(cfg["emb_dim"], self.n_heads    * self.head_dim, bias=False)
        self.k_proj = nn.Linear(cfg["emb_dim"], self.n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(cfg["emb_dim"], self.n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(cfg["emb_dim"], cfg["emb_dim"],                  bias=False)

    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape

        query = self.q_proj(x).view(batch_size, seq_len, self.n_heads,    self.head_dim).transpose(1, 2)
        key   = self.k_proj(x).view(batch_size, seq_len, self.n_kv_heads, self.head_dim).transpose(1, 2)
        value = self.v_proj(x).view(batch_size, seq_len, self.n_kv_heads, self.head_dim).transpose(1, 2)

        query, key = apply_rope(query, key, freqs_cis)

        # Expand KV heads to match Q heads for GQA
        if self.n_rep > 1:
            key   = key.repeat_interleave(self.n_rep, dim=1)
            value = value.repeat_interleave(self.n_rep, dim=1)

        # Flash Attention — causal mask handled internally, no manual triu needed
        out = F.scaled_dot_product_attention(
            query, key, value,
            dropout_p=self.dropout_p if self.training else 0.0,
            is_causal=True,
        )

        out = out.transpose(1, 2).reshape(batch_size, seq_len, -1)
        return self.o_proj(out)


# ---------------------------------------------------------------------------
# Transformer Block
# ---------------------------------------------------------------------------

class TransformerBlock(nn.Module):
    """Pre-norm transformer block (RMSNorm + GQA + SwiGLU with residuals)."""

    def __init__(self, cfg: Dict) -> None:
        super().__init__()
        self.attention         = MultiHeadAttention(cfg)
        self.feed_forward      = FeedForward(cfg)
        self.attention_norm    = RMSNorm(cfg["emb_dim"], eps=cfg["norm_eps"])
        self.feed_forward_norm = RMSNorm(cfg["emb_dim"], eps=cfg["norm_eps"])
        self.attention_dropout = nn.Dropout(cfg["drop_rate"])
        self.ff_dropout        = nn.Dropout(cfg["drop_rate"])

    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.attention(self.attention_norm(x), freqs_cis)
        x = self.attention_dropout(x) + residual

        residual = x
        x = self.feed_forward(self.feed_forward_norm(x))
        x = self.ff_dropout(x) + residual

        return x


# ---------------------------------------------------------------------------
# Dataset & DataLoader
# ---------------------------------------------------------------------------

class TransformerDataset(Dataset):
    """Lazy-loading causal language modelling dataset.

    Token IDs are stored once as a list; tensors are created on-the-fly in
    ``__getitem__`` to avoid pre-allocating O(N) tensors in memory.
    """

    def __init__(self, text: str, tokenizer, max_length: int, stride: int = -1) -> None:
        super().__init__()
        if not text:
            raise ValueError("Input text must not be empty.")
        if max_length < 1:
            raise ValueError(f"max_length must be >= 1, got {max_length}.")
        if stride == -1:
            stride = max_length  # Non-overlapping windows by default
        if stride < 1:
            raise ValueError(f"stride must be >= 1, got {stride}.")

        self._token_ids: list = tokenizer.encode(text)
        self.max_length = max_length
        self.stride     = stride

        # Need max_length input tokens + 1 shifted target token beyond them
        num_samples = (len(self._token_ids) - max_length - 1) // stride + 1
        if num_samples <= 0:
            raise ValueError(
                f"Text too short: {len(self._token_ids)} tokens for "
                f"max_length={max_length}. Need at least {max_length + 2} tokens."
            )
        self._num_samples = num_samples

    def __len__(self) -> int:
        return self._num_samples

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        start = index * self.stride
        end   = start + self.max_length
        input_ids  = torch.tensor(self._token_ids[start:end],         dtype=torch.long)
        target_ids = torch.tensor(self._token_ids[start + 1:end + 1], dtype=torch.long)
        return input_ids, target_ids


def create_dataloader(
    txt: str,
    batch_size: int = 4,
    max_length: int = 256,
    stride: int = -1,
    shuffle: bool = True,
    drop_last: bool = True,
    num_workers: int = 0,
) -> DataLoader:
    """Create a DataLoader for causal language modelling."""
    tokenizer = tiktoken.get_encoding("gpt2")
    dataset   = TransformerDataset(txt, tokenizer, max_length, stride)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
    )


# ---------------------------------------------------------------------------
# GPT Model  (modern LLaMA-style — no learned position embeddings)
# ---------------------------------------------------------------------------

class GPTModel(nn.Module):
    """Decoder-only language model with RoPE, RMSNorm, SwiGLU, and GQA.

    Position information is encoded via RoPE inside each attention layer;
    there are no learned position embeddings.
    """

    def __init__(self, cfg: Dict) -> None:
        super().__init__()
        self.cfg = cfg

        self.token_embedding   = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.embedding_dropout = nn.Dropout(cfg["drop_rate"])
        self.transformer_blocks = nn.ModuleList(
            [TransformerBlock(cfg) for _ in range(cfg["n_layers"])]
        )
        self.final_norm = RMSNorm(cfg["emb_dim"], eps=cfg["norm_eps"])
        self.lm_head    = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)

        # Precompute RoPE frequencies; kept as a non-persistent buffer
        # so it moves with the model when .to(device) is called
        head_dim = cfg["emb_dim"] // cfg["n_heads"]
        # Store RoPE frequencies as real float32 (shape: context_length, head_dim//2, 2).
        # Using view_as_real avoids dtype-cast warnings when moving model to bf16/fp16
        # since PyTorch cannot represent complex numbers in those dtypes.
        self.register_buffer(
            "freqs_cis",
            precompute_rope_freqs(head_dim, cfg["context_length"], cfg["rope_theta"]),
            persistent=False,
        )

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Compute logits for the input token sequence.

        Args:
            input_ids: Long tensor of shape (batch_size, seq_len).

        Returns:
            Float tensor of shape (batch_size, seq_len, vocab_size).

        Raises:
            ValueError: If seq_len exceeds context_length.
        """
        batch_size, seq_len = input_ids.shape
        if seq_len > self.cfg["context_length"]:
            raise ValueError(
                f"Sequence length {seq_len} exceeds context_length "
                f"{self.cfg['context_length']}."
            )

        x         = self.embedding_dropout(self.token_embedding(input_ids))
        freqs_cis = self.freqs_cis[:seq_len]

        for block in self.transformer_blocks:
            x = block(x, freqs_cis)

        return self.lm_head(self.final_norm(x))


# ---------------------------------------------------------------------------
# Text generation
# ---------------------------------------------------------------------------

def generate_text_simple(
    model: nn.Module,
    token_ids: torch.Tensor,
    max_new_tokens: int,
    context_size: int,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
) -> torch.Tensor:
    """Autoregressively generate tokens from a prompt.

    Args:
        model:          GPT model. Temporarily set to eval mode; original
                        training/eval state is restored on return.
        token_ids:      Prompt token indices of shape (batch_size, seq_len).
        max_new_tokens: Number of tokens to generate.
        context_size:   Maximum context length the model supports.
        temperature:    Sampling temperature; must be > 0.
        top_k:          If set, restrict sampling to the top-k logits.

    Raises:
        ValueError: If *temperature* <= 0 or *top_k* <= 0.
    """
    if temperature <= 0:
        raise ValueError(f"temperature must be > 0, got {temperature}.")
    if top_k is not None and top_k <= 0:
        raise ValueError(f"top_k must be > 0, got {top_k}.")

    was_training = model.training
    model.eval()

    for _ in range(max_new_tokens):
        context = token_ids[:, -context_size:]
        with torch.no_grad():
            logits = model(context)[:, -1, :]  # (batch, vocab_size)

        if temperature != 1.0:
            logits = logits / temperature

        if top_k is not None:
            top_values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits = logits.masked_fill(logits < top_values[:, -1:], float("-inf"))

        probs      = torch.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        token_ids  = torch.cat((token_ids, next_token), dim=1)

    if was_training:
        model.train()
    return token_ids
