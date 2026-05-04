"""
Test suite for gpt_model.py — the modern LLaMA-style transformer.

Tests:
  1.  Architecture basics (parameter count, output shape)
  2.  RMSNorm stability in float32 and bfloat16
  3.  Input validation (emb_dim/head_dim, seq_len overflow)
  4.  RoPE encoding (freqs_cis dtype, shape, no complex buffers)
  5.  GQA correctness (KV head count differs from Q head count)
  6.  Flash Attention (SDPA path, causal mask)
  7.  Mixed-precision (bf16 forward, no NaN, no warnings)
  8.  Dataloader (shape, stride, token distribution)
  9.  Gradient flow (backward pass, no dead gradients)
  10. Text generation (greedy, sampling, temperature, top-k)
"""

import sys, os, warnings
sys.path.insert(0, os.path.dirname(__file__))

import torch
import torch.nn as nn

from gpt_model import (
    GPT_CONFIG_124M,
    MODERN_CONFIG_370M,
    RMSNorm,
    GPTModel,
    create_dataloader,
    generate_text_simple,
    precompute_rope_freqs,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PASS = "\033[32mPASS\033[0m"
FAIL = "\033[31mFAIL\033[0m"
_failures = []


def check(name: str, cond: bool, detail: str = ""):
    if cond:
        print(f"  [{PASS}] {name}")
    else:
        print(f"  [{FAIL}] {name}" + (f": {detail}" if detail else ""))
        _failures.append(name)


# ---------------------------------------------------------------------------
# Test 1 — Architecture basics
# ---------------------------------------------------------------------------
def test_architecture():
    print("\n=== TEST 1: Architecture basics ===")

    m = GPTModel(GPT_CONFIG_124M)
    params = sum(p.numel() for p in m.parameters())
    # 124M config should be roughly 85–165M params
    check("param count in reasonable range", 50_000_000 <= params <= 200_000_000,
          f"got {params/1e6:.1f}M")

    x = torch.randint(0, 50257, (2, 128))
    out = m(x)
    check("output shape", out.shape == (2, 128, 50257), str(out.shape))
    check("output dtype float32", out.dtype == torch.float32)

    m370 = GPTModel(MODERN_CONFIG_370M)
    p370 = sum(p.numel() for p in m370.parameters())
    check("370M param count in range",
          300_000_000 <= p370 <= 500_000_000, f"got {p370/1e6:.1f}M")
    print(f"     124M actual: {params/1e6:.1f}M | 370M actual: {p370/1e6:.1f}M")


# ---------------------------------------------------------------------------
# Test 2 — RMSNorm stability
# ---------------------------------------------------------------------------
def test_rmsnorm():
    print("\n=== TEST 2: RMSNorm stability ===")

    norm = RMSNorm(128, eps=1e-6)

    for dtype in (torch.float32, torch.bfloat16):
        x = torch.randn(4, 32, 128).to(dtype)
        out = norm(x)
        check(f"output dtype preserved ({dtype})", out.dtype == dtype)
        check(f"no NaN ({dtype})", not torch.isnan(out).any().item())

    # Extreme values — should not overflow even in bf16
    norm_bf16 = RMSNorm(64).to(torch.bfloat16)
    x_large = torch.full((2, 8, 64), 1e3, dtype=torch.bfloat16)
    out_large = norm_bf16(x_large)
    check("no NaN with large inputs in bf16", not torch.isnan(out_large).any().item())


# ---------------------------------------------------------------------------
# Test 3 — Input validation
# ---------------------------------------------------------------------------
def test_validation():
    print("\n=== TEST 3: Input validation ===")

    # emb_dim not divisible by n_heads
    try:
        GPTModel(dict(GPT_CONFIG_124M, emb_dim=770))
        check("emb_dim % n_heads validation", False, "no exception raised")
    except ValueError:
        check("emb_dim % n_heads validation", True)

    # odd head_dim (RoPE needs even)
    try:
        GPTModel(dict(GPT_CONFIG_124M, emb_dim=756))   # 756/12 = 63 (odd)
        check("odd head_dim validation", False, "no exception raised")
    except ValueError:
        check("odd head_dim validation", True)

    # sequence length exceeds context_length
    m = GPTModel(dict(GPT_CONFIG_124M, context_length=64))
    try:
        m(torch.randint(0, 50257, (1, 128)))
        check("seq_len > context_length raises", False, "no exception raised")
    except ValueError:
        check("seq_len > context_length raises", True)


# ---------------------------------------------------------------------------
# Test 4 — RoPE / freqs_cis
# ---------------------------------------------------------------------------
def test_rope():
    print("\n=== TEST 4: RoPE encoding ===")

    head_dim, seq_len = 64, 512
    freqs = precompute_rope_freqs(head_dim, seq_len, theta=10000.0)

    check("freqs_cis shape", freqs.shape == (seq_len, head_dim // 2, 2),
          str(freqs.shape))
    check("freqs_cis dtype float32", freqs.dtype == torch.float32)
    check("freqs_cis not complex", not freqs.is_complex())

    # After .to(bfloat16), buffer follows but stays real (no cast warning)
    with warnings.catch_warnings():
        warnings.filterwarnings("error")
        try:
            m = GPTModel(MODERN_CONFIG_370M).to(torch.bfloat16)
            check("no warning on .to(bfloat16)", True)
        except Warning as w:
            check("no warning on .to(bfloat16)", False, str(w))

    # Verify different positions produce different encodings
    m_cpu = GPTModel(GPT_CONFIG_124M).eval()
    x = torch.randint(0, 50257, (1, 16))
    out_a = m_cpu(x)
    # Same tokens shifted by 1 position should give different outputs
    out_b = m_cpu(torch.cat([x[:, 1:], x[:, :1]], dim=1))
    check("RoPE produces position-sensitive outputs",
          not torch.allclose(out_a, out_b, atol=1e-3))


# ---------------------------------------------------------------------------
# Test 5 — GQA (n_kv_heads < n_heads)
# ---------------------------------------------------------------------------
def test_gqa():
    print("\n=== TEST 5: Grouped Query Attention ===")

    cfg = dict(MODERN_CONFIG_370M, context_length=64)  # small for speed
    m = GPTModel(cfg).eval()

    n_heads    = cfg["n_heads"]
    n_kv_heads = cfg["n_kv_heads"]
    check("n_kv_heads < n_heads (GQA active)", n_kv_heads < n_heads,
          f"n_heads={n_heads}, n_kv_heads={n_kv_heads}")

    x = torch.randint(0, 50257, (1, 32))
    with torch.no_grad():
        out = m(x)
    check("GQA forward shape", out.shape == (1, 32, 50257), str(out.shape))
    check("GQA no NaN", not torch.isnan(out).any().item())


# ---------------------------------------------------------------------------
# Test 6 — Flash Attention / causal mask
# ---------------------------------------------------------------------------
def test_flash_attention():
    print("\n=== TEST 6: Flash Attention (SDPA) ===")

    m = GPTModel(GPT_CONFIG_124M).eval()
    x = torch.randint(0, 50257, (2, 64))
    with torch.no_grad():
        out = m(x)
    check("SDPA forward runs without error", True)
    check("SDPA output shape", out.shape == (2, 64, 50257), str(out.shape))

    # Causal mask: logits at position i should be identical whether or not
    # tokens at positions > i are different (no future leakage).
    m.eval()
    base = torch.randint(0, 50257, (1, 16))
    modified = base.clone()
    modified[0, 8:] = torch.randint(0, 50257, (8,))
    with torch.no_grad():
        out_base = m(base)
        out_mod  = m(modified)
    check("causal mask: prefix logits unaffected by future tokens",
          torch.allclose(out_base[:, :8], out_mod[:, :8], atol=1e-4))


# ---------------------------------------------------------------------------
# Test 7 — Mixed precision (bf16)
# ---------------------------------------------------------------------------
def test_mixed_precision():
    print("\n=== TEST 7: Mixed precision (bf16) ===")

    if not torch.cuda.is_available():
        print("  [SKIP] No GPU available")
        return

    with warnings.catch_warnings():
        warnings.filterwarnings("error")
        try:
            cfg = dict(MODERN_CONFIG_370M, context_length=128)
            m = GPTModel(cfg).to(DEVICE).to(torch.bfloat16)
            x = torch.randint(0, 50257, (2, 64), device=DEVICE)
            with torch.no_grad():
                with torch.amp.autocast(DEVICE, dtype=torch.bfloat16):
                    out = m(x)
            check("bf16 forward no warning", True)
        except Warning as w:
            check("bf16 forward no warning", False, str(w))

    check("bf16 output no NaN", not torch.isnan(out).any().item())


# ---------------------------------------------------------------------------
# Test 8 — Dataloader
# ---------------------------------------------------------------------------
def test_dataloader():
    print("\n=== TEST 8: Dataloader ===")

    text = "hello world " * 2000   # ~24k chars
    max_length = 64

    dl = create_dataloader(text, batch_size=4, max_length=max_length,
                           stride=max_length, shuffle=False)
    check("dataloader has batches", len(dl) > 0, f"len={len(dl)}")

    x, y = next(iter(dl))
    check("input shape",  x.shape == (4, max_length), str(x.shape))
    check("target shape", y.shape == (4, max_length), str(y.shape))
    check("targets are inputs shifted by 1",
          torch.equal(x[0, 1:], y[0, :-1]))


# ---------------------------------------------------------------------------
# Test 9 — Gradient flow
# ---------------------------------------------------------------------------
def test_gradients():
    print("\n=== TEST 9: Gradient flow ===")

    m = GPTModel(GPT_CONFIG_124M)
    x = torch.randint(0, 50257, (2, 32))
    y = torch.randint(0, 50257, (2, 32))

    logits = m(x)
    loss = nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
    loss.backward()

    # Every parameter that requires grad must have a gradient
    missing = [n for n, p in m.named_parameters()
               if p.requires_grad and p.grad is None]
    check("all params have gradients", not missing,
          f"missing: {missing[:3]}" if missing else "")

    # No exploding gradients (norm < 1e4)
    total_norm = sum(p.grad.norm().item() ** 2 for p in m.parameters()
                     if p.grad is not None) ** 0.5
    check("gradient norm reasonable", total_norm < 1e4, f"norm={total_norm:.1f}")


# ---------------------------------------------------------------------------
# Test 10 — Text generation
# ---------------------------------------------------------------------------
def test_generation():
    print("\n=== TEST 10: Text generation ===")

    import tiktoken
    tokenizer = tiktoken.get_encoding("gpt2")
    m = GPTModel(GPT_CONFIG_124M).eval()
    prompt = tokenizer.encode("The quick brown fox")
    idx = torch.tensor([prompt])

    # Greedy (temperature=1.0, no top_k)
    out = generate_text_simple(m, idx, max_new_tokens=20,
                               context_size=GPT_CONFIG_124M["context_length"])
    check("greedy generation length", out.shape[1] == len(prompt) + 20, str(out.shape))

    # Temperature sampling
    out_t = generate_text_simple(m, idx, max_new_tokens=20,
                                 context_size=GPT_CONFIG_124M["context_length"],
                                 temperature=0.7)
    check("temperature sampling length", out_t.shape[1] == len(prompt) + 20)

    # top-k
    out_k = generate_text_simple(m, idx, max_new_tokens=20,
                                 context_size=GPT_CONFIG_124M["context_length"],
                                 temperature=0.8, top_k=40)
    check("top-k generation length", out_k.shape[1] == len(prompt) + 20)

    # Invalid temperature raises
    try:
        generate_text_simple(m, idx, max_new_tokens=5,
                             context_size=256, temperature=0.0)
        check("temperature=0 raises ValueError", False)
    except ValueError:
        check("temperature=0 raises ValueError", True)

    # Decode and print a sample
    text_out = tokenizer.decode(out_k[0].tolist())
    print(f"     sample: {text_out!r}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 60)
    print(f"  gpt_model.py — Modern LLaMA-style test suite")
    print(f"  device: {DEVICE}")
    print("=" * 60)

    test_architecture()
    test_rmsnorm()
    test_validation()
    test_rope()
    test_gqa()
    test_flash_attention()
    test_mixed_precision()
    test_dataloader()
    test_gradients()
    test_generation()

    print("\n" + "=" * 60)
    if _failures:
        print(f"  FAILED: {len(_failures)} test(s)")
        for f in _failures:
            print(f"    - {f}")
        sys.exit(1)
    else:
        print(f"  All tests passed.")
    print("=" * 60)
