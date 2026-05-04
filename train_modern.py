"""
train_modern.py — Training script for gpt_model.py (LLaMA-style architecture).

Architecture: RoPE + RMSNorm + SwiGLU + GQA + Flash Attention (SDPA)
Default config: MODERN_CONFIG_370M (~370M parameters)
Optimized for: RTX 5090 (32 GB VRAM)

Usage:
    python chap7/train_modern.py                     # default 370M
    python chap7/train_modern.py --config 1b         # 1B variant
    python chap7/train_modern.py --resume chap7/ckpt.pth
"""

import argparse
import math
import os
import sys
import time

import torch
import torch.nn.functional as F
import torch.optim as optim

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from gpt_model import (
    GPT_CONFIG_124M,
    MODERN_CONFIG_370M,
    MODERN_CONFIG_1B,
    MODERN_CONFIG_3B,
    GPTModel,
    create_dataloader,
    generate_text_simple,
)

import tiktoken

# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------
CONFIGS = {
    "124m":  GPT_CONFIG_124M,
    "370m":  MODERN_CONFIG_370M,
    "1b":    MODERN_CONFIG_1B,
    "3b":    MODERN_CONFIG_3B,
}

BATCH_SIZE    = 4
MAX_LENGTH    = 512        # longer context than gpt_model2 (benefits from RoPE)
STRIDE        = 512        # non-overlapping windows
LEARNING_RATE = 3e-4
WEIGHT_DECAY  = 0.1
MAX_GRAD_NORM = 1.0
WARMUP_STEPS  = 200
TOTAL_STEPS   = 5000
EVAL_INTERVAL = 250
DATA_PATH     = "wikitext-103-raw/wiki.train.raw"
DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_lr(step: int, total_steps: int, warmup: int, base_lr: float) -> float:
    """Linear warmup → cosine decay."""
    if step < warmup:
        return base_lr * step / warmup
    progress = (step - warmup) / max(1, total_steps - warmup)
    return base_lr * 0.5 * (1.0 + math.cos(math.pi * progress))


def evaluate(model, dataloader, device, num_batches: int = 20) -> float:
    model.eval()
    total, count = 0.0, 0
    with torch.no_grad():
        for i, (x, y) in enumerate(dataloader):
            if i >= num_batches:
                break
            x, y = x.to(device), y.to(device)
            logits = model(x)
            total += F.cross_entropy(
                logits.view(-1, logits.size(-1)), y.view(-1)
            ).item()
            count += 1
    model.train()
    return total / max(count, 1)


def generate_sample(model, tokenizer, prompt: str, cfg: dict,
                    device: str, max_new: int = 80) -> str:
    tokens = tokenizer.encode(prompt)
    idx = torch.tensor([tokens], dtype=torch.long, device=device)
    model.eval()
    with torch.no_grad():
        out = generate_text_simple(
            model, idx,
            max_new_tokens=max_new,
            context_size=cfg["context_length"],
            temperature=0.8,
            top_k=40,
        )
    model.train()
    return tokenizer.decode(out[0].cpu().tolist())


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train modern LLaMA-style GPT")
    parser.add_argument("--config", default="370m", choices=list(CONFIGS),
                        help="Model size (default: 370m)")
    parser.add_argument("--steps",  type=int, default=TOTAL_STEPS,
                        help="Total training steps")
    parser.add_argument("--batch",  type=int, default=BATCH_SIZE)
    parser.add_argument("--lr",     type=float, default=LEARNING_RATE)
    parser.add_argument("--resume", default=None,
                        help="Path to a checkpoint to resume from")
    parser.add_argument("--data",   default=DATA_PATH)
    args = parser.parse_args()

    cfg       = CONFIGS[args.config]
    save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             f"gpt_modern_{args.config}_best.pth")

    print("\n" + "=" * 70)
    print(f"  Modern GPT training  |  config: {args.config.upper()}  |  device: {DEVICE}")
    print("=" * 70)

    # ── Data ─────────────────────────────────────────────────────────────────
    print(f"Loading {args.data} ...")
    with open(args.data, "r", encoding="utf-8") as f:
        text = f.read()
    print(f"  {len(text):,} chars")

    split = int(len(text) * 0.9)
    train_text, val_text = text[:split], text[split:]

    train_loader = create_dataloader(
        train_text, batch_size=args.batch, max_length=MAX_LENGTH,
        stride=STRIDE, shuffle=True, drop_last=True,
    )
    val_loader = create_dataloader(
        val_text, batch_size=args.batch, max_length=MAX_LENGTH,
        stride=STRIDE, shuffle=False, drop_last=True,
    )
    tokenizer = tiktoken.get_encoding("gpt2")
    print(f"  Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")

    # ── Model ────────────────────────────────────────────────────────────────
    torch.cuda.empty_cache()
    model = GPTModel(cfg).to(DEVICE)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {total_params / 1e6:.1f}M")

    # Mixed-precision (bf16 on Ampere+)
    use_amp = DEVICE == "cuda"
    scaler  = torch.amp.GradScaler("cuda", enabled=use_amp)

    # ── Optimizer ────────────────────────────────────────────────────────────
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=WEIGHT_DECAY,
        betas=(0.9, 0.95),
    )

    start_step   = 0
    best_val_loss = float("inf")

    # ── Resume ───────────────────────────────────────────────────────────────
    if args.resume and os.path.exists(args.resume):
        ckpt = torch.load(args.resume, map_location=DEVICE, weights_only=True)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_step    = ckpt.get("step", 0)
        best_val_loss = ckpt.get("val_loss", float("inf"))
        print(f"  Resumed from step {start_step}, best val_loss={best_val_loss:.4f}")

    # ── Training loop ────────────────────────────────────────────────────────
    model.train()
    step       = start_step
    train_iter = iter(train_loader)

    print(f"\n{'Step':>6}  {'Train Loss':>10}  {'Val Loss':>10}  "
          f"{'Perplexity':>11}  {'LR':>10}  {'tok/s':>8}")
    print("-" * 68)

    t0 = time.time()
    while step < args.steps:
        try:
            x, y = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            x, y = next(train_iter)

        x, y = x.to(DEVICE), y.to(DEVICE)

        lr = get_lr(step, args.steps, WARMUP_STEPS, args.lr)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        optimizer.zero_grad()

        with torch.amp.autocast(DEVICE, dtype=torch.bfloat16, enabled=use_amp):
            logits = model(x)
            loss   = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
        scaler.step(optimizer)
        scaler.update()

        step += 1

        if step % EVAL_INTERVAL == 0 or step == args.steps:
            elapsed = time.time() - t0
            tokens_per_sec = (EVAL_INTERVAL * args.batch * MAX_LENGTH) / elapsed

            val_loss   = evaluate(model, val_loader, DEVICE)
            train_loss = loss.item()
            ppl        = math.exp(min(val_loss, 20))  # cap to avoid overflow

            print(f"{step:>6}  {train_loss:>10.4f}  {val_loss:>10.4f}  "
                  f"{ppl:>11.1f}  {lr:>10.2e}  {tokens_per_sec:>8.0f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save({
                    "step":                step,
                    "model_state_dict":    model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss":            val_loss,
                    "config":              cfg,
                    "config_name":         args.config,
                }, save_path)
                print(f"         ✓ checkpoint saved  (val_loss={val_loss:.4f})")

            t0 = time.time()

    # ── Final generation sample ───────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  TEXT GENERATION SAMPLES")
    print("=" * 70)
    for prompt in [
        "The history of science shows that",
        "In the year 2025, technology has",
        "Once upon a time there was a",
    ]:
        sample = generate_sample(model, tokenizer, prompt, cfg, DEVICE)
        print(f"\n  Prompt: {prompt!r}")
        print(f"  Output: {sample!r}")

    print(f"\nTraining complete.  Best val_loss={best_val_loss:.4f}  "
          f"(perplexity={math.exp(min(best_val_loss,20)):.1f})")
    print(f"Checkpoint: {save_path}")


if __name__ == "__main__":
    main()
