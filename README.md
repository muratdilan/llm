# Modern LLaMA-Style GPT

A from-scratch decoder-only transformer trained on WikiText-103, implementing the full set of architectural upgrades found in LLaMA / Mistral over the original GPT-2.
## About This Project

This project was built to understand modern LLM architecture from first principles — not to use a library, but to implement every component from scratch and understand the design decisions that separate LLaMA/Mistral from the original GPT-2. RoPE, RMSNorm, SwiGLU, GQA, and Flash Attention are not just imported modules here — each is hand-coded and validated with a dedicated test suite covering architecture correctness, numerical stability, gradient flow, and causal mask integrity.

## Architecture

| Component | GPT-2 | This Project |
|---|---|---|
| Position encoding | Learned embeddings | **RoPE** (Rotary Position Embedding) |
| Normalisation | LayerNorm | **RMSNorm** |
| Feed-forward | GELU-FFN | **SwiGLU** |
| Attention | Standard MHA | **Grouped Query Attention (GQA)** |
| Attention kernel | Manual softmax | **Flash Attention (SDPA)** |

Tokenizer: GPT-2 BPE via `tiktoken` (`vocab_size = 50257`).

## Model Configurations

| Name | Params | Layers | `emb_dim` | `n_heads` | `n_kv_heads` | Context |
|---|---|---|---|---|---|---|
| `124m` | ~124M | 12 | 768 | 12 | 12 | 256 |
| `370m` | ~370M | 28 | 1024 | 16 | 4 | 4096 |
| `1b` | ~1B | 24 | 2048 | 16 | 4 | 4096 |
| `3b` | ~3B | 28 | 3072 | 24 | 8 | 4096 |

The `370m` config is the default. All configs are designed to fit on an RTX 5090 (32 GB VRAM) with mixed precision (bf16).

## File Structure

```
gpt_model.py              # Model definition (RoPE, RMSNorm, SwiGLU, GQA, Flash Attention)
train_modern.py           # Training script (AMP, cosine LR schedule, checkpointing)
test_gpt_modern.py        # Test suite (10 tests covering architecture and generation)
gpt_modern_370m_best.pth  # Best checkpoint saved during training
```

## Requirements

```
torch >= 2.0
tiktoken
```

Install:
```bash
pip install torch tiktoken
```

## Training

Download WikiText-103:
```bash
wget https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-103-raw-v1.zip
unzip wikitext-103-raw-v1.zip
```

Train the default 370M model:
```bash
python train_modern.py
```

Options:
```bash
python train_modern.py --config 1b          # 1B variant
python train_modern.py --steps 10000        # custom step count
python train_modern.py --resume gpt_modern_370m_best.pth  # resume from checkpoint
python train_modern.py --data /path/to/corpus.txt         # custom dataset
```

Training output:
```
  Step   Train Loss   Val Loss  Perplexity          LR    tok/s
----------------------------------------------------------------------
   250       5.2341      5.1890       179.5    2.98e-04     8432
   500       4.8712      4.9203       137.2    2.95e-04     8651
   ...
```

Checkpoints are saved to `gpt_modern_{config}_best.pth` whenever validation loss improves.

## Testing

```bash
python test_gpt_modern.py
```

Runs 10 tests:
1. Architecture basics (parameter count, output shape)
2. RMSNorm stability in float32 and bfloat16
3. Input validation (emb_dim/head_dim, seq_len overflow)
4. RoPE encoding (shape, dtype, position sensitivity)
5. GQA correctness (KV head count differs from Q head count)
6. Flash Attention / causal mask (no future token leakage)
7. Mixed-precision bf16 forward (no NaN, no warnings)
8. Dataloader (shape, stride, shifted targets)
9. Gradient flow (no dead gradients, reasonable norm)
10. Text generation (greedy, temperature sampling, top-k)

## Inference

```python
import torch
import tiktoken
from gpt_model import GPTModel, MODERN_CONFIG_370M, generate_text_simple

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load checkpoint
ckpt = torch.load("gpt_modern_370m_best.pth", map_location=device, weights_only=True)
model = GPTModel(ckpt["config"]).to(device)
model.load_state_dict(ckpt["model_state_dict"])
model.eval()

# Generate
tokenizer = tiktoken.get_encoding("gpt2")
prompt = tokenizer.encode("The history of science shows that")
idx = torch.tensor([prompt], device=device)

out = generate_text_simple(
    model, idx,
    max_new_tokens=100,
    context_size=MODERN_CONFIG_370M["context_length"],
    temperature=0.8,
    top_k=40,
)
print(tokenizer.decode(out[0].tolist()))
```

## Training Details

| Hyperparameter | Value |
|---|---|
| Optimizer | AdamW (β₁=0.9, β₂=0.95) |
| Learning rate | 3e-4 with linear warmup + cosine decay |
| Warmup steps | 200 |
| Weight decay | 0.1 |
| Gradient clip | 1.0 |
| Batch size | 4 |
| Sequence length | 512 |
| Precision | bf16 (Ampere+ GPU) |
