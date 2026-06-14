# DHENCHOOO 🔥
**A from-scratch GPT-style code generation AI that trains itself on GitHub**

> No pretrained weights. No HuggingFace. No OpenAI API. Pure PyTorch + raw math.

---

## Architecture

```
User Prompt (English)
        ↓
  BPE Tokenizer (scratch)
        ↓
  DHENCHOOO Transformer
  ┌─────────────────────┐
  │  Token Embedding    │
  │  + Pos Embedding    │
  │         ↓           │
  │  6× TransformerBlock│
  │  ├─ RMSNorm         │
  │  ├─ CausalSelfAttn  │  ← masked, multi-head
  │  ├─ RMSNorm         │
  │  └─ SwiGLU FFN      │
  │         ↓           │
  │  RMSNorm + LM Head  │
  └─────────────────────┘
        ↓
  Next-token logits → sampling → code
```

**Training loop:**
```
GitHub Search API → repo list
    ↓
Fetch file contents via Contents API (no git clone needed!)
    ↓
Tokenize → mini-batches (block_size=512)
    ↓
Forward pass → cross-entropy loss → backward → AdamW update
    ↓
Mark repo as seen → fetch next repo → repeat ∞
    ↓ (on Ctrl+C)
Finish CURRENT repo completely → save checkpoint → exit
```

---

## Files

| File | Purpose |
|------|---------|
| `config.py` | All hyperparameters (edit here to scale up) |
| `tokenizer.py` | BPE tokenizer from scratch (no HuggingFace) |
| `model.py` | GPT Transformer, RMSNorm, SwiGLU, causal attention |
| `github_crawler.py` | GitHub REST API crawler |
| `trainer.py` | Training loop + graceful Ctrl+C shutdown |
| `generator.py` | Inference REPL + one-shot generation |
| `main.py` | CLI entry point |

---

## Quick Start

```bash
# 1. Install deps
pip install -r requirements.txt

# 2. (Optional but recommended) Set GitHub token for higher rate limit
export GITHUB_TOKEN=ghp_yourTokenHere

# 3. Bootstrap the tokenizer (one-time, ~30 seconds)
python main.py bootstrap

# 4. Start training! (runs until you Ctrl+C)
python main.py train

# 5. Generate code
python main.py generate
# or one-shot:
python main.py generate "write a binary search tree in python"
# specify language explicitly:
python main.py generate "@rust implement a stack data structure"
```

---

## Graceful Stop Behaviour

When you press **Ctrl+C** during training:
1. A stop flag is set immediately
2. Training on the **current file** completes normally
3. After the **current repo** (all files) is done, training stops
4. Checkpoint is saved automatically before exit

**You will NEVER lose a partially-trained repo.** The model always finishes what it started.

---

## Scaling Up

Edit `config.py` to make DHENCHOOO bigger:

```python
MODEL_CONFIG = {
    "n_embd":     512,   # was 256   (2× bigger)
    "n_head":     8,     # same
    "n_layer":    12,    # was 6     (2× deeper)
    "block_size": 1024,  # was 512   (2× context)
    "vocab_size": 16384, # was 8192
}
```

On a GPU you can comfortably run:
- `n_embd=512, n_layer=12` → ~85M params (GPT-2 small territory)
- `n_embd=768, n_layer=12` → ~117M params

On CPU / Termux with no GPU, keep defaults (256/6) — trains slowly but works.

---

## Tips for Termux / Android

```bash
# Install PyTorch for ARM (CPU only)
pip install torch --index-url https://download.pytorch.org/whl/cpu

# Run training in background
nohup python main.py train > dhenchooo_train.log 2>&1 &

# Watch the log
tail -f dhenchooo_train.log
```

---

## How prompting works

DHENCHOOO was trained on files wrapped like:
```
<|python|>
# File: solution.py
def binary_search(arr, target):
    ...
<|endoffile|>
```

So at inference, it expects:
```
<|python|>
# Task: write a binary search
# Implementation:
```
…and it continues from there. The `generator.py` handles this automatically.

---

## License
Do whatever you want. This is DHENCHOOO. 🔥
