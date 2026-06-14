"""
DHENCHOOO - GPT-style Transformer (100% scratch, no pretrained weights)
Architecture: Decoder-only Transformer (like GPT-2 but minimal)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


# ─────────────────────────────────────────────────────────────────────────────
#  Building blocks
# ─────────────────────────────────────────────────────────────────────────────

class RMSNorm(nn.Module):
    """Root Mean Square Layer Norm — faster than LayerNorm, used in LLaMA style."""
    def __init__(self, dim: int, eps: float = 1e-8):
        super().__init__()
        self.eps   = eps
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms  = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).sqrt()
        return self.scale * (x / rms)


class CausalSelfAttention(nn.Module):
    """
    Multi-head causal (decoder) self-attention.
    - No cross-attention, purely self.
    - Uses a causal mask so token i can only attend to tokens 0..i
    """
    def __init__(self, n_embd: int, n_head: int, block_size: int, dropout: float):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_head  = n_head
        self.n_embd  = n_embd
        self.head_dim = n_embd // n_head

        # Q, K, V projections fused into one linear for speed
        self.qkv_proj = nn.Linear(n_embd, 3 * n_embd, bias=False)
        self.out_proj  = nn.Linear(n_embd, n_embd, bias=False)
        self.attn_drop = nn.Dropout(dropout)
        self.resid_drop = nn.Dropout(dropout)

        # Register causal mask as buffer (not a parameter)
        mask = torch.tril(torch.ones(block_size, block_size)).view(
            1, 1, block_size, block_size
        )
        self.register_buffer("mask", mask)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        # Compute Q, K, V all at once
        qkv = self.qkv_proj(x)                               # (B, T, 3C)
        q, k, v = qkv.split(self.n_embd, dim=-1)             # each (B, T, C)

        # Reshape to (B, n_head, T, head_dim)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention with causal mask
        scale = 1.0 / math.sqrt(self.head_dim)
        attn  = (q @ k.transpose(-2, -1)) * scale            # (B, H, T, T)
        attn  = attn.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
        attn  = F.softmax(attn, dim=-1)
        attn  = self.attn_drop(attn)

        y = (attn @ v).transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_drop(self.out_proj(y))


class FeedForward(nn.Module):
    """SwiGLU feed-forward network (used in modern LLMs)."""
    def __init__(self, n_embd: int, dropout: float):
        super().__init__()
        hidden = 4 * n_embd
        # Gate + value projections
        self.gate_proj  = nn.Linear(n_embd, hidden, bias=False)
        self.value_proj = nn.Linear(n_embd, hidden, bias=False)
        self.out_proj   = nn.Linear(hidden, n_embd, bias=False)
        self.drop       = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = F.silu(self.gate_proj(x))   # SiLU (Swish) gate
        val  = self.value_proj(x)
        return self.drop(self.out_proj(gate * val))


class TransformerBlock(nn.Module):
    """One decoder block: Attention → FFN with residual connections."""
    def __init__(self, n_embd: int, n_head: int, block_size: int, dropout: float):
        super().__init__()
        self.norm1 = RMSNorm(n_embd)
        self.attn  = CausalSelfAttention(n_embd, n_head, block_size, dropout)
        self.norm2 = RMSNorm(n_embd)
        self.ffn   = FeedForward(n_embd, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


# ─────────────────────────────────────────────────────────────────────────────
#  DHENCHOOO Core Model
# ─────────────────────────────────────────────────────────────────────────────

class DHENCHOOO(nn.Module):
    """
    GPT-style code generation model.
    Built completely from scratch — no pretrained weights, no HuggingFace.

    Input:  token IDs  (B, T)
    Output: logits     (B, T, vocab_size)
    """

    def __init__(
        self,
        vocab_size: int,
        n_embd:     int,
        n_head:     int,
        n_layer:    int,
        block_size: int,
        dropout:    float = 0.1,
    ):
        super().__init__()
        self.block_size = block_size

        # Token + positional embeddings
        self.tok_emb = nn.Embedding(vocab_size, n_embd)
        self.pos_emb = nn.Embedding(block_size, n_embd)
        self.drop    = nn.Dropout(dropout)

        # Stack of transformer blocks
        self.blocks  = nn.ModuleList([
            TransformerBlock(n_embd, n_head, block_size, dropout)
            for _ in range(n_layer)
        ])

        # Final norm + language model head
        self.norm    = RMSNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)

        # Weight tying: share token embedding weights with lm_head
        # (classic GPT trick — reduces parameters, improves generalisation)
        self.lm_head.weight = self.tok_emb.weight

        # Init weights
        self.apply(self._init_weights)
        # Scale residual projections by 1/sqrt(2*n_layer) (GPT-2 paper)
        for name, p in self.named_parameters():
            if name.endswith(("out_proj.weight", "value_proj.weight")):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * n_layer))

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        idx:     torch.Tensor,                  # (B, T) token ids
        targets: Optional[torch.Tensor] = None, # (B, T) next-token targets
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:

        B, T = idx.shape
        assert T <= self.block_size, f"Sequence length {T} > block_size {self.block_size}"

        device = idx.device
        pos    = torch.arange(T, device=device).unsqueeze(0)  # (1, T)

        # Embed tokens + positions
        x = self.drop(self.tok_emb(idx) + self.pos_emb(pos))

        # Pass through transformer blocks
        for block in self.blocks:
            x = block(x)

        x      = self.norm(x)
        logits = self.lm_head(x)   # (B, T, vocab_size)

        loss = None
        if targets is not None:
            # Cross-entropy over all positions
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1,
            )

        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        idx:          torch.Tensor,   # (1, T) seed token ids
        max_new:      int   = 256,
        temperature:  float = 0.8,
        top_k:        int   = 50,
        top_p:        float = 0.95,   # nucleus sampling
        stop_token:   int   = -1,     # EOS token id; -1 to disable
    ) -> torch.Tensor:
        """
        Autoregressively generate tokens.
        Uses top-k + nucleus (top-p) sampling.
        """
        self.eval()
        for _ in range(max_new):
            # Crop context to block_size
            ctx      = idx[:, -self.block_size:]
            logits, _ = self(ctx)
            logits   = logits[:, -1, :] / max(temperature, 1e-9)  # (1, vocab)

            # Top-k filtering
            if top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, -1:]] = float("-inf")

            # Top-p (nucleus) filtering
            if top_p < 1.0:
                sorted_logits, sorted_idx = torch.sort(logits, descending=True)
                cumprobs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                remove   = cumprobs - F.softmax(sorted_logits, dim=-1) > top_p
                remove[:, 0] = False   # always keep top token
                sorted_logits[remove] = float("-inf")
                logits = sorted_logits.scatter(1, sorted_idx, sorted_logits)

            probs    = F.softmax(logits, dim=-1)
            next_tok = torch.multinomial(probs, num_samples=1)  # (1, 1)
            idx      = torch.cat([idx, next_tok], dim=1)

            if stop_token >= 0 and next_tok.item() == stop_token:
                break

        return idx

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


# ─────────────────────────────────────────────────────────────────────────────
#  Checkpoint helpers
# ─────────────────────────────────────────────────────────────────────────────

def save_checkpoint(model: DHENCHOOO, optimizer: torch.optim.Optimizer,
                    step: int, loss: float, path: str) -> None:
    torch.save({
        "step":        step,
        "loss":        loss,
        "model_state": model.state_dict(),
        "optim_state": optimizer.state_dict(),
        "config": {
            "vocab_size": model.tok_emb.num_embeddings,
            "n_embd":     model.tok_emb.embedding_dim,
            "n_head":     model.blocks[0].attn.n_head,
            "n_layer":    len(model.blocks),
            "block_size": model.block_size,
        },
    }, path)


def load_checkpoint(path: str, device: torch.device):
    """Returns (model, optimizer, step, loss) from a saved checkpoint."""
    ckpt = torch.load(path, map_location=device)
    cfg  = ckpt["config"]
    model = DHENCHOOO(**cfg).to(device)
    model.load_state_dict(ckpt["model_state"])
    optimizer = torch.optim.AdamW(model.parameters())
    optimizer.load_state_dict(ckpt["optim_state"])
    return model, optimizer, ckpt["step"], ckpt["loss"]
