"""
DHENCHOOO - Trainer
Continuous training loop:
  1. Stream code from GitHub via GitHubCrawler
  2. Tokenize on-the-fly
  3. Train DHENCHOOO model with gradient updates
  4. GRACEFUL STOP: Ctrl+C / stop-flag completes the current repo's
     training before exiting — never cuts mid-repo.
"""

import os
import time
import signal
import threading
import math
import torch
import torch.nn.functional as F
from typing import List, Optional

from config import MODEL_CONFIG, TRAIN_CONFIG, PATHS
from tokenizer import BPETokenizer
from model import DHENCHOOO, save_checkpoint, load_checkpoint
from github_crawler import GitHubCrawler


# ─────────────────────────────────────────────────────────────────────────────
#  Graceful shutdown flag
# ─────────────────────────────────────────────────────────────────────────────

class StopSignal:
    """Thread-safe flag. Set by SIGINT/SIGTERM. Checked between repos."""
    def __init__(self):
        self._stop = False
        self._lock = threading.Lock()

    def request_stop(self):
        with self._lock:
            self._stop = True
        print(
            "\n\n⚠️  [DHENCHOOO] Stop requested! "
            "Finishing current repo's training before exiting …\n"
        )

    @property
    def should_stop(self) -> bool:
        with self._lock:
            return self._stop


STOP = StopSignal()

# Register signal handlers so Ctrl+C triggers graceful stop
signal.signal(signal.SIGINT,  lambda s, f: STOP.request_stop())
signal.signal(signal.SIGTERM, lambda s, f: STOP.request_stop())


# ─────────────────────────────────────────────────────────────────────────────
#  Data helpers
# ─────────────────────────────────────────────────────────────────────────────

def text_to_batches(
    text: str,
    tokenizer: BPETokenizer,
    block_size: int,
    batch_size: int,
):
    """
    Tokenize text → split into overlapping (input, target) blocks → yield batches.
    Yields (x, y) tensors of shape (batch_size, block_size).
    """
    ids = tokenizer.encode(text, add_special=True)
    if len(ids) < block_size + 1:
        return   # too short to form even one sample

    # Build all possible (x, y) pairs with stride = block_size // 2
    stride  = max(block_size // 2, 1)
    samples: List[torch.Tensor] = []
    for start in range(0, len(ids) - block_size, stride):
        chunk = ids[start: start + block_size + 1]
        samples.append(torch.tensor(chunk, dtype=torch.long))

    if not samples:
        return

    # Shuffle samples so different parts of the file get trained in random order
    import random; random.shuffle(samples)

    # Yield mini-batches
    for i in range(0, len(samples), batch_size):
        batch = samples[i: i + batch_size]
        if len(batch) == 0:
            continue
        # Pad to same length if last batch is smaller
        padded = torch.zeros(len(batch), block_size + 1, dtype=torch.long)
        for j, s in enumerate(batch):
            padded[j, :len(s)] = s
        x = padded[:, :-1]   # input  (B, T)
        y = padded[:, 1:]    # target (B, T)
        yield x, y


# ─────────────────────────────────────────────────────────────────────────────
#  Cosine LR schedule with warmup
# ─────────────────────────────────────────────────────────────────────────────

def get_lr(step: int, lr: float, warmup: int, max_steps: int) -> float:
    if step < warmup:
        return lr * step / max(warmup, 1)
    ratio = (step - warmup) / max(max_steps - warmup, 1)
    return lr * 0.5 * (1.0 + math.cos(math.pi * ratio))


# ─────────────────────────────────────────────────────────────────────────────
#  Main trainer class
# ─────────────────────────────────────────────────────────────────────────────

class Trainer:
    def __init__(self, resume: bool = True):
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else
            "mps"  if torch.backends.mps.is_available() else
            "cpu"
        )
        print(f"[Trainer] Device: {self.device}")

        # Tokenizer ─────────────────────────────────────────────────────────
        self.tokenizer = BPETokenizer(vocab_size=MODEL_CONFIG["vocab_size"])
        if os.path.exists(PATHS["tokenizer"]):
            self.tokenizer.load(PATHS["tokenizer"])
        else:
            # Bootstrap tokenizer on a tiny seed so we can start training immediately.
            # It will grow as it sees more code.
            print("[Trainer] No tokenizer found — bootstrapping on seed text …")
            seed = (
                "def main(): pass\n"
                "for i in range(10): print(i)\n"
                "import os, sys\n"
                "const x = () => {}\n"
                "public class Main { public static void main(String[] args) {} }\n"
            ) * 50
            self.tokenizer.train([seed], verbose=False)
            self.tokenizer.save(PATHS["tokenizer"])

        # Model ─────────────────────────────────────────────────────────────
        if resume and os.path.exists(PATHS["checkpoint"]):
            print(f"[Trainer] Resuming from {PATHS['checkpoint']}")
            self.model, self.optimizer, self.global_step, last_loss = \
                load_checkpoint(PATHS["checkpoint"], self.device)
            print(f"  Resumed at step {self.global_step}, loss {last_loss:.4f}")
        else:
            print("[Trainer] Initialising new model from scratch …")
            # Use actual tokenizer vocab size (may differ from config if tokenizer was loaded)
            cfg = {**MODEL_CONFIG, "vocab_size": len(self.tokenizer)}
            self.model      = DHENCHOOO(**cfg).to(self.device)
            self.optimizer  = torch.optim.AdamW(
                self.model.parameters(),
                lr=TRAIN_CONFIG["learning_rate"],
                weight_decay=TRAIN_CONFIG["weight_decay"],
                betas=(0.9, 0.95),
            )
            self.global_step = 0

        n_params = self.model.num_params()
        print(f"[Trainer] Model params: {n_params/1e6:.2f}M")

        self.crawler     = GitHubCrawler()
        self.block_size  = MODEL_CONFIG["block_size"]
        self.batch_size  = TRAIN_CONFIG["batch_size"]

    # ─────────────────────────────────────────────────────────────────────────

    def _train_on_text(self, text: str, repo_name: str) -> int:
        """
        Train on a single file's text.
        Returns number of gradient steps taken.
        This function runs to COMPLETION regardless of STOP flag —
        that's intentional (we finish the file we're on).
        """
        self.model.train()
        steps        = 0
        total_loss   = 0.0
        max_steps    = TRAIN_CONFIG["max_steps_per_repo"]

        # Ensure tokenizer knows all chars in this text
        self.tokenizer.update([text])

        batches = list(text_to_batches(
            text, self.tokenizer, self.block_size, self.batch_size
        ))
        if not batches:
            return 0

        for x, y in batches:
            if steps >= max_steps:
                break

            x, y = x.to(self.device), y.to(self.device)

            # LR schedule
            lr = get_lr(
                self.global_step,
                TRAIN_CONFIG["learning_rate"],
                TRAIN_CONFIG["warmup_steps"],
                max_steps=10_000,   # soft max for cosine decay
            )
            for pg in self.optimizer.param_groups:
                pg["lr"] = lr

            # Forward + backward
            _, loss = self.model(x, targets=y)
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), TRAIN_CONFIG["grad_clip"]
            )
            self.optimizer.step()

            total_loss       += loss.item()
            steps            += 1
            self.global_step += 1

            if self.global_step % TRAIN_CONFIG["eval_interval"] == 0:
                avg = total_loss / max(steps, 1)
                print(
                    f"  step={self.global_step:6d}  "
                    f"loss={avg:.4f}  lr={lr:.2e}  "
                    f"repo={repo_name}"
                )
                total_loss = 0.0

            if self.global_step % TRAIN_CONFIG["save_interval"] == 0:
                save_checkpoint(
                    self.model, self.optimizer,
                    self.global_step, loss.item(),
                    PATHS["checkpoint"],
                )
                self.tokenizer.save(PATHS["tokenizer"])
                print(f"  [Checkpoint] Saved at step {self.global_step}")

        return steps

    # ─────────────────────────────────────────────────────────────────────────

    def run(self) -> None:
        """
        Main training loop.
        - Streams repos from GitHub
        - Trains on each file
        - After a FULL REPO is processed, checks the stop flag
        - Only stops between repos, never mid-repo
        """
        print("\n" + "═" * 60)
        print("  DHENCHOOO — Training Loop Started")
        print("  Press Ctrl+C to stop (finishes current repo first)")
        print("═" * 60 + "\n")

        current_repo      = None
        repo_texts        = []   # accumulate all files of one repo

        try:
            for repo_name, lang, text in self.crawler.stream_code():

                # ── New repo started ────────────────────────────────────────
                if repo_name != current_repo:
                    # If we have accumulated texts from the previous repo → train
                    if current_repo is not None and repo_texts:
                        print(f"\n[Trainer] Training on repo: {current_repo} ({len(repo_texts)} files)")
                        for t in repo_texts:
                            self._train_on_text(t, current_repo)
                        # ── GRACEFUL STOP CHECK — only HERE, between repos ──
                        if STOP.should_stop:
                            print("\n✅ [DHENCHOOO] Current repo training complete. Stopping now.")
                            break

                    current_repo = repo_name
                    repo_texts   = []
                    print(f"[Crawler] New repo: {repo_name} (lang={lang})")

                repo_texts.append(text)

            else:
                # Iterator exhausted normally
                # Train the last repo
                if current_repo and repo_texts:
                    print(f"\n[Trainer] Training on final repo: {current_repo}")
                    for t in repo_texts:
                        self._train_on_text(t, current_repo)

        finally:
            # Always save on exit
            print(f"\n[Trainer] Saving final checkpoint at step {self.global_step} …")
            save_checkpoint(
                self.model, self.optimizer,
                self.global_step, 0.0,
                PATHS["checkpoint"],
            )
            self.tokenizer.save(PATHS["tokenizer"])
            print("✅ [DHENCHOOO] Saved. Goodbye!")
