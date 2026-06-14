"""
DHENCHOOO - BPE Tokenizer (built from scratch, no HuggingFace)
Byte-Pair Encoding implementation using only Python stdlib + json.
"""

import json
import os
import re
from collections import Counter, defaultdict
from typing import Dict, List, Tuple, Optional

# Special tokens
PAD_TOKEN  = "<|pad|>"
UNK_TOKEN  = "<|unk|>"
BOS_TOKEN  = "<|bos|>"
EOS_TOKEN  = "<|eos|>"
LANG_TOKEN = "<|{lang}|>"   # filled at runtime e.g. <|python|>

SPECIAL_TOKENS = [PAD_TOKEN, UNK_TOKEN, BOS_TOKEN, EOS_TOKEN]


class BPETokenizer:
    """
    Minimal Byte-Pair Encoding tokenizer.
    - Trains on raw text corpora (build_vocab / train_bpe)
    - Encodes / decodes token-id sequences
    - Saves & loads to JSON
    """

    def __init__(self, vocab_size: int = 8192):
        self.vocab_size   = vocab_size
        self.token2id: Dict[str, int] = {}
        self.id2token: Dict[int, str] = {}
        self.merges: List[Tuple[str, str]] = []   # ordered BPE merge rules
        self._built = False

    # ── Internal helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _get_stats(vocab: Dict[str, int]) -> Counter:
        """Count pair frequencies across all words."""
        pairs: Counter = Counter()
        for word, freq in vocab.items():
            symbols = word.split()
            for i in range(len(symbols) - 1):
                pairs[(symbols[i], symbols[i + 1])] += freq
        return pairs

    @staticmethod
    def _merge_vocab(pair: Tuple[str, str], vocab: Dict[str, int]) -> Dict[str, int]:
        """Merge the most frequent pair everywhere in the vocabulary."""
        bigram  = re.escape(" ".join(pair))
        pattern = re.compile(r"(?<!\S)" + bigram + r"(?!\S)")
        new_vocab: Dict[str, int] = {}
        merged = "".join(pair)
        for word in vocab:
            new_word = pattern.sub(merged, word)
            new_vocab[new_word] = vocab[word]
        return new_vocab

    # ── Public API ─────────────────────────────────────────────────────────────

    def train(self, texts: List[str], verbose: bool = True) -> None:
        """
        Build BPE vocab from a list of text strings.
        Call this once on a seed corpus; afterwards call update() for incremental.
        """
        # Start with character-level representation
        # Every char becomes a symbol; words split by space
        raw_vocab: Dict[str, int] = Counter()
        for text in texts:
            for word in text.split():
                # represent each character with a space separator + </w> at end
                key = " ".join(list(word)) + " </w>"
                raw_vocab[key] += 1

        # Seed token2id with all byte characters + special tokens
        base_chars: set = set()
        for word in raw_vocab:
            base_chars.update(word.split())

        self.token2id = {}
        for st in SPECIAL_TOKENS:
            self.token2id[st] = len(self.token2id)
        for ch in sorted(base_chars):
            if ch not in self.token2id:
                self.token2id[ch] = len(self.token2id)

        num_merges = self.vocab_size - len(self.token2id)
        if num_merges <= 0:
            num_merges = 0

        vocab = dict(raw_vocab)
        self.merges = []
        for i in range(num_merges):
            pairs = self._get_stats(vocab)
            if not pairs:
                break
            best = max(pairs, key=pairs.get)
            vocab = self._merge_vocab(best, vocab)
            merged_token = "".join(best)
            self.merges.append(best)
            if merged_token not in self.token2id:
                self.token2id[merged_token] = len(self.token2id)
            if verbose and i % 500 == 0:
                print(f"  [Tokenizer] merge {i}/{num_merges}  vocab={len(self.token2id)}")

        self.id2token = {v: k for k, v in self.token2id.items()}
        self._built = True
        print(f"  [Tokenizer] Training done. Vocab size = {len(self.token2id)}")

    def update(self, texts: List[str]) -> None:
        """
        Add new characters/tokens seen in texts to vocab (no new BPE merges,
        but ensures we don't get <unk> for new bytes).
        """
        if not self._built:
            raise RuntimeError("Call train() first.")
        for text in texts:
            for ch in text:
                if ch not in self.token2id:
                    idx = len(self.token2id)
                    self.token2id[ch] = idx
                    self.id2token[idx] = ch

    def _apply_merges(self, word: str) -> List[str]:
        """Apply learned merge rules to a single word (string of chars)."""
        symbols = list(word) + ["</w>"]
        for (a, b) in self.merges:
            merged = a + b
            i = 0
            while i < len(symbols) - 1:
                if symbols[i] == a and symbols[i + 1] == b:
                    symbols = symbols[:i] + [merged] + symbols[i + 2:]
                else:
                    i += 1
        return symbols

    def encode(self, text: str, add_special: bool = True) -> List[int]:
        """Text → list of token IDs."""
        if not self._built:
            raise RuntimeError("Tokenizer not trained yet.")
        ids: List[int] = []
        if add_special:
            ids.append(self.token2id.get(BOS_TOKEN, 0))
        unk_id = self.token2id.get(UNK_TOKEN, 1)
        for word in text.split():
            tokens = self._apply_merges(word)
            for tok in tokens:
                ids.append(self.token2id.get(tok, unk_id))
            ids.append(self.token2id.get(" ", unk_id))   # space between words
        if add_special:
            ids.append(self.token2id.get(EOS_TOKEN, 0))
        return ids

    def decode(self, ids: List[int]) -> str:
        """List of token IDs → text string."""
        tokens = [self.id2token.get(i, UNK_TOKEN) for i in ids]
        text = "".join(tokens)
        text = text.replace("</w>", " ").replace(BOS_TOKEN, "").replace(EOS_TOKEN, "")
        return text

    def lang_token_id(self, lang: str) -> int:
        """Get or create an ID for a language tag like <|python|>."""
        tag = f"<|{lang}|>"
        if tag not in self.token2id:
            idx = len(self.token2id)
            self.token2id[tag] = idx
            self.id2token[idx] = tag
        return self.token2id[tag]

    # ── Persistence ────────────────────────────────────────────────────────────

    def save(self, path: str) -> None:
        data = {
            "vocab_size": self.vocab_size,
            "token2id":   self.token2id,
            "merges":     self.merges,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"  [Tokenizer] Saved → {path}")

    def load(self, path: str) -> None:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.vocab_size = data["vocab_size"]
        self.token2id   = data["token2id"]
        self.merges     = [tuple(m) for m in data["merges"]]
        self.id2token   = {int(v): k for k, v in self.token2id.items()}
        self._built     = True
        print(f"  [Tokenizer] Loaded from {path}  vocab={len(self.token2id)}")

    def __len__(self) -> int:
        return len(self.token2id)
