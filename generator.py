"""
DHENCHOOO - Code Generator
Takes a user's natural-language prompt → produces code.
Works completely offline after training (no API calls needed for inference).
"""

import os
import torch
from typing import Optional
from config import MODEL_CONFIG, PATHS
from tokenizer import BPETokenizer
from model import DHENCHOOO, load_checkpoint

# Language name → file extension hints (for prompt formatting)
LANG_HINTS = {
    "python": "python", "py": "python",
    "javascript": "javascript", "js": "javascript",
    "typescript": "typescript", "ts": "typescript",
    "go": "go", "golang": "go",
    "rust": "rust", "rs": "rust",
    "java": "java",
    "c": "c", "c++": "cpp", "cpp": "cpp",
    "bash": "bash", "shell": "bash", "sh": "bash",
    "ruby": "ruby", "rb": "ruby",
    "php": "php",
    "csharp": "csharp", "c#": "csharp",
    "kotlin": "kotlin", "kt": "kotlin",
    "swift": "swift",
}


def _detect_language(prompt: str) -> str:
    """Try to guess the desired language from the prompt text."""
    lower = prompt.lower()
    for kw, lang in LANG_HINTS.items():
        if kw in lower:
            return lang
    return "python"   # default


def _build_prompt(user_request: str, language: str) -> str:
    """
    Construct the input prompt that DHENCHOOO was trained to complete.
    Format mirrors the training-time wrapping in the crawler.
    """
    return (
        f"<|{language}|>\n"
        f"# Task: {user_request}\n"
        f"# Implementation:\n"
    )


class CodeGenerator:
    """
    High-level inference wrapper around DHENCHOOO.
    Load once, call .generate() many times.
    """

    def __init__(self, checkpoint: Optional[str] = None, tokenizer_path: Optional[str] = None):
        checkpoint     = checkpoint     or PATHS["checkpoint"]
        tokenizer_path = tokenizer_path or PATHS["tokenizer"]

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else
            "mps"  if torch.backends.mps.is_available() else
            "cpu"
        )

        # Load tokenizer
        self.tokenizer = BPETokenizer()
        if os.path.exists(tokenizer_path):
            self.tokenizer.load(tokenizer_path)
        else:
            raise FileNotFoundError(
                f"Tokenizer not found at '{tokenizer_path}'.\n"
                "Run training first: python main.py train"
            )

        # Load model
        if os.path.exists(checkpoint):
            self.model, _, step, loss = load_checkpoint(checkpoint, self.device)
            print(f"[Generator] Model loaded (step={step}, last_loss={loss:.4f})")
        else:
            raise FileNotFoundError(
                f"Checkpoint not found at '{checkpoint}'.\n"
                "Run training first: python main.py train"
            )

        self.model.eval()

    def generate(
        self,
        prompt:       str,
        language:     Optional[str] = None,
        max_tokens:   int   = 300,
        temperature:  float = 0.7,
        top_k:        int   = 40,
        top_p:        float = 0.92,
    ) -> str:
        """
        Generate code from a natural-language user prompt.

        Args:
            prompt:      What you want the model to write (English description).
            language:    Target language ('python', 'go', etc.). Auto-detected if None.
            max_tokens:  Max new tokens to generate.
            temperature: Randomness (lower = more deterministic).
            top_k:       Top-k sampling.
            top_p:       Nucleus sampling probability.

        Returns:
            Generated code as a string.
        """
        lang = LANG_HINTS.get((language or "").lower()) or _detect_language(prompt)
        full_prompt = _build_prompt(prompt, lang)

        # Encode prompt
        ids = self.tokenizer.encode(full_prompt, add_special=False)
        if not ids:
            return "# [DHENCHOOO] Could not encode prompt — try training more first."

        idx = torch.tensor([ids], dtype=torch.long, device=self.device)

        # Get EOS id for early stopping
        eos_id = self.tokenizer.token2id.get("<|eos|>", -1)

        with torch.no_grad():
            out = self.model.generate(
                idx,
                max_new     = max_tokens,
                temperature = temperature,
                top_k       = top_k,
                top_p       = top_p,
                stop_token  = eos_id,
            )

        # Decode only the newly generated tokens
        new_ids  = out[0, len(ids):].tolist()
        raw      = self.tokenizer.decode(new_ids)

        # Strip trailing special tokens
        for tag in ["<|eos|>", "<|endoffile|>", "<|bos|>"]:
            raw = raw.replace(tag, "")

        return raw.strip()

    def interactive(self) -> None:
        """
        Simple REPL for interactive code generation.
        Type your request in English, get code back.
        """
        print("\n" + "═" * 60)
        print("  DHENCHOOO — Code Generator")
        print("  Type your request in English. Type 'exit' to quit.")
        print("  Tip: mention the language — 'write a python ...'")
        print("═" * 60)

        while True:
            try:
                user_input = input("\n🔥 You: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nBye!")
                break

            if not user_input:
                continue
            if user_input.lower() in ("exit", "quit", "q"):
                print("Bye!")
                break

            # Optional: let user specify explicit language
            lang = None
            if user_input.startswith("@"):
                # e.g. "@rust write a fibonacci function"
                parts = user_input[1:].split(" ", 1)
                if len(parts) == 2:
                    lang, user_input = parts[0], parts[1]
                else:
                    user_input = parts[0]

            print(f"\n🤖 DHENCHOOO [{lang or 'auto'}]:\n")
            code = self.generate(user_input, language=lang)
            print(code)
            print()
