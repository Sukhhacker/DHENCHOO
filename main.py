"""
DHENCHOOO — Main CLI
Usage:
  python main.py train              # Start/resume training from GitHub
  python main.py generate           # Interactive code generation REPL
  python main.py generate "prompt"  # One-shot code generation
  python main.py info               # Show model/tokenizer info
  python main.py bootstrap          # Train tokenizer on seed code (first-time setup)

Environment variables:
  GITHUB_TOKEN=ghp_xxx              # Optional: raises rate limit from 60 to 5000 req/hr
"""

import sys
import os


def cmd_train(args):
    """Start or resume continuous GitHub training loop."""
    from trainer import Trainer
    resume = "--fresh" not in args
    t = Trainer(resume=resume)
    t.run()


def cmd_generate(args):
    """Generate code from a prompt."""
    from generator import CodeGenerator
    gen = CodeGenerator()
    if args:
        # One-shot: argument is the prompt
        prompt = " ".join(args)
        lang   = None
        if prompt.startswith("@"):
            parts = prompt[1:].split(" ", 1)
            if len(parts) == 2:
                lang, prompt = parts[0], parts[1]
        print(f"\n🤖 DHENCHOOO:\n")
        print(gen.generate(prompt, language=lang))
    else:
        gen.interactive()


def cmd_info(args):
    """Print model and tokenizer info."""
    import torch
    from config import PATHS, MODEL_CONFIG
    from tokenizer import BPETokenizer
    from model import load_checkpoint

    print("\n" + "═" * 50)
    print("  DHENCHOOO — Model Info")
    print("═" * 50)

    tok_path = PATHS["tokenizer"]
    if os.path.exists(tok_path):
        tok = BPETokenizer()
        tok.load(tok_path)
        print(f"  Tokenizer vocab size : {len(tok)}")
        print(f"  Tokenizer merges     : {len(tok.merges)}")
    else:
        print(f"  Tokenizer            : NOT FOUND ({tok_path})")

    ckpt_path = PATHS["checkpoint"]
    if os.path.exists(ckpt_path):
        device = torch.device("cpu")
        model, _, step, loss = load_checkpoint(ckpt_path, device)
        n = model.num_params()
        print(f"  Model checkpoint     : {ckpt_path}")
        print(f"  Training steps       : {step}")
        print(f"  Last loss            : {loss:.4f}")
        print(f"  Parameters           : {n/1e6:.2f}M")
    else:
        print(f"  Checkpoint           : NOT FOUND ({ckpt_path})")

    crawl_log = PATHS["crawl_log"]
    if os.path.exists(crawl_log):
        with open(crawl_log) as f:
            lines = [l.strip() for l in f if l.strip()]
        print(f"  Repos trained on     : {len(lines)}")
    else:
        print(f"  Repos trained on     : 0")

    print("═" * 50 + "\n")


def cmd_bootstrap(args):
    """Bootstrap the BPE tokenizer on a small seed corpus without GitHub."""
    from tokenizer import BPETokenizer
    from config import PATHS, MODEL_CONFIG

    seed_code = """\
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

class Node:
    def __init__(self, value):
        self.value = value
        self.next = None

def binary_search(arr, target):
    left, right = 0, len(arr) - 1
    while left <= right:
        mid = (left + right) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    return -1

const express = require('express');
const app = express();
app.get('/', (req, res) => res.send('Hello World'));
app.listen(3000);

func main() {
    fmt.Println("Hello, World!")
    for i := 0; i < 10; i++ {
        fmt.Println(i)
    }
}

public class Solution {
    public int maxProfit(int[] prices) {
        int minPrice = Integer.MAX_VALUE;
        int maxProfit = 0;
        for (int price : prices) {
            if (price < minPrice) minPrice = price;
            else if (price - minPrice > maxProfit) maxProfit = price - minPrice;
        }
        return maxProfit;
    }
}

fn bubble_sort(arr: &mut Vec<i32>) {
    let n = arr.len();
    for i in 0..n {
        for j in 0..n-i-1 {
            if arr[j] > arr[j+1] {
                arr.swap(j, j+1);
            }
        }
    }
}
""" * 30   # repeat to give more training signal

    print("[Bootstrap] Training tokenizer on seed code …")
    tok = BPETokenizer(vocab_size=MODEL_CONFIG["vocab_size"])
    tok.train([seed_code], verbose=True)
    tok.save(PATHS["tokenizer"])
    print(f"[Bootstrap] Done! Tokenizer saved to {PATHS['tokenizer']}")
    print("Now run: python main.py train")


COMMANDS = {
    "train":     cmd_train,
    "generate":  cmd_generate,
    "gen":       cmd_generate,
    "info":      cmd_info,
    "bootstrap": cmd_bootstrap,
}


def main():
    print(r"""
  ██████╗ ██╗  ██╗███████╗███╗   ██╗ ██████╗██╗  ██╗ ██████╗  ██████╗  ██████╗
  ██╔══██╗██║  ██║██╔════╝████╗  ██║██╔════╝██║  ██║██╔═══██╗██╔═══██╗██╔═══██╗
  ██║  ██║███████║█████╗  ██╔██╗ ██║██║     ███████║██║   ██║██║   ██║██║   ██║
  ██║  ██║██╔══██║██╔══╝  ██║╚██╗██║██║     ██╔══██║██║   ██║██║   ██║██║   ██║
  ██████╔╝██║  ██║███████╗██║ ╚████║╚██████╗██║  ██║╚██████╔╝╚██████╔╝╚██████╔╝
  ╚═════╝ ╚═╝  ╚═╝╚══════╝╚═╝  ╚═══╝ ╚═════╝╚═╝  ╚═╝ ╚═════╝  ╚═════╝  ╚═════╝
  Code Generation AI — Trained from scratch on GitHub public repos
""")

    if len(sys.argv) < 2:
        print("Usage:")
        for cmd in COMMANDS:
            print(f"  python main.py {cmd}")
        print("\nFirst-time setup:")
        print("  1. python main.py bootstrap   # build tokenizer")
        print("  2. python main.py train        # start training from GitHub")
        print("  3. python main.py generate     # start generating code")
        sys.exit(0)

    cmd  = sys.argv[1].lower()
    rest = sys.argv[2:]

    if cmd not in COMMANDS:
        print(f"Unknown command '{cmd}'. Available: {', '.join(COMMANDS)}")
        sys.exit(1)

    COMMANDS[cmd](rest)


if __name__ == "__main__":
    main()
