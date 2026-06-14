"""
DHENCHOOO - Configuration
All hyperparameters and settings in one place.
"""

# ─── Model Architecture ────────────────────────────────────────────────────────
MODEL_CONFIG = {
    "vocab_size":    8192,    # BPE vocabulary size
    "n_embd":        256,     # Embedding dimension  (scale up as you get more GPU)
    "n_head":        8,       # Attention heads
    "n_layer":       6,       # Transformer blocks
    "block_size":    512,     # Context window (tokens)
    "dropout":       0.1,
}

# ─── Training ─────────────────────────────────────────────────────────────────
TRAIN_CONFIG = {
    "batch_size":        4,
    "learning_rate":     3e-4,
    "weight_decay":      0.1,
    "grad_clip":         1.0,
    "eval_interval":     500,   # steps between loss prints
    "save_interval":     1000,  # steps between checkpoint saves
    "warmup_steps":      100,
    "max_steps_per_repo": 300,  # max gradient steps on one repo before moving on
    "min_file_chars":    200,   # ignore files shorter than this
    "max_file_chars":    50_000,# ignore files larger than this (avoid huge generated files)
}

# ─── GitHub Crawler ────────────────────────────────────────────────────────────
GITHUB_CONFIG = {
    # Put your GitHub PAT here for 5000 req/hr instead of 60 req/hr (optional)
    "token": "",   # export GITHUB_TOKEN=ghp_xxx  OR put it here
    "languages": ["Python", "JavaScript", "TypeScript", "Go", "Rust", "C", "C++", "Java"],
    "min_stars":  10,          # only repos with at least this many stars
    "max_repos_per_session": 9999,  # how many repos to crawl in one run (set low for testing)
    "files_per_repo":  30,     # max files to pull from a single repo
    "extensions": {            # file extensions to train on, mapped to language tag
        ".py": "python", ".js": "javascript", ".ts": "typescript",
        ".go": "go",     ".rs": "rust",       ".c": "c",
        ".cpp": "cpp",   ".h": "c",           ".java": "java",
        ".sh": "bash",   ".rb": "ruby",       ".php": "php",
        ".cs": "csharp", ".kt": "kotlin",     ".swift": "swift",
    },
}

# ─── Paths ─────────────────────────────────────────────────────────────────────
PATHS = {
    "checkpoint":  "dhenchooo_checkpoint.pt",
    "tokenizer":   "dhenchooo_tokenizer.json",
    "crawl_log":   "dhenchooo_crawled.txt",   # repos already seen (avoid re-training)
}
