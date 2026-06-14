"""
DHENCHOOO - GitHub Crawler
Fetches public repos → code files → yields text for training.
No git clone needed — uses GitHub REST API directly.
Respects rate limits automatically.
"""

import os
import time
import base64
import random
import requests
from typing import Iterator, List, Optional, Tuple, Set
from config import GITHUB_CONFIG, PATHS


class GitHubCrawler:
    """
    Crawls GitHub public repos for code files.

    Usage:
        crawler = GitHubCrawler()
        for repo_name, lang, code_text in crawler.stream_code():
            # train on code_text
            ...
    """

    BASE_URL = "https://api.github.com"

    def __init__(self):
        self.token      = GITHUB_CONFIG["token"] or os.environ.get("GITHUB_TOKEN", "")
        self.languages  = GITHUB_CONFIG["languages"]
        self.min_stars  = GITHUB_CONFIG["min_stars"]
        self.extensions = GITHUB_CONFIG["extensions"]
        self.max_files  = GITHUB_CONFIG["files_per_repo"]
        self.seen_repos: Set[str] = self._load_seen()
        self._search_page: int    = 1
        self._lang_idx:    int    = 0

    # ── Rate-limit-aware requests ──────────────────────────────────────────────

    def _headers(self) -> dict:
        h = {"Accept": "application/vnd.github+json",
             "X-GitHub-Api-Version": "2022-11-28"}
        if self.token:
            h["Authorization"] = f"Bearer {self.token}"
        return h

    def _get(self, url: str, params: dict = None, retries: int = 3) -> Optional[dict]:
        for attempt in range(retries):
            try:
                r = requests.get(url, headers=self._headers(), params=params, timeout=15)
                if r.status_code == 200:
                    return r.json()
                if r.status_code == 403:
                    # Rate limited
                    reset_at = int(r.headers.get("X-RateLimit-Reset", time.time() + 60))
                    wait     = max(reset_at - time.time(), 5) + 2
                    print(f"  [GitHub] Rate limited. Waiting {wait:.0f}s …")
                    time.sleep(wait)
                elif r.status_code == 404:
                    return None
                else:
                    print(f"  [GitHub] HTTP {r.status_code} for {url}")
                    time.sleep(2 ** attempt)
            except requests.RequestException as e:
                print(f"  [GitHub] Request error: {e}")
                time.sleep(2 ** attempt)
        return None

    # ── Seen-repos log ─────────────────────────────────────────────────────────

    def _load_seen(self) -> Set[str]:
        path = PATHS["crawl_log"]
        if os.path.exists(path):
            with open(path) as f:
                return set(line.strip() for line in f if line.strip())
        return set()

    def _mark_seen(self, full_name: str) -> None:
        self.seen_repos.add(full_name)
        with open(PATHS["crawl_log"], "a") as f:
            f.write(full_name + "\n")

    # ── Repo search ────────────────────────────────────────────────────────────

    def _search_repos(self, language: str, page: int) -> List[dict]:
        """Returns a list of repo metadata dicts."""
        query  = f"language:{language} stars:>={self.min_stars} fork:false"
        params = {"q": query, "sort": "updated", "order": "desc",
                  "per_page": 30, "page": page}
        data = self._get(f"{self.BASE_URL}/search/repositories", params)
        if not data:
            return []
        items = data.get("items", [])
        # Shuffle slightly so we don't always pick the most-starred repos
        random.shuffle(items)
        return items

    # ── File fetching ──────────────────────────────────────────────────────────

    def _list_repo_files(self, owner: str, repo: str, path: str = "") -> List[dict]:
        """Recursively list files in a repo up to self.max_files total."""
        url  = f"{self.BASE_URL}/repos/{owner}/{repo}/contents/{path}"
        data = self._get(url)
        if not data or not isinstance(data, list):
            return []
        files = []
        for item in data:
            if item["type"] == "file":
                _, ext = os.path.splitext(item["name"])
                if ext.lower() in self.extensions:
                    files.append(item)
            elif item["type"] == "dir" and len(files) < self.max_files:
                sub = self._list_repo_files(owner, repo, item["path"])
                files.extend(sub)
            if len(files) >= self.max_files:
                break
        return files[:self.max_files]

    def _fetch_file_content(self, file_meta: dict) -> Optional[str]:
        """Download and decode a single file's content."""
        data = self._get(file_meta["url"])
        if not data:
            return None
        encoding = data.get("encoding", "")
        content  = data.get("content", "")
        if encoding == "base64":
            try:
                return base64.b64decode(content).decode("utf-8", errors="replace")
            except Exception:
                return None
        return content or None

    # ── Main streaming interface ───────────────────────────────────────────────

    def stream_code(
        self, max_repos: int = GITHUB_CONFIG["max_repos_per_session"]
    ) -> Iterator[Tuple[str, str, str]]:
        """
        Infinite iterator that yields (repo_full_name, language, code_text).
        Cycles through all configured languages, rotating pages automatically.
        """
        repos_done = 0
        lang_cycle = list(self.languages)
        lang_idx   = 0
        page       = 1

        while repos_done < max_repos:
            lang     = lang_cycle[lang_idx % len(lang_cycle)]
            lang_idx += 1

            print(f"\n[Crawler] Searching GitHub: language={lang} page={page}")
            repos = self._search_repos(lang, page)
            page  = (page % 10) + 1   # cycle through pages 1-10

            if not repos:
                print(f"[Crawler] No results for {lang} p{page}, skipping.")
                time.sleep(2)
                continue

            for repo in repos:
                if repos_done >= max_repos:
                    return

                full_name = repo["full_name"]
                if full_name in self.seen_repos:
                    continue

                owner, repo_name = full_name.split("/", 1)
                print(f"[Crawler] → {full_name} ⭐{repo.get('stargazers_count',0)}")

                files = self._list_repo_files(owner, repo_name)
                if not files:
                    self._mark_seen(full_name)
                    continue

                repo_text_parts = []
                for f in files:
                    ext      = os.path.splitext(f["name"])[1].lower()
                    file_lang = self.extensions.get(ext, lang.lower())
                    content   = self._fetch_file_content(f)
                    if not content:
                        continue
                    n = len(content)
                    min_c = 200   # from config inline (avoid import cycle)
                    max_c = 50_000
                    if n < min_c or n > max_c:
                        continue
                    # Wrap with metadata tags so model learns language context
                    wrapped = (
                        f"<|{file_lang}|>\n"
                        f"# File: {f['name']}\n"
                        f"{content}\n"
                        f"<|endoffile|>\n"
                    )
                    repo_text_parts.append((full_name, file_lang, wrapped))
                    time.sleep(0.1)   # be polite to API

                for repo_full, file_lang, text in repo_text_parts:
                    yield repo_full, file_lang, text

                self._mark_seen(full_name)
                repos_done += 1
                time.sleep(0.5)   # small pause between repos
