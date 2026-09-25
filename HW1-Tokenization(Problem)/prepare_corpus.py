#!/usr/bin/env python3
"""HW1 corpus preparation.

Cuts three deterministic slices out of WikiText-103 (raw).

    data/corpus_train.txt   20M words by default    Part 1-6, Part 2
    data/corpus_valid.txt   500K words by default   token counts, evaluation
    data/corpus_small.txt   5 MB by default         Part 1, own implementation

Usage

    pip install datasets
    python prepare_corpus.py

    # from a local file
    python prepare_corpus.py --input /path/to/wikitext-103-raw/wiki.train.raw

    # without Hugging Face access
    python prepare_corpus.py --source github --train-words 5500000

The SHA-256 of each slice goes to data/checksums.txt. Publish those values so
that everyone works from the same corpus.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys

HF_REPO = "Salesforce/wikitext"
HF_CONFIG = "wikitext-103-raw-v1"

# For --source github, used where Hugging Face is unreachable. Collects the
# public corpora NLTK publishes through GitHub Pages plus a Gutenberg excerpt,
# giving about 6M words of English with case and punctuation intact.
NLTK_BASE = ("https://raw.githubusercontent.com/nltk/nltk_data"
             "/gh-pages/packages/corpora")
NLTK_PACKAGES = [
    ("gutenberg", ".txt"),      # literature, about 2.14M words
    ("reuters", None),          # news, about 1.40M words, no file extension
    ("abc", ".txt"),            # news, about 0.66M words
    ("state_union", ".txt"),    # speeches, about 0.35M words
    ("webtext", ".txt"),        # web text, about 0.30M words
    ("inaugural", ".txt"),      # speeches, about 0.14M words
]
EXTRA_URLS = [
    # Gutenberg excerpts, about 1.10M words
    ("big.txt", "https://raw.githubusercontent.com/dscape/spell"
                "/master/test/resources/big.txt"),
]
SKIP_NAMES = {"README", "cats.txt", "LICENSE", "citation.bib"}


def iter_hf_lines():
    """Stream the WikiText-103 raw training split through Hugging Face datasets."""
    try:
        from datasets import load_dataset
    except ImportError:
        sys.exit("datasets is missing. Run `pip install datasets`, or pass a "
                 "local text file with --input.")
    ds = load_dataset(HF_REPO, HF_CONFIG, split="train", streaming=True)
    for row in ds:
        yield row["text"]


def _fetch(url, dest):
    import urllib.request
    if os.path.exists(dest):
        return dest
    os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
    print("  downloading %s" % os.path.basename(dest))
    urllib.request.urlretrieve(url, dest)
    return dest


def _paragraphs(path):
    """Yield paragraphs, split on blank lines, one per line."""
    with open(path, encoding="utf-8", errors="replace") as f:
        buf = []
        for line in f:
            line = line.rstrip()
            if line.strip():
                buf.append(line.strip())
            elif buf:
                yield " ".join(buf)
                buf = []
        if buf:
            yield " ".join(buf)


def iter_github_lines(cache_dir):
    """Fetch the public corpora from GitHub and stream them paragraph by paragraph."""
    import re
    import unicodedata
    import zipfile

    dl = os.path.join(cache_dir, "_download")
    ex = os.path.join(cache_dir, "_extract")
    paths = []

    for name, ext in NLTK_PACKAGES:
        zpath = _fetch("%s/%s.zip" % (NLTK_BASE, name), os.path.join(dl, name + ".zip"))
        with zipfile.ZipFile(zpath) as z:
            z.extractall(ex)
        root = os.path.join(ex, name)
        for dirpath, _, names in os.walk(root):
            for n in sorted(names):
                if n in SKIP_NAMES or n.startswith("."):
                    continue
                if ext and not n.endswith(ext):
                    continue
                paths.append(os.path.join(dirpath, n))

    for name, url in EXTRA_URLS:
        paths.append(_fetch(url, os.path.join(dl, name)))

    print("  %d source files" % len(paths))
    ws = re.compile(r"\s+")
    for p in sorted(paths):
        for par in _paragraphs(p):
            s = ws.sub(" ", unicodedata.normalize("NFC", par)).strip()
            if len(s.split()) >= 5:
                yield s


def iter_file_lines(path):
    """Read a local text file or directory line by line."""
    if os.path.isdir(path):
        names = sorted(n for n in os.listdir(path) if not n.startswith("."))
        paths = [os.path.join(path, n) for n in names]
    else:
        paths = [path]
    for p in paths:
        with open(p, encoding="utf-8", errors="replace") as f:
            for line in f:
                yield line


def keep(line):
    """Drop blank lines and heading-only lines. Keep case and punctuation."""
    s = line.strip()
    if not s:
        return None
    if s.startswith("=") and s.endswith("="):   # " = Heading = "
        return None
    return s


def write_slice(lines, out_path, limit_words=None, limit_bytes=None):
    """Write lines until the word or byte limit is reached."""
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    n_words = n_bytes = n_lines = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for raw in lines:
            s = keep(raw)
            if s is None:
                continue
            w = len(s.split())
            b = len(s.encode("utf-8")) + 1
            if limit_words is not None and n_words + w > limit_words:
                break
            if limit_bytes is not None and n_bytes + b > limit_bytes:
                break
            f.write(s + "\n")
            n_words += w
            n_bytes += b
            n_lines += 1
    return {"path": out_path, "lines": n_lines, "words": n_words, "bytes": n_bytes}


def sha256(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(description="Build the HW1 corpus slices")
    ap.add_argument("--source", choices=["hf", "github"], default="hf",
                    help="hf: WikiText-103, recommended. "
                         "github: fallback where Hugging Face is unreachable")
    ap.add_argument("--input", help="local text file or directory; "
                                    "overrides --source")
    ap.add_argument("--out-dir", default="data")
    ap.add_argument("--train-words", type=int, default=20_000_000,
                    help="the github source tops out near 5_500_000")
    ap.add_argument("--valid-words", type=int, default=500_000)
    ap.add_argument("--small-mb", type=float, default=5.0)
    args = ap.parse_args()

    if args.input:
        source = lambda: iter_file_lines(args.input)
    elif args.source == "github":
        source = lambda: iter_github_lines(args.out_dir)
    else:
        source = iter_hf_lines
    stats = []

    print("building the training slice ...")
    it = source()
    stats.append(write_slice(it, os.path.join(args.out_dir, "corpus_train.txt"),
                             limit_words=args.train_words))
    print("building the validation slice, continuing from the training slice ...")
    stats.append(write_slice(it, os.path.join(args.out_dir, "corpus_valid.txt"),
                             limit_words=args.valid_words))

    print("building the small slice ...")
    train_path = stats[0]["path"]
    with open(train_path, encoding="utf-8") as f:
        stats.append(write_slice(f, os.path.join(args.out_dir, "corpus_small.txt"),
                                 limit_bytes=int(args.small_mb * 1_000_000)))

    lines_out = []
    print()
    print("%-24s %10s %12s %12s" % ("file", "lines", "words", "bytes"))
    for s in stats:
        print("%-24s %10d %12d %12d"
              % (os.path.basename(s["path"]), s["lines"], s["words"], s["bytes"]))
        lines_out.append("%s  %s" % (sha256(s["path"]), os.path.basename(s["path"])))

    check_path = os.path.join(args.out_dir, "checksums.txt")
    with open(check_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines_out) + "\n")
    print()
    print("SHA-256 -> %s" % check_path)
    for line in lines_out:
        print("  " + line)


if __name__ == "__main__":
    main()
