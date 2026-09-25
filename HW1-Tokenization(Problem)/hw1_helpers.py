"""Shared helpers for HW1, used by both the student and reference notebooks.

Nothing here is graded. These functions exist so that no time goes into metric
definitions or file handling.
"""

from __future__ import annotations

import csv
import os
import time

import numpy as np


# ------------------------------------------------------------------ files

def read_lines(path, max_lines=None):
    """Read a text file line by line, dropping blank lines."""
    out = []
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if max_lines is not None and i >= max_lines:
                break
            line = line.strip()
            if line:
                out.append(line)
    return out


def corpus_stats(path):
    """Count lines, characters and whitespace-delimited words."""
    lines = chars = words = 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            lines += 1
            chars += len(line)
            words += len(line.split())
    return {"lines": lines, "chars": chars, "words": words}


def save_table(rows, path):
    """Write a list of dictionaries as CSV."""
    if not rows:
        return
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    keys = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    return path


# ------------------------------------------------------------------ tokenizers

def train_hf_bpe(corpus_path, vocab_size, save_path=None):
    """Train a byte-level BPE tokenizer with the `tokenizers` library."""
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

    tok = Tokenizer(models.BPE(unk_token=None))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        show_progress=False,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    )
    t0 = time.time()
    tok.train([corpus_path], trainer)
    elapsed = time.time() - t0
    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        tok.save(save_path)
    return tok, elapsed


def load_hf_bpe(path):
    from tokenizers import Tokenizer
    return Tokenizer.from_file(path)


def count_tokens(tokenizer, path, max_lines=None):
    """Count the tokens the tokenizer produces over a corpus."""
    total = 0
    with open(path, encoding="utf-8") as f:
        batch = []
        for i, line in enumerate(f):
            if max_lines is not None and i >= max_lines:
                break
            line = line.strip()
            if not line:
                continue
            batch.append(line)
            if len(batch) >= 2000:
                total += sum(len(e.ids) for e in tokenizer.encode_batch(batch))
                batch = []
        if batch:
            total += sum(len(e.ids) for e in tokenizer.encode_batch(batch))
    return total


class WordSentences:
    """gensim input that streams whitespace-delimited tokens."""

    def __init__(self, path, lowercase=False):
        self.path = path
        self.lowercase = lowercase

    def __iter__(self):
        with open(self.path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                if self.lowercase:
                    line = line.lower()
                yield line.split()


class BpeSentences:
    """gensim input that streams subword tokens."""

    def __init__(self, path, tokenizer, batch=2000):
        self.path = path
        self.tokenizer = tokenizer
        self.batch = batch

    def __iter__(self):
        buf = []
        with open(self.path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                buf.append(line)
                if len(buf) >= self.batch:
                    for enc in self.tokenizer.encode_batch(buf):
                        yield enc.tokens
                    buf = []
        if buf:
            for enc in self.tokenizer.encode_batch(buf):
                yield enc.tokens


# ------------------------------------------------------------------ evaluation

def eval_word_list(path):
    """Path to an evaluation file shipped with gensim."""
    import gensim
    base = os.path.join(os.path.dirname(gensim.__file__), "test", "test_data")
    return os.path.join(base, path)


def analogy_words(path=None):
    """The set of words appearing in questions-words.txt."""
    path = path or eval_word_list("questions-words.txt")
    words = set()
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.startswith(":"):
                continue
            words.update(line.split())
    return words


def wordsim_words(path=None):
    """The set of words appearing in wordsim353.tsv."""
    path = path or eval_word_list("wordsim353.tsv")
    words = set()
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) >= 3:
                words.update(parts[:2])
    return words


def coverage(vocab, words):
    """Fraction of the evaluation words present in the vocabulary."""
    words = set(words)
    if not words:
        return 0.0
    return sum(1 for w in words if w in vocab) / len(words)


def restrict_kv(kv, words):
    """Restrict a KeyedVectors to a given set of words.

    Subword models and word models must be scored over the same candidate set.
    Analogy accuracies computed over different candidate sets are not comparable.
    """
    from gensim.models import KeyedVectors

    keys = [w for w in sorted(set(words)) if w in kv.key_to_index]
    out = KeyedVectors(vector_size=kv.vector_size)
    if keys:
        out.add_vectors(keys, np.asarray([kv[w] for w in keys], dtype=np.float32))
    return out, len(keys)


def top_words(path, n=50_000, min_count=5):
    """The n most frequent words of a corpus, used as the candidate set."""
    from collections import Counter
    c = Counter()
    with open(path, encoding="utf-8") as f:
        for line in f:
            c.update(line.split())
    return [w for w, k in c.most_common(n) if k >= min_count]


def subword_word_vectors(kv, tokenizer, words, prefix_space=True):
    """Build word vectors from a subword embedding as the mean of its subwords.

    Returns:
        (KeyedVectors, number of words built)
    """
    from gensim.models import KeyedVectors

    words = sorted(set(words))
    texts = [(" " + w) if prefix_space else w for w in words]
    encs = tokenizer.encode_batch(texts)

    keys, vecs = [], []
    for w, enc in zip(words, encs):
        parts = [kv[t] for t in enc.tokens if t in kv.key_to_index]
        if not parts:
            continue
        keys.append(w)
        vecs.append(np.mean(parts, axis=0))

    out = KeyedVectors(vector_size=kv.vector_size)
    if keys:
        out.add_vectors(keys, np.asarray(vecs, dtype=np.float32))
    return out, len(keys)


def evaluate(kv, restrict_vocab=300_000):
    """Compute analogy accuracy and the WordSim-353 Spearman correlation."""
    result = {}
    try:
        acc, sections = kv.evaluate_word_analogies(
            eval_word_list("questions-words.txt"),
            restrict_vocab=restrict_vocab,
            dummy4unknown=False,
        )
        result["analogy_acc"] = acc
        result["analogy_sections"] = {
            s["section"]: (len(s["correct"]),
                           len(s["correct"]) + len(s["incorrect"]))
            for s in sections
        }
    except Exception as exc:                       # vocabulary too small to score
        result["analogy_acc"] = float("nan")
        result["analogy_error"] = str(exc)

    try:
        pearson, spearman, oov = kv.evaluate_word_pairs(
            eval_word_list("wordsim353.tsv"), dummy4unknown=False)
        result["wordsim_spearman"] = getattr(spearman, "statistic", None)
        if result["wordsim_spearman"] is None:
            result["wordsim_spearman"] = float(spearman[0])
        result["wordsim_oov_pct"] = oov
    except Exception as exc:
        result["wordsim_spearman"] = float("nan")
        result["wordsim_error"] = str(exc)

    return result
