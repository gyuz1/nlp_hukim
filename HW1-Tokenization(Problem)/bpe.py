"""Part 1. Byte Pair Encoding.

Natural Language Processing (2026) -- Homework 1

Implement the algorithm on slides 14-21 using the **standard library only**.
Do not import `tokenizers`, `transformers`, or `numpy` in this file.

Specification -- grading checks that your code follows it exactly.

1. The base vocabulary is the set of characters in the training corpus.
   Use no word-boundary marker, matching the slides, which write
   "hugs" as h+u+g+s.
2. Each step merges the most frequent pair. Ties break **alphabetically**.
   Never merge a pair whose frequency is below 2.
3. `encode` splits text on whitespace, replays the merges on each word
   independently, and inserts the special token `<sp>` between words.
   `<sp>` takes no part in training; it only restores whitespace.
4. Characters absent from the training corpus encode as `<unk>`.
   **Do not raise an exception.**
5. `build_vocab` orders tokens as `<unk>`, `<sp>`, base characters in
   alphabetical order, then merged tokens in the order they were learned.
   That order defines the token ids.

Therefore `decode(encode(t)) == t` holds only for text whose whitespace is
normalized to single spaces and whose characters are in the training alphabet.

Run:
    python test_bpe_public.py
"""

from __future__ import annotations

import heapq
from collections import Counter, defaultdict

UNK = "<unk>"
SP = "<sp>"
SPECIALS = [UNK, SP]


# -----------------------------------------------------------------------------
# 1-a. Word frequencies
# -----------------------------------------------------------------------------
def count_words(lines):
    """Count whitespace-delimited word frequencies over a sequence of lines.

    Args:
        lines: a sequence of strings.
    Returns:
        dict[str, int] mapping word to frequency.

    >>> count_words(["hug pug hug"]) == {"hug": 2, "pug": 1}
    True
    """
    # HINT: collections.Counter accepts any iterable of words.
    counts = Counter()
    for line in lines:
        counts.update(line.split())
    return dict(counts) #type conversion


# -----------------------------------------------------------------------------
# 1-b. Merge training
# -----------------------------------------------------------------------------
def train_bpe(word_freqs, num_merges):
    """Learn merge rules from a word frequency table.

    Follow the procedure on slide 14.
      1. Split each word into characters.
      2. Count adjacent pairs.
      3. Select the most frequent pair (ties break alphabetically).
      4. Merge that pair into a single token.
      5. Repeat steps 2-4 `num_merges` times.

    Stop early when no pair reaches a frequency of 2.

    Args:
        word_freqs: mapping from word to frequency.
        num_merges: number of merges, k.
    Returns:
        list[tuple[str, str]] of merge rules in the order they were learned.

    >>> train_bpe({"hug": 10, "pug": 5, "pun": 12, "bun": 4, "hugs": 5}, 1)
    [('u', 'g')]
    """
    # HINT: rescanning every word on each merge is slow. Keep an index from each
    #       pair to the words that contain it and update only those words.
    # HINT: do not recount pair frequencies from scratch. Subtract and add the
    #       contribution of the words that changed.
    # HINT: a 5 MB corpus with 1,000 merges must finish within a few minutes.
    splits = {word: list(word) for word in word_freqs}
    merges = []

    # Count every pair once, and remember which words contain it.
    pair_freqs = Counter()
    pair_to_words = defaultdict(set)
    for word, freq in word_freqs.items():
        symbols = splits[word]
        for i in range(len(symbols) - 1):
            pair = (symbols[i], symbols[i + 1])
            pair_freqs[pair] += freq
            pair_to_words[pair].add(word)

    # Max-heap on frequency; (-freq, pair) also breaks ties alphabetically.
    # Entries go stale when a frequency changes and are skipped when popped.
    heap = [(-f, p) for p, f in pair_freqs.items()]
    heapq.heapify(heap)

    while len(merges) < num_merges:
        while heap and pair_freqs.get(heap[0][1], 0) != -heap[0][0]:
            heapq.heappop(heap)
        if not heap or -heap[0][0] < 2:
            break
        best = heap[0][1]
        merges.append(best)

        # Update only the words that contain the merged pair.
        a, b = best
        changed = set()
        for word in list(pair_to_words[best]):
            symbols = splits[word]
            freq = word_freqs[word]
            for i in range(len(symbols) - 1):
                pair = (symbols[i], symbols[i + 1])
                pair_freqs[pair] -= freq
                pair_to_words[pair].discard(word)
                changed.add(pair)

            new = []
            i = 0
            while i < len(symbols):
                if i < len(symbols) - 1 and symbols[i] == a and symbols[i + 1] == b:
                    new.append(a + b)
                    i += 2
                else:
                    new.append(symbols[i])
                    i += 1
            splits[word] = new

            for i in range(len(new) - 1):
                pair = (new[i], new[i + 1])
                pair_freqs[pair] += freq
                pair_to_words[pair].add(word)
                changed.add(pair)

        for pair in changed:
            if pair_freqs[pair] > 0:
                heapq.heappush(heap, (-pair_freqs[pair], pair))
            else:
                del pair_freqs[pair]
                pair_to_words.pop(pair, None)

    return merges


# -----------------------------------------------------------------------------
# 1-c. Vocabulary
# -----------------------------------------------------------------------------
def build_vocab(word_freqs, merges):
    """Build the token list. Its index is the token id.

    Keep this order exactly:
        `<unk>`, `<sp>`, base characters (alphabetical), merged tokens
        (in the order they were learned)

    Args:
        word_freqs: mapping from word to frequency.
        merges: merge rules returned by `train_bpe`.
    Returns:
        list[str] of tokens.
    """
    # HINT: the base characters are the distinct characters of every word.
    chars = sorted({ch for word in word_freqs for ch in word})
    return SPECIALS + chars + [a + b for a, b in merges]


# -----------------------------------------------------------------------------
# Provided. Bookkeeping tables derived from the merges and the vocabulary.
# -----------------------------------------------------------------------------
def build_tables(merges, vocab):
    """Return (ranks, alphabet, index).

    ranks    -- {pair: rank}, rank 0 being the merge learned first
    alphabet -- the set of base characters, the ones a word may be split into
    index    -- {token: id}
    """
    ranks = {pair: i for i, pair in enumerate(merges)}
    alphabet = {t for t in vocab if len(t) == 1 and t not in SPECIALS}
    index = {t: i for i, t in enumerate(vocab)}
    return ranks, alphabet, index


# -----------------------------------------------------------------------------
# 1-d. Segmentation
# -----------------------------------------------------------------------------
def tokenize(text, merges, vocab):
    """Segment text into subwords, keeping word boundaries.

    Split on whitespace and segment each word on its own. Within a word,
    repeatedly merge the adjacent pair of the **highest rank**, which replays
    the merges in the order they were learned. A character outside the training
    alphabet becomes `<unk>`.

    Args:
        text: a string.
        merges: merge rules returned by `train_bpe`.
        vocab: token list returned by `build_vocab`.
    Returns:
        list[list[str]], one inner list per word.

    >>> merges = train_bpe({"hug": 10, "pug": 5, "pun": 12, "bun": 4, "hugs": 5}, 3)
    >>> tokenize("hugs", merges, build_vocab({"hug": 10, "hugs": 5}, merges))
    [['hug', 's']]
    """
    # HINT: `build_tables` gives you the rank table and the alphabet.
    # HINT: words repeat often, so caching word -> segmentation speeds this up.
    ranks, alphabet, _ = build_tables(merges, vocab)
    cache = {}
    result = []
    for word in text.split():
        if word not in cache:
            symbols = [ch if ch in alphabet else UNK for ch in word]
            while len(symbols) > 1:
                pairs = [(symbols[i], symbols[i + 1]) for i in range(len(symbols) - 1)]
                best = min(pairs, key=lambda p: ranks.get(p, float("inf")))
                if best not in ranks:
                    break
                a, b = best
                new = []
                i = 0
                while i < len(symbols):
                    if i < len(symbols) - 1 and symbols[i] == a and symbols[i + 1] == b:
                        new.append(a + b)
                        i += 2
                    else:
                        new.append(symbols[i])
                        i += 1
                symbols = new
            cache[word] = symbols
        result.append(list(cache[word]))
    return result


# -----------------------------------------------------------------------------
# 1-e. Encoding and decoding
# -----------------------------------------------------------------------------
def encode(text, merges, vocab):
    """Encode text as token ids, inserting the id of `<sp>` between words.

    Args:
        text: a string.
        merges: merge rules returned by `train_bpe`.
        vocab: token list returned by `build_vocab`.
    Returns:
        list[int]
    """
    # HINT: segment with `tokenize`, then map each token through the index table.
    # HINT: a token missing from the vocabulary becomes the id of `<unk>`.
    _, _, index = build_tables(merges, vocab)
    unk_id = index[UNK]
    sp_id = index[SP]
    ids = []
    for n, tokens in enumerate(tokenize(text, merges, vocab)):
        if n > 0:
            ids.append(sp_id)
        ids.extend(index.get(t, unk_id) for t in tokens)
    return ids


def decode(ids, vocab):
    """Decode token ids back to text, rendering `<sp>` as a single space.

    Args:
        ids: a sequence of token ids.
        vocab: token list returned by `build_vocab`.
    Returns:
        str
    """
    return "".join(" " if vocab[i] == SP else vocab[i] for i in ids)
