"""HW1 Part 1 public tests. Pass these before submitting.

    python -m pytest test_bpe_public.py -q
    or
    python test_bpe_public.py
"""

import bpe

# the example on slides 15-21
TOY = {"hug": 10, "pug": 5, "pun": 12, "bun": 4, "hugs": 5}


def test_count_words():
    got = bpe.count_words(["hug pug hug", "pun  bun", "hugs"])
    assert got == {"hug": 2, "pug": 1, "pun": 1, "bun": 1, "hugs": 1}


def test_merge_order():
    """The first three merges are u+g (20), u+n (16), h+ug (15)."""
    assert bpe.train_bpe(TOY, 3) == [("u", "g"), ("u", "n"), ("h", "ug")]


def test_vocab_growth():
    """Reproduce the vocabulary growth on slides 15, 19 and 21."""
    base = ["b", "g", "h", "n", "p", "s", "u"]
    expected = {
        0: base,
        1: base + ["ug"],
        2: base + ["ug", "un"],
        3: base + ["ug", "un", "hug"],
    }
    for k, want in expected.items():
        vocab = bpe.build_vocab(TOY, bpe.train_bpe(TOY, k))
        got = [t for t in vocab if t not in bpe.SPECIALS]
        assert got == want, (k, got)


def test_tokenize_matches_slide21():
    merges = bpe.train_bpe(TOY, 3)
    vocab = bpe.build_vocab(TOY, merges)
    got = bpe.tokenize("hug pug pun bun hugs", merges, vocab)
    assert got == [["hug"], ["p", "ug"], ["p", "un"], ["b", "un"], ["hug", "s"]]


def test_round_trip():
    """Round-trip holds for text within the training alphabet."""
    merges = bpe.train_bpe(TOY, 3)
    vocab = bpe.build_vocab(TOY, merges)
    for text in ["hug", "hugs bun", "pun pug hug bun hugs", "b u n"]:
        assert bpe.decode(bpe.encode(text, merges, vocab), vocab) == text


def test_ids_are_in_range():
    merges = bpe.train_bpe(TOY, 3)
    vocab = bpe.build_vocab(TOY, merges)
    ids = bpe.encode("hugs and bun", merges, vocab)
    assert all(0 <= i < len(vocab) for i in ids)


if __name__ == "__main__":
    fails = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print("PASS  " + name)
            except AssertionError as exc:
                fails += 1
                print("FAIL  %s  %s" % (name, exc))
            except NotImplementedError:
                fails += 1
                print("TODO  %s  (not implemented yet)" % name)
    print()
    print("%d failed" % fails)
    raise SystemExit(1 if fails else 0)
