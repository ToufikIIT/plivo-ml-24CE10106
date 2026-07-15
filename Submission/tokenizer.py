"""Baseline tokenizer: raw UTF-8 bytes, vocab of 256. Simple, never fails on
unseen text — and treats a Devanagari character as 3 tokens. Think about
what that does to your model's context window and your token budget on the
Hindi part of the corpus.

You may replace this with anything you train ON THE PROVIDED CORPUS ONLY
(e.g., BPE), as long as:
  1. it can encode ARBITRARY UTF-8 text (byte-level fallback) and it is
     LOSSLESS: decode(encode(text)) == text, exactly. The scorer and the
     graders both verify this round-trip — a lossy tokenizer makes bpb
     meaningless and disqualifies the run.
  2. this file keeps exposing:  load() -> tokenizer object with
     .encode(str) -> list[int], .decode(list[int]) -> str, .vocab_size.
     train.py and evaluate.py call load() with NO arguments — keep any
     extra parameters optional.
  3. anything it needs is saved under your submission folder and loaded by
     load() with no internet. Grading runs with cwd = your folder; resolve
     saved files relative to __file__ to be safe.
"""
# import json


# class ByteTokenizer:
#     vocab_size = 256

#     def encode(self, text):
#         return list(text.encode("utf-8"))

#     def decode(self, ids):
#         return bytes(ids).decode("utf-8", errors="replace")

#     def save(self, path):
#         with open(path, "w") as f:
#             json.dump({"type": "byte"}, f)


# def load(path=None):
#     """Return the tokenizer used by evaluate.py. Replace as needed."""
#     return ByteTokenizer()

"""BPE tokenizer trained on train_corpus.txt only. Byte-level base (256
symbols) + learned merges. Always lossless: every merge is just a
concatenation of byte sequences, and unmerged bytes fall through as-is, so
decode(encode(text)) == text for ANY UTF-8 text, seen or not."""

import json
import os
from collections import Counter

SAVE_PATH = os.path.join(os.path.dirname(__file__), "bpe_merges.json")


class BPETokenizer:
    def __init__(self, merges=None):
        self.merges = merges or []
        self.merge_rank = {tuple(pair): i for i, (pair, _) in enumerate(self.merges)}
        self.vocab = {i: bytes([i]) for i in range(256)}
        next_id = 256
        for pair, _ in self.merges:
            self.vocab[next_id] = self.vocab[pair[0]] + self.vocab[pair[1]]
            next_id += 1
        self.vocab_size = next_id

    # def _merge_ids(self, ids):
    #     ids = list(ids)
    #     while len(ids) >= 2:
    #         pairs = [(ids[i], ids[i + 1]) for i in range(len(ids) - 1)]
    #         ranked = [(self.merge_rank[p], i) for i, p in enumerate(pairs)
    #                   if p in self.merge_rank]
    #         if not ranked:
    #             break
    #         _, i = min(ranked)
    #         a, b = ids[i], ids[i + 1]
    #         new_id = 256 + self.merge_rank[(a, b)]
    #         ids = ids[:i] + [new_id] + ids[i + 2:]
    #     return ids
    
    def _merge_ids(self, ids):
        ids = list(ids)
        for pair, new_id in self.merges:
            merged = []
            i = 0
            while i < len(ids):
                if i < len(ids) - 1 and (ids[i], ids[i + 1]) == pair:
                    merged.append(new_id)
                    i += 2
                else:
                    merged.append(ids[i])
                    i += 1
            ids = merged
        return ids

    def encode(self, text):
        return self._merge_ids(list(text.encode("utf-8")))

    def decode(self, ids):
        return b"".join(self.vocab[i] for i in ids).decode("utf-8", errors="replace")

    def save(self, path=SAVE_PATH):
        with open(path, "w") as f:
            json.dump([[list(p), nid] for p, nid in self.merges], f)

    @classmethod
    def load_trained(cls, path=SAVE_PATH):
        with open(path) as f:
            data = json.load(f)
        return cls([(tuple(p), nid) for p, nid in data])


def train_bpe(text, num_merges=256, verbose=True):
    ids = list(text.encode("utf-8"))
    merges = []
    next_id = 256
    for step in range(num_merges):
        pairs = Counter(zip(ids, ids[1:]))
        if not pairs:
            break
        pair, count = pairs.most_common(1)[0]
        if count < 2:
            break
        merges.append((pair, next_id))
        new_ids, i = [], 0
        while i < len(ids):
            if i < len(ids) - 1 and (ids[i], ids[i + 1]) == pair:
                new_ids.append(next_id); i += 2
            else:
                new_ids.append(ids[i]); i += 1
        ids = new_ids
        next_id += 1
        if verbose and (step + 1) % 64 == 0:
            print(f"  merge {step+1}/{num_merges}  vocab={next_id}  seq_len={len(ids):,}")
    return merges


def load(path=None):
    """train.py / evaluate.py call this with NO arguments."""
    merge_path = path or SAVE_PATH
    if os.path.exists(merge_path):
        return BPETokenizer.load_trained(merge_path)
    return BPETokenizer(merges=[])  