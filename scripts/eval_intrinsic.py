"""Intrinsic evaluation of Hindi embedding models.

For each model: Hindi-WS word similarity (Spearman rho), fastText Hindi analogies (accuracy on the
questions whose four words are all among the 300k most frequent words), and lookup coverage of the
tokens HinExSum actually queries (stemmed, stopwords removed) on XL-Sum validation documents.
Stem models are queried with stemmed words, as HinExSum would query them.
"""
from __future__ import annotations

import argparse
import json
import sys

import numpy as np
from scipy.stats import spearmanr

sys.path.insert(0, "/home/mudasir/projects/hindi-summarizer/code")
from hinexsum import preprocessing as pp  # noqa: E402


def load(path: str):
    from gensim.models import KeyedVectors
    if path.endswith(".kv"):
        return KeyedVectors.load(path, mmap="r")
    from gensim.models.fasttext import load_facebook_vectors
    return load_facebook_vectors(path)


def in_vocab(kv, w: str) -> bool:
    return w in kv.key_to_index


def word_similarity(kv, path: str, stem: bool) -> dict:
    gold, pred, oov = [], [], 0
    for line in open(path, encoding="utf-8"):
        parts = line.strip().split(",")
        if len(parts) != 3:
            continue
        a, b, s = parts
        a, b = pp.normalize(a), pp.normalize(b)
        if stem:
            a, b = pp.stem(a), pp.stem(b)
        if not (in_vocab(kv, a) and in_vocab(kv, b)):
            oov += 1
            continue
        gold.append(float(s))
        pred.append(float(kv.similarity(a, b)))
    rho = spearmanr(gold, pred).statistic if len(gold) > 2 else float("nan")
    return {"ws_spearman": round(float(rho), 4), "ws_pairs_used": len(gold), "ws_pairs_oov": oov}


def analogies(kv, path: str, stem: bool, restrict: int = 300_000) -> dict:
    vocab = set(kv.index_to_key[:restrict])
    norm = kv.get_normed_vectors()[:restrict] if hasattr(kv, "get_normed_vectors") else None
    idx = {w: i for i, w in enumerate(kv.index_to_key[:restrict])}
    total = covered = correct = 0
    for line in open(path, encoding="utf-8"):
        if line.startswith(":"):
            continue
        q = [pp.normalize(w) for w in line.split()]
        if len(q) != 4:
            continue
        total += 1
        if stem:
            q = [pp.stem(w) for w in q]
        if len(set(q)) < 4 or not all(w in vocab for w in q):
            continue
        covered += 1
        a, b, c, d = (idx[w] for w in q)
        target = norm[b] - norm[a] + norm[c]
        sims = norm @ (target / np.linalg.norm(target))
        sims[[a, b, c]] = -np.inf
        correct += int(int(np.argmax(sims)) == d)
    return {"analogy_acc": round(correct / covered, 4) if covered else None,
            "analogy_covered": covered, "analogy_total": total}


def coverage(kv, jsonl: str, limit: int) -> dict:
    seen = hit = 0
    with open(jsonl, encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            if i >= limit:
                break
            for s in pp.split_sentences(json.loads(line)["text"]):
                for t in pp.preprocess_sentence(s):
                    seen += 1
                    hit += in_vocab(kv, t)
    return {"hinexsum_token_coverage": round(hit / seen, 4), "hinexsum_tokens": seen}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--stem", action="store_true", help="model was trained on stemmed tokens")
    ap.add_argument("--ws", default="data/eval_sets/Hindi-WS.txt")
    ap.add_argument("--analogy", default="data/eval_sets/questions-words-hi.txt")
    ap.add_argument("--coverage-jsonl", default="/home/mudasir/projects/hindi-summarizer/xlsum_hindi_val.jsonl")
    ap.add_argument("--coverage-docs", type=int, default=200)
    ap.add_argument("--out", required=True, help="JSONL file to append the result row to")
    args = ap.parse_args()

    kv = load(args.model)
    row = {"model": args.name, "path": args.model, "stem": args.stem, "vocab": len(kv.index_to_key)}
    row |= word_similarity(kv, args.ws, args.stem)
    row |= analogies(kv, args.analogy, args.stem)
    row |= coverage(kv, args.coverage_jsonl, args.coverage_docs)
    print(json.dumps(row, ensure_ascii=False))
    with open(args.out, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
