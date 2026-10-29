"""Train one Hindi static embedding model (Word2Vec or fastText) with gensim.

Example:
  python scripts/train.py --algo fasttext --arch sg --corpus data/corpus/corpus_surface.txt \
      --out models/hi-ft-sg-surface-300
Writes <out>.bin (word2vec binary, or fastText .bin with subwords for fastText),
<out>.kv (gensim KeyedVectors) and <out>.json (training configuration and stats).
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import time

from gensim.models import FastText, Word2Vec
from gensim.models.fasttext import save_facebook_model
from gensim.models.word2vec import LineSentence


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--algo", choices=["word2vec", "fasttext"], required=True)
    ap.add_argument("--arch", choices=["sg", "cbow"], default="sg")
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--dim", type=int, default=300)
    ap.add_argument("--window", type=int, default=5)
    ap.add_argument("--min-count", type=int, default=5)
    ap.add_argument("--max-vocab", type=int, default=1_000_000)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--negative", type=int, default=10)
    ap.add_argument("--buckets", type=int, default=1_000_000, help="fastText n-gram buckets")
    ap.add_argument("--workers", type=int, default=30)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    logging.basicConfig(format="%(asctime)s %(levelname)s %(message)s", level=logging.INFO,
                        handlers=[logging.FileHandler(args.out + ".log"), logging.StreamHandler()])
    common = dict(vector_size=args.dim, window=args.window, min_count=args.min_count,
                  max_final_vocab=args.max_vocab, sg=1 if args.arch == "sg" else 0,
                  negative=args.negative, workers=args.workers, seed=args.seed)
    t0 = time.time()
    sentences = LineSentence(args.corpus, max_sentence_length=10_000)
    if args.algo == "word2vec":
        model = Word2Vec(**common)
    else:
        model = FastText(bucket=args.buckets, min_n=3, max_n=6, **common)
    model.build_vocab(sentences)
    model.train(sentences, total_examples=model.corpus_count, epochs=args.epochs)

    model.wv.save(args.out + ".kv")
    if args.algo == "word2vec":
        model.wv.save_word2vec_format(args.out + ".bin", binary=True)
    else:
        save_facebook_model(model, args.out + ".bin")
    meta = {**vars(args), "vocab_size": len(model.wv), "corpus_sentences": model.corpus_count,
            "corpus_tokens": model.corpus_total_words, "train_seconds": round(time.time() - t0),
            "gensim": __import__("gensim").__version__}
    with open(args.out + ".json", "w") as fh:
        json.dump(meta, fh, indent=2)
    logging.info("done: %s", meta)


if __name__ == "__main__":
    main()
