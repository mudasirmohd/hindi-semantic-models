"""Build tokenized Hindi training corpora for the embedding models.

Uses HinExSum's own preprocessing (NFC normalization, danda-aware sentence
splitting, Devanagari tokenizer, light suffix stemmer) so that the trained
vocabulary matches the tokens HinExSum looks up. Writes two corpora, one
sentence per line, space-separated tokens:

  corpus_surface.txt  surface tokens (stopwords kept)
  corpus_stem.txt     stemmed tokens (stopwords kept)

Every sentence that also occurs in an evaluation document (XL-Sum Hindi
validation/test, ILSUM Hindi) is dropped, so the embeddings never see the
evaluation text.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import re
import sys
import time
from multiprocessing import Pool

HINEXSUM_CODE = os.path.join(os.environ.get("HINEXSUM_HOME", os.path.expanduser("~/projects/hindi-summarizer")), "code")
sys.path.insert(0, HINEXSUM_CODE)
from hinexsum import preprocessing as pp  # noqa: E402

MIN_TOKENS = 3
_DOC_TAG = re.compile(r"^</?doc[^>]*>$")


def sent_key(tokens: list[str]) -> bytes:
    return hashlib.blake2b(" ".join(tokens).encode(), digest_size=8).digest()


def eval_keys(paths: list[str]) -> set[bytes]:
    keys: set[bytes] = set()
    for p in paths:
        with open(p, encoding="utf-8") as fh:
            for line in fh:
                doc = json.loads(line)
                for field in ("text", "summary", "title"):
                    if isinstance(doc.get(field), str):
                        for s in pp.split_sentences(doc[field]):
                            toks = pp.tokenize(s)
                            if len(toks) >= MIN_TOKENS:
                                keys.add(sent_key(toks))
    return keys


_EVAL: set[bytes] = set()
_TEXT = False


def _init(keys: set[bytes], text: bool = False) -> None:
    global _EVAL, _TEXT
    _EVAL, _TEXT = keys, text


def process_chunk(lines: list[str]) -> tuple[list[str], list[str], int, int]:
    surf, stem, kept, dropped = [], [], 0, 0
    for line in lines:
        line = line.strip()
        if not line or _DOC_TAG.match(line):
            continue
        for s in pp.split_sentences(line):
            toks = pp.tokenize(s)
            if len(toks) < MIN_TOKENS:
                continue
            if sent_key(toks) in _EVAL:
                dropped += 1
                continue
            kept += 1
            if _TEXT:
                surf.append(s)
                continue
            surf.append(" ".join(toks))
            stem.append(" ".join(pp.stem(t) for t in toks))
    return surf, stem, kept, dropped


def chunks(paths: list[str], size: int):
    buf: list[str] = []
    for p in paths:
        with open(p, encoding="utf-8", errors="ignore") as fh:
            for line in fh:
                buf.append(line)
                if len(buf) >= size:
                    yield buf
                    buf = []
    if buf:
        yield buf


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--inputs", nargs="+", required=True, help="text files or globs")
    ap.add_argument("--eval-jsonl", nargs="+", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--workers", type=int, default=28)
    ap.add_argument("--chunk", type=int, default=20000)
    ap.add_argument("--text", action="store_true",
                    help="write only corpus_text.txt: normalized sentences with punctuation (for BERT)")
    args = ap.parse_args()

    paths = sorted(p for g in args.inputs for p in glob.glob(g, recursive=True) if os.path.isfile(p))
    t0 = time.time()
    keys = eval_keys(args.eval_jsonl)
    print(f"[{time.strftime('%H:%M:%S')}] {len(keys)} evaluation sentence keys from {len(args.eval_jsonl)} files; {len(paths)} input files", flush=True)

    os.makedirs(args.out_dir, exist_ok=True)
    kept = dropped = 0
    names = ("corpus_text.txt", os.devnull) if args.text else ("corpus_surface.txt", "corpus_stem.txt")
    with open(os.path.join(args.out_dir, names[0]), "w", encoding="utf-8") as fs, \
         open(os.path.join(args.out_dir, names[1]), "w", encoding="utf-8") as ft, \
         Pool(args.workers, initializer=_init, initargs=(keys, args.text)) as pool:
        for i, (surf, stem, k, d) in enumerate(pool.imap(process_chunk, chunks(paths, args.chunk))):
            fs.write("\n".join(surf) + ("\n" if surf else ""))
            ft.write("\n".join(stem) + ("\n" if stem else ""))
            kept += k
            dropped += d
            if i % 50 == 0:
                print(f"[{time.strftime('%H:%M:%S')}] chunk {i}: kept {kept:,} sentences, dropped {dropped:,} eval overlaps", flush=True)
    stats = {"input_files": len(paths), "sentences_kept": kept, "eval_overlaps_dropped": dropped,
             "eval_sentence_keys": len(keys), "min_tokens": MIN_TOKENS, "seconds": round(time.time() - t0)}
    with open(os.path.join(args.out_dir, "stats_text.json" if args.text else "stats.json"), "w") as fh:
        json.dump(stats, fh, indent=2)
    print(json.dumps(stats), flush=True)


if __name__ == "__main__":
    main()
