"""Train a Hindi WordPiece tokenizer for BERT pre-training.

Accent stripping and lower-casing are OFF: BERT's default accent stripping deletes Devanagari vowel
signs (matras), which would destroy Hindi words.
"""
from __future__ import annotations

import argparse
import itertools
import os

from tokenizers import BertWordPieceTokenizer
from transformers import BertTokenizerFast


def lines(path: str, limit: int):
    with open(path, encoding="utf-8") as fh:
        yield from itertools.islice(fh, limit)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--vocab-size", type=int, default=32000)
    ap.add_argument("--max-lines", type=int, default=10_000_000)
    args = ap.parse_args()

    tok = BertWordPieceTokenizer(clean_text=True, handle_chinese_chars=False,
                                 strip_accents=False, lowercase=False)
    tok.train_from_iterator(lines(args.corpus, args.max_lines), vocab_size=args.vocab_size,
                            min_frequency=5, limit_alphabet=1000,
                            special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"])
    os.makedirs(args.out, exist_ok=True)
    tok.save_model(args.out)
    # Build from the trained object: under transformers 5, BertTokenizerFast(vocab_file=...) loads
    # only the special tokens and turns every Hindi word into [UNK].
    hf = BertTokenizerFast(tokenizer_object=tok._tokenizer, unk_token="[UNK]", sep_token="[SEP]",
                           pad_token="[PAD]", cls_token="[CLS]", mask_token="[MASK]",
                           do_lower_case=False, strip_accents=False, tokenize_chinese_chars=False)
    hf.save_pretrained(args.out)
    reloaded = BertTokenizerFast.from_pretrained(args.out)
    sample = "भारत की राजधानी नई दिल्ली है।"
    toks = reloaded.tokenize(sample)
    print(sample, "->", toks, "| vocab", reloaded.vocab_size)
    if "[UNK]" in toks or reloaded.vocab_size != args.vocab_size:
        raise SystemExit("tokenizer canary failed")


if __name__ == "__main__":
    main()
