"""BLEU and chrF++ for every saved HinExSum summary set (results/summaries), as a second family of
automatic metrics next to ROUGE.

* chrF++ (character n-grams up to 6 plus word n-grams up to 2, sacrebleu) on the raw text: character
  n-grams are robust to Hindi inflection, which word-level ROUGE and BLEU penalise.
* BLEU (sacrebleu) on text pre-tokenised with HinExSum's Devanagari-aware tokenizer (the same tokens
  ROUGE counts), with tokenize="none", because sacrebleu's default tokenizer does not split the danda.

Corpus-level scores are reported for each system and set. Differences from the cc.hi.300 baseline are
tested with a paired bootstrap over documents on sentence-level scores (1000 resamples, seed 42, the same
resample matrix as the ROUGE analysis). Writes results/text_metrics.jsonl.
"""
from __future__ import annotations

import glob
import json
import os
import sys

import numpy as np
from sacrebleu.metrics import BLEU, CHRF

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import eval_extrinsic as ee  # noqa: E402

sys.path.insert(0, os.path.join(ee.H, "code"))
from hinexsum.rouge import _tokens  # noqa: E402

OUT = os.path.join(ee.EXP, "results/text_metrics.jsonl")
CHRF_PP = CHRF(word_order=2)
BLEU_C = BLEU(tokenize="none")
BLEU_S = BLEU(tokenize="none", effective_order=True)


def tok(s: str) -> str:
    return " ".join(_tokens(s))


def main() -> None:
    names = {}
    for r in ee.rows():
        names[(r["model"], r["dataset"], r["split"])] = r
    refs_cache, rows = {}, []
    for (model, ds, split) in sorted(names):
        path = ee.perdoc_path(model, ds, split).replace(ee.PERDOC, ee.SUMMARIES)
        if not os.path.exists(path):
            print(f"no summaries yet: {model} {ds} {split}", file=sys.stderr)
            continue
        if (ds, split) not in refs_cache:
            refs_cache[(ds, split)] = [ref for _, ref in ee.load_docs(ds, split, ee.N_DOCS)]
        refs = refs_cache[(ds, split)]
        with open(path, encoding="utf-8") as fh:
            hyps = [json.loads(l)["summary"] for l in fh]
        assert len(hyps) == len(refs), (model, ds, split)
        chrf_doc = np.array([CHRF_PP.sentence_score(h, [r]).score for h, r in zip(hyps, refs)])
        bleu_doc = np.array([BLEU_S.sentence_score(tok(h), [tok(r)]).score for h, r in zip(hyps, refs)])
        rows.append({"model": model, "dataset": ds, "split": split,
                     "chrfpp": round(CHRF_PP.corpus_score(hyps, [refs]).score, 2),
                     "bleu": round(BLEU_C.corpus_score([tok(h) for h in hyps], [[tok(r) for r in refs]]).score, 2),
                     "_chrf_doc": chrf_doc, "_bleu_doc": bleu_doc})
    base = {(r["dataset"], r["split"]): r for r in rows if r["model"] == ee.BASELINE}
    with open(OUT, "w", encoding="utf-8") as out:
        for r in rows:
            b = base.get((r["dataset"], r["split"]))
            if b is not None and r["model"] != ee.BASELINE:
                n = len(r["_chrf_doc"])
                idx = np.random.default_rng(ee.SEED).integers(0, n, (ee.BOOT, n))
                for key, short in (("_chrf_doc", "chrfpp"), ("_bleu_doc", "bleu")):
                    d = r[key] - b[key]
                    lo, hi = np.percentile(d[idx].mean(1), [2.5, 97.5])
                    r[f"{short}_doc_gap"] = round(float(d.mean()), 3)
                    r[f"{short}_doc_gap_ci95"] = [round(float(lo), 3), round(float(hi), 3)]
            clean = {k: v for k, v in r.items() if not k.startswith("_")}
            out.write(json.dumps(clean, ensure_ascii=False) + "\n")
            print(f"{r['dataset']:9} {r['split']:4} {r['model']:24} chrF++ {r['chrfpp']:6.2f}  BLEU {r['bleu']:6.2f}  "
                  f"gap chrF++ {clean.get('chrfpp_doc_gap', '')} {clean.get('chrfpp_doc_gap_ci95', '')}  "
                  f"BLEU {clean.get('bleu_doc_gap', '')} {clean.get('bleu_doc_gap_ci95', '')}")


if __name__ == "__main__":
    main()
