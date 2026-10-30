"""Extrinsic evaluation: HinExSum with each embedding backend (protocol in docs/PLAN.md).

One call scores one backend on one evaluation set and appends a row to results/extrinsic.jsonl.
The configuration is the one Chapter 5 reports for the published equal-weight system, at a fixed
budget of three sentences:

  XL-Sum    all 7 features, equal weights, POS on, cluster-rr selection   (Ch 5 "Ours-full")
  ILSUM 2.0 all 7 features, equal weights, POS on, global-topn selection  (Ch 5 "all7 (equal)")

Sets: XL-Sum val = first 200 of xlsum_hindi_val.jsonl, XL-Sum test = first 200 of
xlsum_hindi_test.jsonl; ILSUM val = first 200 train rows, ILSUM test = first 200 test rows.

Backends:
  --kv PATH            static vectors (gensim .kv, or fastText .bin as used in Ch 5). The model
                       supplies both the big-vector expansion and the sentence vectors.
  --bert DIR           Hindi BERT: sentence vectors are the mean of the final-layer token
                       representations (special tokens and padding excluded). The big-vector
                       expansion uses --bv-kv, or with --bv-kv auto the trained static model with
                       the best mean validation ROUGE-1 F.

Rules enforced here, not left to discipline:
  * a test set is scored at most once per backend (a second attempt exits with an error), and only
    after the backend has validation rows for both corpora;
  * Stanza must be available, so the POS features never silently fall back to the heuristic;
  * per-document scores are kept, and every non-baseline row carries a paired-bootstrap ROUGE-1 F gap
    against the cc.hi.300 baseline on the same documents (1000 resamples, seed 42).
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import sys
import time

import numpy as np

H = os.environ.get("HINEXSUM_HOME", os.path.expanduser("~/projects/hindi-summarizer"))
EXP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(EXP, "results/extrinsic.jsonl")
PERDOC = os.path.join(EXP, "results/perdoc")
POS_CACHE = os.path.join(EXP, "results/cache/pos_counts.json")
BASELINE = "cc.hi.300 (Facebook)"
STATIC = ["hi-w2v-sg-surface-300", "hi-w2v-sg-stem-300", "hi-ft-sg-surface-300", "hi-ft-sg-stem-300"]
N_SENT, N_DOCS, SEED, BOOT = 3, 200, 42, 1000
SETS = {  # (dataset, split) -> (file, row filter, selection)
    ("XL-Sum", "val"): ("xlsum_hindi_val.jsonl", None, "cluster-rr"),
    ("XL-Sum", "test"): ("xlsum_hindi_test.jsonl", None, "cluster-rr"),
    ("ILSUM 2.0", "val"): ("ilsum_hindi.jsonl", "train", "global-topn"),
    ("ILSUM 2.0", "test"): ("ilsum_hindi.jsonl", "test", "global-topn"),
}
METRICS = ("rouge-1", "rouge-2", "rouge-l", "rouge-su4")
KEYS = {"rouge-1": "rouge1_f", "rouge-2": "rouge2_f", "rouge-l": "rougeL_f", "rouge-su4": "rougeSU4_f"}


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", file=sys.stderr, flush=True)


def rows() -> list[dict]:
    if not os.path.exists(RESULTS):
        return []
    with open(RESULTS, encoding="utf-8") as fh:
        return [json.loads(l) for l in fh if l.strip()]


def load_docs(dataset: str, split: str, limit: int) -> list[tuple[str, str]]:
    fname, want, _ = SETS[(dataset, split)]
    docs = []
    with open(os.path.join(H, fname), encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            r = json.loads(line)
            if want and r.get("split") != want:
                continue
            if r.get("text") and r.get("summary"):
                docs.append((r["text"], r["summary"]))
            if len(docs) >= limit:
                break
    return docs


def perdoc_path(model: str, dataset: str, split: str) -> str:
    safe = lambda s: s.replace(" ", "_").replace("(", "").replace(")", "").replace(".", "_")
    return os.path.join(PERDOC, f"{safe(model)}__{safe(dataset)}__{split}.jsonl")


# ---------------------------------------------------------------------------------------------
# POS: backend-independent, so tag each sentence once and share it across all backends
# ---------------------------------------------------------------------------------------------
def install_pos_cache():
    from hinexsum import ranking
    if ranking._get_stanza() is None:
        sys.exit("Stanza Hindi pipeline unavailable: POS features would fall back to the heuristic. "
                 "Run in ~/projects/hindi-summarizer/.venv (stanza + stanza.download('hi')).")
    cache = {}
    if os.path.exists(POS_CACHE):
        with open(POS_CACHE, encoding="utf-8") as fh:
            cache = json.load(fh)
    tagger = ranking.pos_counts

    def cached(sentence: str):
        hit = cache.get(sentence)
        if hit is None:
            d = tagger(sentence)
            hit = cache[sentence] = [d["nvc"], d["propn"]]
        return {"nvc": hit[0], "propn": hit[1]}

    ranking.pos_counts = cached  # compute_features looks the name up at call time

    def save():
        os.makedirs(os.path.dirname(POS_CACHE), exist_ok=True)
        tmp = POS_CACHE + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(cache, fh, ensure_ascii=False)
        os.replace(tmp, POS_CACHE)
    return save


# ---------------------------------------------------------------------------------------------
# BERT sentence encoder
# ---------------------------------------------------------------------------------------------
class BertEncoder:
    def __init__(self, path: str, max_len: int = 128, batch: int = 32):
        import torch
        from transformers import AutoModel, AutoTokenizer
        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(path)
        self.dev = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = AutoModel.from_pretrained(path).to(self.dev).eval()
        self.max_len, self.batch = max_len, batch
        probe = self.tok.tokenize("भारत की राजधानी नई दिल्ली है।")
        if "[UNK]" in probe:
            sys.exit(f"BERT tokenizer at {path} maps Hindi to [UNK]: {probe}")

    def encode(self, sentences: list[str]) -> list[np.ndarray]:
        torch, out = self.torch, []
        for i in range(0, len(sentences), self.batch):
            enc = self.tok(sentences[i:i + self.batch], padding=True, truncation=True,
                           max_length=self.max_len, return_tensors="pt",
                           return_special_tokens_mask=True)
            special = enc.pop("special_tokens_mask")
            enc = {k: v.to(self.dev) for k, v in enc.items()}
            with torch.no_grad():
                hidden = self.model(**enc).last_hidden_state.float()
            keep = (enc["attention_mask"].bool() & ~special.to(self.dev).bool()).unsqueeze(-1)
            summed = (hidden * keep).sum(1)
            count = keep.sum(1).clamp(min=1)
            out.extend((summed / count).cpu().numpy())
        return out


def best_static_by_validation() -> str:
    """Trained static model with the best mean validation ROUGE-1 F over both corpora."""
    val = [r for r in rows() if r["split"] == "val" and r["model"] in STATIC]
    means = {}
    for m in STATIC:
        got = {r["dataset"]: r["rouge1_f"] for r in val if r["model"] == m}
        if len(got) == 2:
            means[m] = sum(got.values()) / 2
    if len(means) < len(STATIC):
        sys.exit(f"--bv-kv auto needs validation rows for all static models; have {sorted(means)}")
    return max(means, key=means.get)


# ---------------------------------------------------------------------------------------------
def bootstrap(per_doc: np.ndarray, base: np.ndarray | None):
    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, len(per_doc), size=(BOOT, len(per_doc)))
    ci = np.percentile(per_doc[idx].mean(1), [2.5, 97.5])
    if base is None:
        return ci, None, None
    diff = per_doc - base
    gci = np.percentile(diff[idx].mean(1), [2.5, 97.5])
    return ci, float(diff.mean()), gci


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", required=True, help="backend name as used in results (e.g. hi-w2v-sg-stem-300)")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--kv")
    src.add_argument("--bert")
    ap.add_argument("--bv-kv", help="static vectors for the big-vector expansion of the BERT backend, or 'auto'")
    ap.add_argument("--dataset", required=True, choices=["XL-Sum", "ILSUM 2.0"])
    ap.add_argument("--split", required=True, choices=["val", "test"])
    ap.add_argument("--limit", type=int, default=N_DOCS, help="smoke tests only; results are not recorded unless 200")
    args = ap.parse_args()

    record = args.limit == N_DOCS
    if not record and args.split == "test":
        sys.exit("smoke tests run on validation sets only; test sets are scored once, in full")
    done = {(r["model"], r["dataset"], r["split"]) for r in rows()}
    if record and (args.name, args.dataset, args.split) in done:
        sys.exit(f"{args.name} already has a {args.dataset} {args.split} result; refusing to run it again")
    if record and args.split == "test":
        missing = [d for d in ("XL-Sum", "ILSUM 2.0") if (args.name, d, "val") not in done]
        if missing:
            sys.exit(f"test run refused: {args.name} has no validation result for {missing}")

    sys.path.insert(0, os.path.join(H, "code"))
    from hinexsum import HindiSummarizer, load_embeddings, rouge_scores, average_scores
    save_pos = install_pos_cache()

    bv_name = None
    if args.kv:
        model = load_embeddings(args.kv)
        encoder = None
    else:
        if not args.bv_kv:
            sys.exit("--bert needs --bv-kv PATH or --bv-kv auto")
        bv_path = args.bv_kv
        if bv_path == "auto":
            bv_name = best_static_by_validation()
            bv_path = os.path.join(EXP, "models", bv_name + ".kv")
        else:
            bv_name = os.path.splitext(os.path.basename(bv_path))[0]
        log(f"big-vector expansion from {bv_name}")
        model = load_embeddings(bv_path)
        encoder = BertEncoder(args.bert)

    _, _, selection = SETS[(args.dataset, args.split)]
    summ = HindiSummarizer(model, use_pos=True, features=None, weights=None, selection=selection)
    docs = load_docs(args.dataset, args.split, args.limit)
    log(f"{args.name} | {args.dataset} {args.split} | {len(docs)} docs | {selection}")

    per_doc, t0 = [], time.time()
    try:
        for i, (text, ref) in enumerate(docs, 1):
            if encoder is None:
                sentences, token_lists, _, scores = summ.score_sentences(text)
            else:
                from hinexsum import split_sentences, preprocess_sentence
                from hinexsum.ranking import rank_sentences
                sentences = split_sentences(text)
                token_lists = [preprocess_sentence(s) for s in sentences]
                vecs = encoder.encode(sentences) if sentences else []
                scores = rank_sentences(sentences, token_lists, vecs, use_pos=True, weights=None)
            if sentences:
                idx = summ.select_indices(sentences, token_lists, scores, N_SENT, selection)
                summary = " ".join(sentences[j] for j in sorted(idx))
            else:
                summary = ""
            per_doc.append(rouge_scores(summary, [ref]))
            if i % 25 == 0:
                log(f"  {i}/{len(docs)}  ({(time.time() - t0) / i:.1f}s/doc)")
                save_pos()
    finally:
        save_pos()

    avg = average_scores(per_doc)
    r1 = np.array([s["rouge-1"]["f"] for s in per_doc])
    base = None
    if args.name != BASELINE and os.path.exists(perdoc_path(BASELINE, args.dataset, args.split)):
        with open(perdoc_path(BASELINE, args.dataset, args.split), encoding="utf-8") as fh:
            b = [json.loads(l)["rouge-1"]["f"] for l in fh]
        if len(b) == len(r1):
            base = np.array(b)
    ci, gap, gci = bootstrap(r1, base)

    row = {"model": args.name, "dataset": args.dataset, "split": args.split, "n_docs": len(docs),
           "budget": N_SENT, "selection": selection, "pos": True, "weights": "equal",
           **{KEYS[m]: round(avg[m]["f"], 4) for m in METRICS},
           "rouge1_f_ci95": [round(float(x), 4) for x in ci],
           "backend": args.kv or args.bert, "big_vector_model": bv_name,
           "date": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")}
    if gap is not None:
        row["r1_gap_vs_baseline"] = round(gap, 4)
        row["r1_gap_ci95"] = [round(float(x), 4) for x in gci]
    print(json.dumps(row, ensure_ascii=False))
    if not record:
        log(f"smoke test ({args.limit} docs): not recorded")
        return
    os.makedirs(PERDOC, exist_ok=True)
    with open(perdoc_path(args.name, args.dataset, args.split), "w", encoding="utf-8") as fh:
        for s in per_doc:
            fh.write(json.dumps(s) + "\n")
    with open(RESULTS, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    log(f"recorded: R1-F {row['rouge1_f']}" + (f", gap vs baseline {gap:+.4f}" if gap is not None else ""))


if __name__ == "__main__":
    main()
