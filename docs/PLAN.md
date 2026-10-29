# Hindi distributional semantic models: plan (written 2026-10-29, before any model is trained)

Purpose: close registered thesis objective 2 ("Develop distributional semantic models for the Hindi
language by utilizing various semantic algorithms such as Word2Vec and BERT") and test the models as
HinExSum backends (objective 3).

## Corpus
- Hindi Wikipedia dump (hiwiki-latest, Oct 2026, CC BY-SA) + first 8 GB of IndicCorp v2 `hi-1.txt`
  (AI4Bharat, CC-0).
- Processed with HinExSum's own preprocessing (NFC, danda-aware split, Devanagari tokenizer).
  Two views: surface tokens and HinExSum-stemmed tokens. Stopwords kept.
- Every sentence that also occurs in an evaluation document (XL-Sum Hindi val+test, all ILSUM Hindi
  splits; text, summary, title) is removed. Sample check: 42 overlaps in 3 MB of IndicCorp.

## Models (static)
Word2Vec skip-gram and fastText skip-gram (subwords 3-6), each on surface and stemmed corpus:
`w2v-sg-surface`, `w2v-sg-stem`, `ft-sg-surface`, `ft-sg-stem`. dim 300, window 5, min_count 5,
max vocab 1M, negative 10, 5 epochs, seed 42. Baseline: Facebook `cc.hi.300.bin` (used in Ch 5).

## Evaluation (fixed now)
1. Intrinsic: Hindi-WS (Akhtar et al. 2017, 235 pairs; Spearman rho, OOV pairs reported) and the
   fastText Hindi analogy set (Grave et al. 2018; accuracy over questions fully in vocabulary, top 300k
   words, coverage reported).
2. Lookup coverage: fraction of HinExSum content tokens (stemmed, stopwords removed) of XL-Sum val
   documents that the model has a vector for (fastText: in-vocabulary, not subword-composed).
3. Extrinsic, model SELECTION on validation only: HinExSum, published equal-weight configuration,
   3-sentence budget (`--fixed --max-sentences 3`), XL-Sum val first 200 and ILSUM val (run_ilsum.py
   protocol), ROUGE-1/2/L F with paired bootstrap (1000, seed 42), against the cc.hi.300 backend.
4. Test: every model run ONCE on the same test sets as Ch 5 (XL-Sum test first 200; ILSUM test),
   reported in full whatever the result. No model or setting is changed after seeing test scores.

## Expectation stated in advance
Chapter 5 found sentence position is the only contributing feature on these lead-biased benchmarks, so
a better embedding backend is NOT expected to move ROUGE much. A null extrinsic result is a valid outcome
and will be reported as such; the models' value is then shown by the intrinsic results and coverage.
