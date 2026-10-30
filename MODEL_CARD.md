# Model card: Hindi Semantic Models

Covers `hi-w2v-sg-surface-300`, `hi-w2v-sg-stem-300`, `hi-ft-sg-surface-300`, `hi-ft-sg-stem-300` and
`hi-bert-medium`.

## Developers
Showket Ahmad Khan, Mudasir Mohd, Hilal Ahmad Khanday. Department of Computer Sciences, South Campus,
University of Kashmir. Released October 2026.

## Intended use
Hindi word similarity and nearest-neighbour lookup, semantic features for extractive summarization
(HinExSum backends), and sentence representations for Hindi NLP research. The stemmed models are
intended for pipelines that apply HinExSum's preprocessing.

## Training data
- Hindi Wikipedia, dump of October 2026 (CC BY-SA), markup removed with WikiExtractor.
- First 8 GB of the Hindi portion of IndicCorp v2 (AI4Bharat, CC0).
- Processed with HinExSum's preprocessing: Unicode NFC, danda-aware sentence splitting, Devanagari
  tokenizer; sentences with fewer than 3 tokens dropped.
- **Evaluation decontamination:** every sentence whose token sequence matches a sentence of XL-Sum Hindi
  validation or test, or of any ILSUM Hindi split (text, summary and title fields; 1,048,682 distinct
  sentences) was removed: 120,612 corpus sentences.
- Result: 35,753,033 sentences, 657,022,691 tokens, in three views of the same sentences:
  surface (punctuation removed), stemmed (HinExSum light stemmer), text (punctuation kept, used for BERT).
  Stopwords are kept in all views.

## Static models
gensim 4.4.0, skip-gram, 300 dimensions, window 5, negative sampling 10, 5 epochs, min count 5,
vocabulary capped at 1,000,000, seed 42. fastText: character n-grams 3-6, 1,000,000 buckets.

| Model | View | Vocabulary | Format |
|---|---|---|---|
| hi-w2v-sg-surface-300 | surface | 563,147 | word2vec binary |
| hi-w2v-sg-stem-300 | stemmed | 466,271 | word2vec binary |
| hi-ft-sg-surface-300 | surface | 563,147 | Facebook fastText .bin (split in 2 parts) |
| hi-ft-sg-stem-300 | stemmed | 466,271 | Facebook fastText .bin (split in 2 parts) |

## hi-bert-medium
- Architecture: BERT, 8 layers, hidden 512, 8 heads, FFN 2,048; 42,162,944 parameters.
- Tokenizer: WordPiece, 32,000 units, trained on 10 M sentences of the text view; **lower-casing and accent
  stripping disabled** (accent stripping deletes Devanagari vowel signs).
- Pre-training: masked LM (15%), sequences of 128 tokens packed from the text view (6,109,615 blocks),
  effective batch 256 (64 x 4 accumulation), AdamW, peak LR 5e-4, 10,000 warm-up steps then linear decay,
  weight decay 0.01, bf16, 100,000 steps (about 4.2 epochs) on one RTX 5070 Laptop GPU (8 GB).
- Held-out masked-LM loss on 2,000 blocks: 1.881 (perplexity 6.6).

## Evaluation
Protocol registered before training (`docs/PLAN.md`).

| Model | Word sim. ρ (235 pairs) | Analogy acc. | HinExSum coverage |
|---|---|---|---|
| cc.hi.300 (baseline) | 0.551 | 0.268 | 97.2% |
| hi-w2v-sg-surface-300 | 0.581 | 0.268 | 95.9% |
| hi-w2v-sg-stem-300 | 0.623 | 0.207 | 99.4% |
| hi-ft-sg-surface-300 | 0.513 | 0.229 | 95.9% |
| hi-ft-sg-stem-300 | 0.562 | 0.148 | 99.4% |

As HinExSum backends (published equal-weight configuration, 3 sentences, XL-Sum and ILSUM 2.0 Hindi,
200 documents per set), no model differs significantly from cc.hi.300 in ROUGE-1 F (largest paired
difference 0.006; all 95% bootstrap intervals contain zero). For the BERT backend, sentence vectors are
the mean of final-layer token states and the big-vector expansion uses hi-w2v-sg-stem-300, the best
static model on validation. Full numbers: `results/`.

## Limitations
- IndicCorp v2 is used in part (8 GB), and BERT is smaller than BERT-base and trained on 128-token
  sequences, to fit a single laptop GPU.
- The word similarity set is small (235 pairs); small differences in ρ are not reliable.
- Evaluation decontamination removes exact sentence matches only; near-duplicates may remain.
- Stemming merges inflected forms, which lowers analogy accuracy; use surface models where inflection matters.
- The corpora are news- and encyclopedia-heavy and carry their topical and social biases.

## Licence
CC BY-SA 4.0 (models); MIT (code).
