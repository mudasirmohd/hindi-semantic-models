# Hindi Semantic Models

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23794581.svg)](https://doi.org/10.5281/zenodo.23794581)

Distributional semantic models for Hindi: four static embedding models (Word2Vec and fastText,
skip-gram, 300-d) and a compact Hindi BERT pre-trained from scratch. All were trained on a cleaned
Hindi corpus from which every sentence of the XL-Sum Hindi and ILSUM Hindi evaluation data was removed.

The models were built for [HinExSum](https://github.com/mudasirmohd/HinExSum)
([Software Impacts 30 (2026) 100872](https://doi.org/10.1016/j.simpa.2026.100872)), and are described
in Chapter 6 of the PhD thesis *Semantic Models Summarization for Hindi Language Text* (S. A. Khan,
University of Kashmir).

| Model | Algorithm | Training text | Vocabulary | Download (release assets) |
|---|---|---|---|---|
| `hi-w2v-sg-surface-300` | Word2Vec skip-gram | surface tokens | 563,147 | `hi-w2v-sg-surface-300.bin` |
| `hi-w2v-sg-stem-300` | Word2Vec skip-gram | HinExSum-stemmed tokens | 466,271 | `hi-w2v-sg-stem-300.bin` |
| `hi-ft-sg-surface-300` | fastText skip-gram, subwords 3-6 | surface tokens | 563,147 | `hi-ft-sg-surface-300.bin.part0/1` |
| `hi-ft-sg-stem-300` | fastText skip-gram, subwords 3-6 | HinExSum-stemmed tokens | 466,271 | `hi-ft-sg-stem-300.bin.part0/1` |
| `hi-bert-medium` | BERT, MLM, 8 layers, 512 hidden, 42.16 M parameters | punctuated text | 32,000 WordPiece | `hi-bert-medium.tar.gz` |

See [MODEL_CARD.md](MODEL_CARD.md) for training data, preprocessing, hyperparameters, evaluation and
limitations. Checksums are in `SHA256SUMS` on the release page. The release is archived on Zenodo:
[doi:10.5281/zenodo.23794581](https://doi.org/10.5281/zenodo.23794581).

## Usage

```python
# Word2Vec (word2vec binary format)
from gensim.models import KeyedVectors
kv = KeyedVectors.load_word2vec_format("hi-w2v-sg-surface-300.bin", binary=True)
kv.most_similar("सरकार", topn=5)
```

```bash
# fastText: the .bin exceeds GitHub's 2 GB asset limit and is split in two parts
cat hi-ft-sg-surface-300.bin.part0 hi-ft-sg-surface-300.bin.part1 > hi-ft-sg-surface-300.bin
sha256sum -c SHA256SUMS --ignore-missing
```

```python
from gensim.models.fasttext import load_facebook_vectors   # or: fasttext.load_model(...)
ft = load_facebook_vectors("hi-ft-sg-surface-300.bin")      # subword vectors for unseen words
```

```python
# Hindi BERT (tar -xzf hi-bert-medium.tar.gz)
from transformers import AutoTokenizer, AutoModelForMaskedLM
tok = AutoTokenizer.from_pretrained("hi-bert-medium")
bert = AutoModelForMaskedLM.from_pretrained("hi-bert-medium")
```

**Stemmed models expect stemmed input.** Query `*-stem-*` models with words passed through HinExSum's
preprocessing, exactly as they were trained:

```python
from hinexsum.preprocessing import preprocess_sentence
tokens = preprocess_sentence("लोगों ने सरकार से संबंधी सवाल पूछे")   # NFC, tokenise, drop stopwords, stem
```

## Results

Intrinsic evaluation (Hindi-WS word similarity, Akhtar et al. 2017; fastText Hindi analogies within the
300k most frequent words; coverage of the tokens HinExSum looks up on 200 XL-Sum validation documents):

| Model | Word sim. (Spearman ρ) | Analogy acc. | Coverage |
|---|---|---|---|
| cc.hi.300 (Grave et al. 2018, baseline) | 0.551 | 0.268 | 97.2% |
| hi-w2v-sg-surface-300 | 0.581 | 0.268 | 95.9% |
| **hi-w2v-sg-stem-300** | **0.623** | 0.207 | **99.4%** |
| hi-ft-sg-surface-300 | 0.513 | 0.229 | 95.9% |
| hi-ft-sg-stem-300 | 0.562 | 0.148 | 99.4% |

Hindi BERT: 100,000 steps, held-out masked-LM loss 1.881 (perplexity 6.6).

As the semantic backend of HinExSum (published equal-weight configuration, 3 sentences, 200 documents
per set; ROUGE-1 F on the test sets):

| Backend | XL-Sum | ILSUM 2.0 |
|---|---|---|
| cc.hi.300 (baseline) | 0.186 | 0.256 |
| hi-w2v-sg-surface-300 | 0.184 | 0.258 |
| hi-w2v-sg-stem-300 | 0.187 | 0.259 |
| hi-ft-sg-surface-300 | 0.185 | 0.259 |
| hi-ft-sg-stem-300 | 0.187 | 0.257 |
| hi-bert-medium | 0.186 | 0.256 |

No backend differs significantly from the baseline (all 20 paired-bootstrap 95% intervals, validation
and test, contain zero; the largest difference is 0.006). This was the outcome anticipated in the
protocol registered before training ([docs/PLAN.md](docs/PLAN.md)): on these lead-biased news
benchmarks sentence position dominates extractive selection. Full results, including validation sets
and confidence intervals, are in `results/`.

## Reproducing

The scripts assume HinExSum is checked out at `~/projects/hindi-summarizer` (override with
`HINEXSUM_HOME`) and installed in the Python environment.

| Step | Script |
|---|---|
| Corpus (hiwiki + IndicCorp v2 hi, eval-overlap removal) | `scripts/run_build.sh`, `scripts/build_corpus.py` |
| Static models + intrinsic evaluation | `scripts/run_static.sh`, `scripts/train.py`, `scripts/eval_intrinsic.py` |
| BERT tokenizer and pre-training | `scripts/run_bert.sh`, `scripts/bert_tokenizer.py`, `scripts/pretrain_bert.py` |
| HinExSum with each backend | `scripts/run_extrinsic.sh`, `scripts/eval_extrinsic.py`, `scripts/paired_gaps.py` |

## Licence

`results/summaries/` contains system summaries, which are sentences extracted from XL-Sum Hindi
(CC BY-NC-SA 4.0) and ILSUM 2.0 Hindi articles; that text remains under the licences of those datasets.

Code: MIT ([LICENSE](LICENSE)). Models: CC BY-SA 4.0 ([LICENSE-MODELS.md](LICENSE-MODELS.md)), because
part of the training text is Hindi Wikipedia (CC BY-SA); the IndicCorp v2 portion is CC0.

## Citation

See [CITATION.cff](CITATION.cff). Please also cite HinExSum:

> S. A. Khan, M. Mohd, N. Sheikh, M. A. Wani, A. H. Wani, H. A. Khanday and N. A. Wani, "HinExSum: Hindi
> extractive text summarization using word embeddings," *Software Impacts* 30 (2026) 100872.
> https://doi.org/10.1016/j.simpa.2026.100872
