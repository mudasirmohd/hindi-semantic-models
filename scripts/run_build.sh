#!/usr/bin/env bash
# Wait for the IndicCorp download, then build both tokenized corpora.
set -e
cd "$(dirname "$0")/.."
until grep -q INDIC_DONE logs/download.log; do sleep 20; done
H=${HINEXSUM_HOME:-$HOME/projects/hindi-summarizer}
.venv/bin/python -I scripts/build_corpus.py \
  --inputs 'data/wiki_extracted/**/wiki_*' data/raw/indiccorp_v2_hi-1_first8GB.txt \
  --eval-jsonl $H/xlsum_hindi_val.jsonl $H/xlsum_hindi_test.jsonl $H/ilsum_hindi.jsonl \
  --out-dir data/corpus --workers 28
echo BUILD_DONE
