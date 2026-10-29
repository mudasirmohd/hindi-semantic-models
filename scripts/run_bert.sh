#!/usr/bin/env bash
# After the punctuated text corpus exists: tokenizer, a 300-step throughput probe, then pre-training.
set -eu
cd "$(dirname "$0")/.."
PY=.venv-bert/bin/python
until [ -f data/corpus/stats_text.json ]; do sleep 30; done
if [ ! -f models/bert/tokenizer-hi-wp32k/tokenizer.json ]; then
  rm -rf models/bert/tokenizer-hi-wp32k
  $PY -I scripts/bert_tokenizer.py --corpus data/corpus/corpus_text.txt --out models/bert/tokenizer-hi-wp32k --max-lines 10000000
fi
STEPS=${STEPS:-100000}
$PY -I scripts/pretrain_bert.py --corpus data/corpus/corpus_text.txt --tokenizer models/bert/tokenizer-hi-wp32k \
  --out models/bert/hi-bert-medium --size medium --steps $STEPS --batch 64 --grad-accum 4 --workers 6
echo BERT_DONE
