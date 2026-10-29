#!/usr/bin/env bash
# Baseline intrinsic eval, then train + evaluate the four static models one at a time.
# Resumable: a model whose .json exists is skipped.
set -u
cd "$(dirname "$0")/.."
PY=.venv/bin/python
OUT=results/intrinsic.jsonl
mkdir -p models results
if ! grep -q '"model": "cc.hi.300 (Facebook)"' $OUT 2>/dev/null; then
  echo "[$(date +%T)] baseline cc.hi.300"
  # Word vectors only (.vec): the 6.9 GB .bin with subword buckets exceeds WSL memory when loaded.
  if [ ! -f data/baseline/cc.hi.300.kv ]; then
    mkdir -p data/baseline
    curl -sSL -o data/baseline/cc.hi.300.vec.gz https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.hi.300.vec.gz
    $PY -I -c "from gensim.models import KeyedVectors as K; K.load_word2vec_format('data/baseline/cc.hi.300.vec.gz').save('data/baseline/cc.hi.300.kv')"
  fi
  $PY -I scripts/eval_intrinsic.py --model data/baseline/cc.hi.300.kv --name "cc.hi.300 (Facebook)" --out $OUT
fi
for spec in "word2vec surface" "word2vec stem" "fasttext surface" "fasttext stem"; do
  set -- $spec; algo=$1; view=$2
  short=$([ $algo = word2vec ] && echo w2v || echo ft)
  name=hi-$short-sg-$view-300
  if [ ! -f models/$name.json ]; then
    echo "[$(date +%T)] train $name"
    $PY -I scripts/train.py --algo $algo --arch sg --corpus data/corpus/corpus_$view.txt --out models/$name --workers 30 || { echo "FAILED $name"; continue; }
  fi
  if ! grep -q "\"model\": \"$name\"" $OUT 2>/dev/null; then
    stemflag=$([ $view = stem ] && echo --stem || true)
    $PY -I scripts/eval_intrinsic.py --model models/$name.kv --name $name $stemflag --out $OUT
  fi
done
echo "[$(date +%T)] STATIC_DONE"
