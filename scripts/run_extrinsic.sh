#!/usr/bin/env bash
# Extrinsic evaluation queue (docs/PLAN.md). Waits for all training to finish, then scores every
# backend on both validation sets, and only then on both test sets, once each. Resumable: a
# (backend, set) pair already in results/extrinsic.jsonl is skipped.
set -uo pipefail
cd "$(dirname "$0")/.."
PY=~/projects/hindi-summarizer/.venv/bin/python
BERT=models/bert/hi-bert-medium/final
# Ch 5 used the full fastText binary (subword vectors); the baseline must reproduce Ch 5's numbers.
BASE_KV=~/projects/hindi-summarizer/cc.hi.300.bin

until [ -f STATIC_DONE ] && [ -f $BERT/config.json ] && ! pgrep -f "scripts/(train|pretrain_bert).py" >/dev/null; do
  sleep 300
done
echo "[$(date +%T)] training finished; starting extrinsic evaluation"

have() { grep -qF "\"model\": \"$1\", \"dataset\": \"$2\", \"split\": \"$3\"" results/extrinsic.jsonl 2>/dev/null; }
run() {  # name dataset split backend-args...
  local name=$1 ds=$2 split=$3; shift 3
  have "$name" "$ds" "$split" && { echo "skip $name $ds $split"; return; }
  $PY -I scripts/eval_extrinsic.py --name "$name" --dataset "$ds" --split "$split" "$@" || echo "FAILED $name $ds $split"
}
static() {  # split
  run "cc.hi.300 (Facebook)" "$2" "$1" --kv $BASE_KV
  for m in hi-w2v-sg-surface-300 hi-w2v-sg-stem-300 hi-ft-sg-surface-300 hi-ft-sg-stem-300; do
    run $m "$2" "$1" --kv models/$m.kv
  done
}
for split in val test; do
  for ds in "XL-Sum" "ILSUM 2.0"; do static $split "$ds"; done
  for ds in "XL-Sum" "ILSUM 2.0"; do run hi-bert-medium "$ds" $split --bert $BERT --bv-kv auto; done
done
touch EXTRINSIC_DONE
echo "[$(date +%T)] extrinsic evaluation done"
