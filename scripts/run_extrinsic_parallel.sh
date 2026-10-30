#!/usr/bin/env bash
# Same protocol as run_extrinsic.sh, in two lanes to fit the deadline:
#   lane A: cc.hi.300 baseline (needs ~9 GB, so it runs alone in its lane)
#   lane B: static models on validation -> BERT on validation (big vectors from the best static
#           model on validation) -> static models on test -> BERT on test.
# Every test run still happens once, after that backend's validation runs.
set -uo pipefail
cd "$(dirname "$0")/.."
PY=~/projects/hindi-summarizer/.venv/bin/python
BERT=models/bert/hi-bert-medium/final
BASE_KV=~/projects/hindi-summarizer/cc.hi.300.bin
STATIC="hi-w2v-sg-surface-300 hi-w2v-sg-stem-300 hi-ft-sg-surface-300 hi-ft-sg-stem-300"
have() { grep -qF "\"model\": \"$1\", \"dataset\": \"$2\", \"split\": \"$3\"" results/extrinsic.jsonl 2>/dev/null; }
run() {
  local name=$1 ds=$2 split=$3; shift 3
  have "$name" "$ds" "$split" && { echo "skip $name $ds $split"; return; }
  $PY -I scripts/eval_extrinsic.py --name "$name" --dataset "$ds" --split "$split" "$@" || echo "FAILED $name $ds $split"
}
laneA() {
  while pgrep -f "eval_extrinsic.py --name cc.hi.300" >/dev/null; do sleep 10; done  # run already in flight
  for split in val test; do for ds in "XL-Sum" "ILSUM 2.0"; do
    run "cc.hi.300 (Facebook)" "$ds" $split --kv $BASE_KV; done; done
}
laneB() {
  for ds in "XL-Sum" "ILSUM 2.0"; do for m in $STATIC; do run $m "$ds" val --kv models/$m.kv; done; done
  for ds in "XL-Sum" "ILSUM 2.0"; do run hi-bert-medium "$ds" val --bert $BERT --bv-kv auto; done
  for ds in "XL-Sum" "ILSUM 2.0"; do for m in $STATIC; do run $m "$ds" test --kv models/$m.kv; done; done
  for ds in "XL-Sum" "ILSUM 2.0"; do run hi-bert-medium "$ds" test --bert $BERT --bv-kv auto; done
}
laneA > logs/extrinsic_laneA.log 2>&1 &
laneB > logs/extrinsic_laneB.log 2>&1 &
wait
touch EXTRINSIC_DONE
echo "[$(date +%T)] extrinsic evaluation done"
