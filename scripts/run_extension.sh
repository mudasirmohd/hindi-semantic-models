#!/usr/bin/env bash
# Paper extension (2026-10-30): (1) regenerate the summaries of all 24 recorded runs (deterministic; each
# must reproduce its recorded per-document ROUGE exactly) so BLEU and chrF++ can be scored; (2) add three
# published external backends under the same protocol: validation first, each test set once.
set -uo pipefail
cd "$(dirname "$0")/.."
PY=~/projects/hindi-summarizer/.venv/bin/python
BASE_KV=~/projects/hindi-summarizer/cc.hi.300.bin
STATIC="hi-w2v-sg-surface-300 hi-w2v-sg-stem-300 hi-ft-sg-surface-300 hi-ft-sg-stem-300"
SUMM=results/summaries
safe() { echo "$1" | sed 's/ /_/g; s/[()]//g; s/\./_/g'; }
have() { grep -qF "\"model\": \"$1\", \"dataset\": \"$2\", \"split\": \"$3\"" results/extrinsic.jsonl 2>/dev/null; }
regen() {  # name ds split args...
  local name=$1 ds=$2 split=$3; shift 3
  [ -f "$SUMM/$(safe "$name")__$(safe "$ds")__$split.jsonl" ] && return
  $PY -I scripts/eval_extrinsic.py --name "$name" --dataset "$ds" --split "$split" --regenerate "$@" || echo "FAILED regen $name $ds $split"
}
run() {
  local name=$1 ds=$2 split=$3; shift 3
  have "$name" "$ds" "$split" && return
  $PY -I scripts/eval_extrinsic.py --name "$name" --dataset "$ds" --split "$split" "$@" || echo "FAILED $name $ds $split"
}
for split in val test; do for ds in "XL-Sum" "ILSUM 2.0"; do
  for m in $STATIC; do regen $m "$ds" $split --kv models/$m.kv; done
  regen hi-bert-medium "$ds" $split --bert models/bert/hi-bert-medium/final --bv-kv auto
done; done
for split in val test; do for ds in "XL-Sum" "ILSUM 2.0"; do
  regen "cc.hi.300 (Facebook)" "$ds" $split --kv $BASE_KV
done; done
echo "[$(date +%T)] regeneration done"

until grep -q WIKI_OK logs/download_external.log && [ "$(grep -c HF_OK logs/download_external.log)" = 2 ]; do sleep 30; done
for split in val test; do for ds in "XL-Sum" "ILSUM 2.0"; do
  run "wiki.hi (Facebook)" "$ds" $split --kv data/external/wiki.hi.vec
  run muril-base-cased "$ds" $split --bert models/external/muril-base-cased --bv-kv auto
  run IndicBERTv2-MLM-only "$ds" $split --bert models/external/IndicBERTv2-MLM-only --bv-kv auto
done; done
touch EXTENSION_DONE
echo "[$(date +%T)] extension done"
