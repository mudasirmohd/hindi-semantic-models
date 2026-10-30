#!/usr/bin/env bash
# When the extension queue finishes: paired ROUGE gaps, BLEU/chrF++, paper rebuild.
cd "$(dirname "$0")/.."
until [ -f EXTENSION_DONE ]; do sleep 60; done
PY=~/projects/hindi-summarizer/.venv/bin/python
$PY scripts/paired_gaps.py > logs/paired_gaps.log 2>&1
$PY -I scripts/text_metrics.py > logs/text_metrics.log 2>&1
~/projects/paper-hindi-semantic-models/build.sh > logs/paper_build.log 2>&1
touch AFTER_DONE
