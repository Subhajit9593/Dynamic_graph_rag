#!/usr/bin/env bash
# Full experiment pipeline after extraction (PLAN.md §9). Each step is resumable / idempotent.
#   1. streaming experiment with LLM judge     -> data/db/{dynamic,static}.sqlite + results/stream_experiment.json
#   2. streaming experiment without judge      -> results/stream_experiment_nojudge.json (pure graph-work timing)
#   3. retrieval ablations (no LLM)            -> results/ablation_retrieval.json
#   4. QA evaluation, 5 systems x 2,556 qs     -> results/eval/<system>.jsonl + _metrics.json
set -euo pipefail
cd "$(dirname "$0")/.."
source ~/ml-env/bin/activate
mkdir -p data/logs
log() { echo "$(date +%T) === $*"; }

log "stream (judge)";    dgrag stream --batches 10            > data/logs/stream.log 2>&1
log "stream (no judge)"; dgrag stream --batches 10 --no-judge > data/logs/stream_nojudge.log 2>&1
log "ablations"
python -c "
from dgrag.config import load_config, setup_logging; setup_logging('WARNING')
from dgrag.eval.ablation import run_ablations; run_ablations(load_config())" > data/logs/ablation.log 2>&1
log "eval"; dgrag eval dynamic static vector vector_rerank none > data/logs/eval_full.log 2>&1
log "done"
