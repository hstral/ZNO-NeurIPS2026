#!/usr/bin/env bash
# Prepare the datasets of the paper under $DATA_DIR (default: data/), in the
# layout that `python -m experiments.build` expects.
#
#   bash scripts/prepare_data.sh [step ...]      # default: all steps, in this order
#
#   controlled     resonant ARMA, sixth-order IIR cascade, nonlinear NARX
#                  (1024/256/256 trajectories of length 2048)
#   difficulty     the five near-unit-circle radius bins of the resonant ARMA task
#   extrapolation  the same generators with test sequences of length 4096 and 8192
#   public         Silverbox, Wiener-Hammerstein, EMPS, Cascaded Tanks, CED
#                  (downloaded through nonlinear-benchmarks; needs network access)
#   views          test-free copies of the controlled and public data for the
#                  validation-only selection runs
#   s5-views       the same arrays exported for the official S5 core
#
# The generators are seeded, so the synthetic datasets are those of the paper.
# Set PYTHON to choose the interpreter (default: python).
set -euo pipefail
cd "$(dirname "$0")/.."

DATA="${DATA_DIR:-data}"
PY="${PYTHON:-python}"
CONTROLLED="resonant_arma_near_unit_circle_v4 high_order_iir_cascade_v5 narx_softplus"
DIFFICULTY="resonant_arma_lm_r0900_0930_v1 resonant_arma_lm_r0930_0960_v1 resonant_arma_lm_r0960_0980_v1
            resonant_arma_lm_r0980_0990_v1 resonant_arma_lm_r0990_0995_v1"
PUBLIC="silverbox_public_v1 wiener_hammer_public_v1 emps_public_v1 cascaded_tanks_public_v1 ced_public_v1"
SPLITS="--sequence-length 2048 --train-size 1024 --val-size 256 --test-size 256"

generate() {  # generate <output dir> <cases> [extra args]
  local out="$1" cases="$2"; shift 2
  for case in $cases; do
    "$PY" benchmark_discrete/generate_suite.py --suite "$case" --output-dir "$out" $SPLITS --overwrite "$@"
  done
}

step_controlled() { generate "$DATA/controlled" "$CONTROLLED"; }

step_difficulty() { generate "$DATA/difficulty" "$DIFFICULTY"; }

step_extrapolation() {
  local long="--train-sequence-length 2048 --val-sequence-length 2048 --test-sequence-length"
  for length in 4096 8192; do
    generate "$DATA/extrapolation/T$length" "$CONTROLLED" $long "$length"
    generate "$DATA/difficulty_extrapolation/T$length" "$DIFFICULTY" $long "$length"
  done
}

step_public() {
  for case in $PUBLIC; do
    # Window settings of each benchmark: PUBLIC_DATA_BUILD in experiments/protocols.py.
    args=$("$PY" -c "from experiments.protocols import PUBLIC_DATA_BUILD as B; print(' '.join(B['$case']))")
    "$PY" benchmark_discrete/build_external_benchmark.py --suite "$case" --protocol public \
      --output-dir "$DATA/public" $args --overwrite --auto-install
  done
}

step_views() {
  "$PY" -m experiments.build_selection_views --source-root "$DATA/controlled" \
    --output-root "$DATA/selection_views/controlled"
  "$PY" -m experiments.build_selection_views --source-root "$DATA/public" \
    --output-root "$DATA/selection_views/public" --cases public
}

step_s5_views() {
  "$PY" -m s5_baseline.export_views --data-root "$DATA/controlled" --output-root "$DATA/s5_views/controlled" \
    --cases $CONTROLLED
  "$PY" -m s5_baseline.export_views --data-root "$DATA/public" --output-root "$DATA/s5_views/public" \
    --cases $PUBLIC
}

steps=("$@")
[ ${#steps[@]} -eq 0 ] && steps=(controlled difficulty extrapolation public views s5-views)
for step in "${steps[@]}"; do
  echo "== $step"
  "step_${step//-/_}"
done
