#!/bin/bash
# Runs coli-swift on the oracle fixtures' inputs (the tiny synthetic models and Qwen3-0.6B)
# and reports, per case, whether its greedy ids match upstream colibri's and where they
# first diverge. See scripts/oracle/README.md.
#
#   scripts/oracle/compare.sh                         # cpu backend, all cases
#   scripts/oracle/compare.sh --backend metal
#   scripts/oracle/compare.sh --cases olmoe_q8_0,qwen3moe_q4_K --tiny-only
#   scripts/oracle/compare.sh --reference default     # upstream int8 path vs coli-swift --int8
#   scripts/oracle/compare.sh -- --experts stream     # extra coli-swift arguments after --
#
# Options (everything else is passed to compare.py):
#   --no-build      use the existing release build
#   --regen         rewrite the tiny models even if they are present
#
# Environment: ORACLE_DIR (default ${TMPDIR:-/tmp}/colibri-oracle; the tiny models
# go to $ORACLE_DIR/tiny), SCRATCH (SwiftPM scratch path, default .build-oracle).
set -euo pipefail

REPO=$(cd "$(dirname "$0")/../.." && pwd)
ORACLE_DIR=${ORACLE_DIR:-${TMPDIR:-/tmp}/colibri-oracle}
MODELS=$ORACLE_DIR/tiny
SCRATCH=${SCRATCH:-$REPO/.build-oracle}

build=1 regen=0 pass=()
while [[ $# -gt 0 ]]; do
    case $1 in
        --no-build) build=0 ;;
        --regen) regen=1 ;;
        --models) MODELS=$2; shift ;;
        --) pass+=("$@"); break ;;
        *) pass+=("$1") ;;
    esac
    shift
done

cd "$REPO"
if [[ $build == 1 ]]; then
    echo "building coli-swift (release, $SCRATCH) ..." >&2
    swift build -c release --scratch-path "$SCRATCH" --product coli-swift >&2
fi
BIN=$(swift build -c release --scratch-path "$SCRATCH" --show-bin-path)/coli-swift
[[ -x $BIN ]] || { echo "no coli-swift at $BIN (drop --no-build)" >&2; exit 2; }

if [[ $regen == 1 || ! -f $MODELS/manifest.json ]]; then
    echo "writing tiny models to $MODELS (TinyModelTests/writeOracleModels, debug build) ..." >&2
    mkdir -p "$MODELS"
    COLIBRI_TINY_OUT=$MODELS swift test --scratch-path "$SCRATCH" --filter TinyModelTests/writeOracleModels >&2
fi

exec python3 -I "$REPO/scripts/oracle/compare.py" --coli-swift "$BIN" --models "$MODELS" ${pass[@]+"${pass[@]}"}
