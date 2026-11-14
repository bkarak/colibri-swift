#!/bin/bash
# Builds upstream colibri's qwen36 and olmoe engines natively, from a copy of the upstream
# checkout, so the checkout itself is never touched. Single-threaded (no libomp needed).
#
#   scripts/oracle/build-upstream.sh            # build if the binaries are missing
#   scripts/oracle/build-upstream.sh --force    # recopy and rebuild
#
# Environment:
#   COLIBRI_UPSTREAM   upstream checkout (default ~/devel/personal/_upstream/colibri)
#   ORACLE_DIR         work directory (default ${TMPDIR:-/tmp}/colibri-oracle)
set -euo pipefail

UPSTREAM=${COLIBRI_UPSTREAM:-$HOME/devel/personal/_upstream/colibri}
ORACLE_DIR=${ORACLE_DIR:-${TMPDIR:-/tmp}/colibri-oracle}
EXPECTED_COMMIT=0a5504a

force=0
[[ ${1:-} == --force ]] && force=1

if [[ $force == 0 && -x $ORACLE_DIR/c/qwen36 && -x $ORACLE_DIR/c/olmoe ]]; then
    echo "upstream engines present in $ORACLE_DIR/c"
    exit 0
fi

[[ -f $UPSTREAM/c/qwen36.c ]] || { echo "no upstream checkout at $UPSTREAM" >&2; exit 1; }
if git -C "$UPSTREAM" rev-parse --short HEAD >/dev/null 2>&1; then
    head=$(git -C "$UPSTREAM" rev-parse --short=7 HEAD)
    [[ $head == "$EXPECTED_COMMIT" ]] || echo "warning: upstream is at $head, fixtures were recorded at $EXPECTED_COMMIT" >&2
fi

mkdir -p "$ORACLE_DIR"
rm -rf "$ORACLE_DIR/c"
cp -R "$UPSTREAM/c" "$ORACLE_DIR/c"
cd "$ORACLE_DIR/c"
make qwen36 olmoe
ls -l qwen36 olmoe
