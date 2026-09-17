#!/usr/bin/env bash
set -euo pipefail
REPORT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPORT_ROOT"
mkdir -p build/cache build/config tmp
export XDG_CACHE_HOME="$REPORT_ROOT/build/cache"
export XDG_CONFIG_HOME="$REPORT_ROOT/build/config"
export TMPDIR="$REPORT_ROOT/tmp"
if [[ -n "${TECTONIC_BIN:-}" ]]; then
  "$TECTONIC_BIN" --only-cached --keep-logs --keep-intermediates report.tex
elif command -v tectonic >/dev/null 2>&1; then
  tectonic --keep-logs --keep-intermediates report.tex
elif command -v xelatex >/dev/null 2>&1; then
  for pass in 1 2 3; do
    xelatex -interaction=nonstopmode -halt-on-error report.tex
  done
else
  echo 'Need an existing XeLaTeX or Tectonic installation; no automatic installation.' >&2
  exit 1
fi
