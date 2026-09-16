#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$PROJECT_ROOT/docs/report"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$PROJECT_ROOT/.cache}"
"${TECTONIC:-tectonic}" --keep-logs report.tex
