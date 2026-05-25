#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

cd "${REPO_ROOT}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib-${USER:-codex}}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/tmp/xdg-cache-${USER:-codex}}"
mkdir -p "${MPLCONFIGDIR}"
mkdir -p "${XDG_CACHE_HOME}"

python experiments/ai4sci_paper_draw/risk_radar_compare/risk_radar_compare.py \
  --config experiments/ai4sci_paper_draw/risk_radar_compare/risk_radar_compare.yaml
