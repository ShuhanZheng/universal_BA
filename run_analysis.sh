#!/bin/bash
# Convenience wrapper: run the full pipeline and show the tail of the log.
#
# Usage:
#   ./run_analysis.sh                 # uses `python3` from PATH
#   PYTHON=/path/to/python ./run_analysis.sh
#
# Analysis.py must be executed from the repository root, because Input.py
# resolves every data path relative to the working directory.
set -euo pipefail

cd "$(dirname "$0")"
PYTHON="${PYTHON:-python3}"
LOG="${LOG:-analysis_run.log}"

"$PYTHON" Analysis.py > "$LOG" 2>&1
tail -40 "$LOG"
