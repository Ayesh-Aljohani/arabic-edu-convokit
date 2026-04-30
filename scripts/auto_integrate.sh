#!/usr/bin/env bash
# Auto-integration loop. Waits for each grouped result file to appear, runs
# integrate_results.py, recompiles the paper. Exits when all 5 grouped JSONs
# exist or after MAX_WAIT seconds.

set -e
cd /Users/ayesh/Projects/arabic-edu-convokit
source .venv/bin/activate

EXPECTED=(
  results/classification/focusing_questions_arabert_grouped_results.json
  results/classification/focusing_questions_mbert_grouped_results.json
  results/classification/focusing_questions_xlmr_grouped_results.json
  results/classification/student_reasoning_mbert_grouped_results.json
  results/classification/uptake_xlmr_grouped_results.json
)

PAPER_DIR="paper/the new paper"
SEEN=()

while true; do
  ALL=true
  for f in "${EXPECTED[@]}"; do
    if [ ! -f "$f" ]; then
      ALL=false
    elif ! [[ " ${SEEN[*]:-} " =~ " ${f} " ]]; then
      echo "[$(date +%H:%M:%S)] New file: $f"
      SEEN+=("$f")
      python scripts/integrate_results.py 2>&1 | tail -3
      cd "$PAPER_DIR"
      pdflatex -interaction=nonstopmode main.tex > /tmp/build_a.log 2>&1
      pdflatex -interaction=nonstopmode main.tex > /tmp/build_b.log 2>&1
      grep "Output written" /tmp/build_b.log
      cd /Users/ayesh/Projects/arabic-edu-convokit
    fi
  done
  if $ALL; then
    echo "[$(date +%H:%M:%S)] All 5 grouped JSONs present. Final integration."
    python scripts/integrate_results.py 2>&1 | tail -3
    cd "$PAPER_DIR"
    pdflatex -interaction=nonstopmode main.tex > /tmp/build_a.log 2>&1
    bibtex main > /tmp/build_b.log 2>&1
    pdflatex -interaction=nonstopmode main.tex > /tmp/build_c.log 2>&1
    pdflatex -interaction=nonstopmode main.tex > /tmp/build_d.log 2>&1
    grep "Output written" /tmp/build_d.log
    grep -E "^!" main.log | head -3 || echo "no errors"
    echo "[$(date +%H:%M:%S)] AUTO_INTEGRATE_COMPLETE"
    exit 0
  fi
  sleep 60
done
