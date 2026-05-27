#!/usr/bin/env bash
# Clean caches and verify which version of vlm_engine is loaded
set -euo pipefail
cd "$(dirname "$0")/.."

echo "=== 1. Cleaning all __pycache__ and .pyc ==="
find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
find . -name "*.pyc" -delete 2>/dev/null || true
echo "Done."

echo ""
echo "=== 2. Checking vlm_engine.py for fps=4 fix ==="
if grep -q '"fps": 4' src/perception/vlm_engine.py; then
    echo "OK: vlm_engine.py uses fps=4 (2x oversample, deterministic)"
else
    echo "ERROR: vlm_engine.py does NOT use fps=4"
    exit 1
fi

echo ""
echo "=== 3. Verifying _infer has no monkey-patch ==="
if grep -q "_force_uniform\|_orig_load\|_orig_fps" src/perception/vlm_engine.py; then
    echo "ERROR: _infer still contains monkey-patch code"
    exit 1
else
    echo "OK: _infer is clean (no monkey-patch)"
fi
echo ""
echo "=== All checks passed ==="
