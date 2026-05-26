#!/usr/bin/env bash
# Clean caches and verify which version of vlm_engine is loaded
set -euo pipefail
cd "$(dirname "$0")/.."

echo "=== 1. Cleaning all __pycache__ and .pyc ==="
find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
find . -name "*.pyc" -delete 2>/dev/null || true
echo "Done."

echo ""
echo "=== 2. Checking vlm_engine.py for fps=None fix ==="
if grep -q "processor.fps = None" src/perception/vlm_engine.py; then
    echo "OK: vlm_engine.py contains fps=None fix"
else
    echo "ERROR: vlm_engine.py does NOT contain fps=None fix"
    exit 1
fi

echo ""
echo "=== 3. Importing vlm_engine (same import chain as main.py) ==="
python -c "
import sys, inspect
sys.path.insert(0, '.')
from src.perception.vlm_engine import VLMEngine
src = inspect.getsource(VLMEngine._infer)
if 'fps = None' in src:
    print('OK: loaded _infer has fps=None fix')
else:
    print('ERROR: loaded _infer is OLD version, missing fps=None fix')
    for line in src.split('\n'):
        s = line.strip()
        if 'processor' in s.lower() and 'fps' in s.lower():
            print('  line:', s)
    exit(1)
"
echo ""
echo "=== All checks passed ==="
