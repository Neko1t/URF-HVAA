#!/usr/bin/env bash
# ============================================================
# Extract video files (.mp4) from compressed archives in
# the video folder. Handles .zip, .7z, .tar.gz, .rar.
#
# Usage:
#   bash scripts/extract_videos.sh                          # default: data/ucf_crime/videos
#   bash scripts/extract_videos.sh data/xd_violence/videos  # specific folder
#   bash scripts/extract_videos.sh --keep-archives          # don't delete archives after
# ============================================================
set -euo pipefail

cd "$(dirname "$0")/.."

VIDEO_DIR="${1:-data/ucf_crime/videos}"
KEEP_ARCHIVES=false

for arg in "$@"; do
    case "$arg" in
        --keep-archives) KEEP_ARCHIVES=true ;;
        --help|-h)
            echo "Usage: bash scripts/extract_videos.sh [video_dir] [--keep-archives]"
            echo "  video_dir        Target directory (default: data/ucf_crime/videos)"
            echo "  --keep-archives  Don't delete archives after extraction"
            exit 0
            ;;
    esac
done

if [ ! -d "$VIDEO_DIR" ]; then
    echo "[ERROR] Directory not found: $VIDEO_DIR"
    exit 1
fi

echo "=== Extract videos ==="
echo "  Target: $VIDEO_DIR"
echo "  Keep archives: $KEEP_ARCHIVES"
echo ""

extracted=0
skipped=0

_extract_zip() {
    local zip="$1"
    local dir="$2"
    python - "$zip" "$dir" << 'PYEOF'
import os, sys, zipfile

zf = zipfile.ZipFile(sys.argv[1], 'r')
target = sys.argv[2]
mp4s = [n for n in zf.namelist() if n.lower().endswith('.mp4')]

if not mp4s:
    print(f"    [SKIP] No .mp4 files in {os.path.basename(sys.argv[1])}")
    sys.exit(0)

for mp4_path in mp4s:
    fname = os.path.basename(mp4_path)
    dest = os.path.join(target, fname)
    if os.path.exists(dest):
        print(f"    [SKIP] {fname} already exists")
        continue
    print(f"    Extracting: {mp4_path}")
    zf.extract(mp4_path, target)
    extracted_path = os.path.join(target, mp4_path)
    if os.path.exists(extracted_path) and extracted_path != dest:
        os.rename(extracted_path, dest)
        # Clean up empty parent dirs
        parent = os.path.dirname(extracted_path)
        while parent != target:
            try:
                os.rmdir(parent)
            except OSError:
                break
            parent = os.path.dirname(parent)
    print(f"      -> {fname}")
zf.close()
PYEOF
}

# ---- Find and process archives ----
shopt -s nullglob

for ext in zip 7z "tar.gz" rar; do
    # Use find to handle both single-extension and double-extension patterns
    while IFS= read -r -d '' archive; do
        basename_archive=$(basename "$archive")
        echo "  [$basename_archive]"

        case "$ext" in
            zip)
                _extract_zip "$archive" "$VIDEO_DIR"
                ;;
            7z)
                mp4_count=$(7z l "$archive" 2>/dev/null | grep -ci '\.mp4$' || true)
                if [ "${mp4_count:-0}" -gt 0 ]; then
                    7z x "$archive" -o"$VIDEO_DIR" -y -aos > /dev/null
                    count=$(7z l "$archive" 2>/dev/null | grep -ci '\.mp4$' || true)
                    echo "    Extracted $count .mp4 file(s)"
                    ((extracted+=count))
                else
                    echo "    [SKIP] No .mp4 files"
                fi
                ;;
            "tar.gz")
                tar -xzf "$archive" -C "$VIDEO_DIR" --wildcards '*.mp4' 2>/dev/null || \
                    echo "    [SKIP] No .mp4 files or extraction failed"
                ;;
            rar)
                if command -v unrar &>/dev/null; then
                    unrar x -y "$archive" "$VIDEO_DIR/" '*.mp4' 2>/dev/null || \
                        echo "    [SKIP] No .mp4 files or extraction failed"
                else
                    echo "    [SKIP] unrar not installed"
                fi
                ;;
        esac

        if [ "$KEEP_ARCHIVES" = false ]; then
            rm -f "$archive"
            echo "    Removed $basename_archive"
        fi
    done < <(find "$VIDEO_DIR" -maxdepth 1 -name "*.$ext" -print0 2>/dev/null || true)
done

# Handle Kaggle double-zip: Anomaly-Videos-Part-X.zip.zip
while IFS= read -r -d '' double_zip; do
    echo "  [$(basename "$double_zip")] unwrapping double-zip"
    # Extract inner zip
    unzip -o "$double_zip" -d "$VIDEO_DIR" > /dev/null 2>&1 || true
    rm -f "$double_zip"
    # Now extract the inner zip (which was extracted from the double zip)
    inner_zip="${double_zip%.zip}"
    if [ -f "$inner_zip" ]; then
        _extract_zip "$inner_zip" "$VIDEO_DIR"
        rm -f "$inner_zip"
    fi
done < <(find "$VIDEO_DIR" -maxdepth 1 -name "*.zip.zip" -print0 2>/dev/null || true)

echo ""
echo "=== Done ==="
echo "  Extracted: $extracted"
echo "  Skipped:   $skipped (already exist)"
echo "  Files in $VIDEO_DIR:"
ls -1 "$VIDEO_DIR"/*.mp4 2>/dev/null | while read -r f; do
    echo "    $(basename "$f")"
done || echo "    (none)"
