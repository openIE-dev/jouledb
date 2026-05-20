#!/bin/bash
# Download SIFT1M dataset from corpus-texmex.irisa.fr
# Dataset: 1M vectors, 128 dimensions, Euclidean distance

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SIFT_DIR="$SCRIPT_DIR/sift"

mkdir -p "$SIFT_DIR"
cd "$SIFT_DIR"

echo "=== SIFT1M Dataset Download ==="
echo "Target directory: $SIFT_DIR"
echo ""

# Check if already downloaded
if [ -f "sift_base.fvecs" ] && [ -f "sift_query.fvecs" ] && [ -f "sift_groundtruth.ivecs" ]; then
    echo "SIFT1M dataset already exists. Skipping download."
    echo ""
    echo "Files present:"
    ls -lh sift_*.fvecs sift_*.ivecs 2>/dev/null || true
    exit 0
fi

# Download the archive
ARCHIVE="sift.tar.gz"
URL="ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz"

echo "Downloading SIFT1M from $URL..."
echo "(This is ~160MB, may take a few minutes)"
echo ""

if command -v curl &> /dev/null; then
    curl -L -o "$ARCHIVE" "$URL" --progress-bar
elif command -v wget &> /dev/null; then
    wget -O "$ARCHIVE" "$URL" --show-progress
else
    echo "Error: Neither curl nor wget found. Please install one."
    exit 1
fi

echo ""
echo "Extracting archive..."
tar -xzf "$ARCHIVE"

# Move files from nested directory if needed
if [ -d "sift" ]; then
    mv sift/* . 2>/dev/null || true
    rmdir sift 2>/dev/null || true
fi

# Clean up
rm -f "$ARCHIVE"

echo ""
echo "=== Download Complete ==="
echo ""
echo "Files:"
ls -lh *.fvecs *.ivecs 2>/dev/null || echo "Warning: Expected files not found"

echo ""
echo "Dataset statistics:"
echo "  - sift_base.fvecs:        1,000,000 vectors x 128 dims (~512MB)"
echo "  - sift_query.fvecs:       10,000 vectors x 128 dims (~5MB)"
echo "  - sift_groundtruth.ivecs: 10,000 x 100 nearest neighbors"
echo ""
echo "Run benchmark with: cargo bench --bench ann_benchmark -- --dataset sift"
