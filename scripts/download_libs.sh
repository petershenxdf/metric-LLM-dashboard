#!/usr/bin/env bash
# Download D3 and d3-lasso into static/lib/ for offline use.
set -e

LIB_DIR="$(dirname "$0")/../static/lib"
mkdir -p "$LIB_DIR"

echo "Downloading D3 v7..."
curl -sSL -o "$LIB_DIR/d3.v7.min.js" \
    https://cdnjs.cloudflare.com/ajax/libs/d3/7.8.5/d3.min.js

echo "Downloading d3-lasso..."
curl -sSL -o "$LIB_DIR/d3-lasso.min.js" \
    https://cdn.jsdelivr.net/npm/d3-lasso@0.0.5/build/d3-lasso.min.js

echo "Done. Libraries installed to $LIB_DIR"
ls -la "$LIB_DIR"
