#! /bin/bash
set -e

#find parent repo root, as this repo is included as cpm package in tt-metal
PARENT_REPO_ROOT=$(git rev-parse --show-toplevel)

#if parent repo root is not called tt-metal, exit
if [[ "$(basename "$PARENT_REPO_ROOT")" != "tt-metal" ]]; then
    echo "This script is intended to be run from the tt-metal repository."
    exit 1
fi

#create directory for created json files if not already existing
TARGET_DIR="$PARENT_REPO_ROOT/mlp-op-perf_tracking_details"
mkdir -p "$TARGET_DIR"

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
JSON_FILE="$TARGET_DIR/tt-metal_tracking_info_$TIMESTAMP.json"

cat <<EOF > "$JSON_FILE"
{
  "timestamp": "$TIMESTAMP",
  "tt-metal commit": "$(git rev-parse HEAD)"
}
EOF

echo "Metal tracking info saved to $JSON_FILE"