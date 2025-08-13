#! /bin/bash
set -e

# Find parent repo root, as this repo may be a cpm package
PARENT_REPO_ROOT=$(git rev-parse --show-toplevel)

#if parent repo root is not called tt-metal, exit
if [[ "$(basename "$PARENT_REPO_ROOT")" != "tt-metal" ]]; then
    echo "This script is intended to be run from the tt-metal repository."
    exit 1
fi

TARGET_DIR="$PARENT_REPO_ROOT/mlp-op-perf_tracking_details"

mkdir -p "$TARGET_DIR"

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
JSON_FILE="$TARGET_DIR/tt-metal_tracking_info_$TIMESTAMP.json"

cat <<EOF > "$JSON_FILE"
{
  "test_name": "mlp_op_performance_test",
  "timestamp": "$TIMESTAMP"
}
EOF

echo "Test results saved to $JSON_FILE"