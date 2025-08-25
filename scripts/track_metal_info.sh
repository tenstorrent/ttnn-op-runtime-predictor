#! /bin/bash

# SPDX-FileCopyrightText: (c) 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

set -e

usage() {
  echo "Usage: $0 <op_name>"
  exit 1
}

require_command() {
  command -v "$1" >/dev/null 2>&1 || { echo "Error: $1 not found in PATH"; exit 1; }
}

get_parent_repo_root() {
    git rev-parse --show-toplevel
}

check_repo_root() {
    local repo_root="$1"
    if [[ "$(basename "$repo_root")" != "tt-metal" ]]; then
        echo "This script is intended to be run from the tt-metal repository."
        exit 1
    fi
}

get_timestamp() {
    date +"%Y%m%d_%H%M%S_%Z"
}

# Runs tt-smi -s and parses for info. Exports HOSTNAME, DRIVER, BOARD_TYPE, DEVICE_NAME.
get_tt_smi_info() {
    local output
    if ! output=$(tt-smi -s 2>/dev/null); then
        echo "Error: tt-smi -s failed"
        exit 1
    fi
    eval "$(
        echo "$output" | python3 -c "
import sys, json
try:
    data = json.load(sys.stdin)
    host = data.get('host_info', {}).get('Hostname', '')
    driver = data.get('host_info', {}).get('Driver', '')
    device_info = data.get('device_info', [])
    if isinstance(device_info, list) and device_info:
        board_info = device_info[0].get('board_info', {})
        board_type = board_info.get('board_type', '')
    else:
        board_type = ''
    board_type_lower = board_type.lower()
    board_type_key = ''.join(
        c for c in board_type_lower
        if c.isalnum()
    )
    prefix = ''
    for c in board_type_lower:
        if c.isalnum():
            prefix += c
        else:
            break
    board_type_key = prefix
    device_map = {
        'e150': 'Grayskull',
        'e300': 'Grayskull',
        'e75': 'Grayskull',
        'nbcb': 'Wormhole',
        'wh4u': 'Wormhole',
        'n300': 'Wormhole',
        'n150': 'Wormhole',
        'ttgalaxywh': 'Wormhole',
        'bhscrappy': 'Blackhole',
        'p100a': 'Blackhole',
        'p150a': 'Blackhole',
        'p150b': 'Blackhole'
    }
    device_name = device_map.get(board_type_key, 'N/A')
    print(f'export HOSTNAME=\"{host}\"')
    print(f'export DRIVER=\"{driver}\"')
    print(f'export BOARD_TYPE=\"{board_type}\"')
    print(f'export DEVICE_NAME=\"{device_name}\"')
except Exception as e:
    print('Error: python3 failed to parse tt-smi output', file=sys.stderr)
    sys.exit(1)
" )" || { echo "Error: python3 failed to parse tt-smi output"; exit 1; }
}

write_json() {
    local name="$1"
    local file="$2"
    local timestamp="$3"
    local commit="$4"
    local hostname="$5"
    local driver="$6"
    local board_type="$7"
    local device_name="$8"
    cat <<EOF > "$file"
{
  "metal_tracking_info": {
    "op_name": "$name",
    "timestamp": "$timestamp",
    "tt-metal_commit": "$commit",
    "hostname": "$hostname",
    "driver": "$driver",
    "board_type": "$board_type",
    "device_name": "$device_name"
  }
}
EOF
}

main() {
  if [[ $# -ne 1 ]]; then
    usage
  fi

  require_command git
  require_command tt-smi
  require_command python3

  local parent_repo_root
  parent_repo_root=$(get_parent_repo_root)
  check_repo_root "$parent_repo_root"

  local timestamp
  timestamp=$(get_timestamp)

  get_tt_smi_info

  OP_NAME="$1"

  local dir="$parent_repo_root/ttnn_op_runtime_predictor_tracking_details"
  mkdir -p "$dir"

  local json_file="$dir/tt-metal_tracking_info_${timestamp}.json"
  local commit
  commit=$(git rev-parse HEAD)

  write_json "$OP_NAME" "$json_file" "$timestamp" "$commit" "$HOSTNAME" "$DRIVER" "$BOARD_TYPE" "$DEVICE_NAME"

  echo "Metal tracking info saved to $json_file"
}
main "$@"
