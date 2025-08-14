#! /bin/bash
set -e

usage() {
  echo "Usage: $0 <op_name>"
  exit 1
}

require_command() {
  command -v "$1" >/dev/null 2>&1 || { echo "Error: $1 not found in PATH"; exit 1; }
}

#get the repo root directory, should be tt-metal
get_parent_repo_root() {
    git rev-parse --show-toplevel
}

#check if repo root is tt-metal, if not, exit
check_repo_root() {
    local repo_root="$1"
    if [[ "$(basename "$repo_root")" != "tt-metal" ]]; then
        echo "This script is intended to be run from the tt-metal repository."
        exit 1
    fi
}

#get current timestamp
get_timestamp() {
    date +"%Y%m%d_%H%M%S_%Z"
}

#runs tt-smi -s (snapshot) and parses for info. On failure, exit with an error.
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
    print(f'HOSTNAME=\"{host}\"')
    print(f'DRIVER=\"{driver}\"')
except Exception as e:
    print('Error: python3 failed to parse tt-smi output', file=sys.stderr)
    sys.exit(1)
" )" || { echo "Error: python3 failed to parse tt-smi output"; exit 1; }
}

#writes the tracking info to a JSON file
write_json() {
    local name="$1"
    local file="$2"
    local timestamp="$3"
    local commit="$4"
    local hostname="$5"
    local driver="$6"
    cat <<EOF > "$file"
{
  "metal_tracking_info": {
    "op_name": "$name",
    "timestamp": "$timestamp",
    "tt-metal_commit": "$commit",
    "hostname": "$hostname",
    "driver": "$driver"
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

    write_json "$OP_NAME" "$json_file" "$timestamp" "$commit" "$HOSTNAME" "$DRIVER"

    echo "Metal tracking info saved to $json_file"
}

main "$@"