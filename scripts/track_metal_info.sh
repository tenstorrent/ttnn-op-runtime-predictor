#! /bin/bash
set -e

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

get_tt_smi_info() {
    eval "$(tt-smi -s | python3 -c "
import sys, json
data = json.load(sys.stdin)
host = data.get('host_info', {}).get('Hostname', '')
driver = data.get('host_info', {}).get('Driver', '')
print(f'HOSTNAME=\"{host}\"')
print(f'DRIVER=\"{driver}\"')
")"
}

create_target_dir() {
    local dir="$1"
    mkdir -p "$dir"
}

write_json() {
    local file="$1"
    local timestamp="$2"
    local commit="$3"
    local hostname="$4"
    local driver="$5"
    cat <<EOF > "$file"
{
  "timestamp": "$timestamp",
  "tt-metal_commit": "$commit",
  "hostname": "$hostname",
  "driver": "$driver"
}
EOF
}

main() {
    local parent_repo_root
    parent_repo_root=$(get_parent_repo_root)
    check_repo_root "$parent_repo_root"

    local timestamp
    timestamp=$(get_timestamp)

    get_tt_smi_info

    local target_dir="$parent_repo_root/mlp-op-perf_tracking_details"
    create_target_dir "$target_dir"

    local json_file="$target_dir/tt-metal_tracking_info_${timestamp}.json"
    local commit
    commit=$(git rev-parse HEAD)

    write_json "$json_file" "$timestamp" "$commit" "$HOSTNAME" "$DRIVER"

    echo "Metal tracking info saved to $json_file"
}

main "$@"