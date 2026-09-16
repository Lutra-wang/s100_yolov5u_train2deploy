#!/usr/bin/env bash
set -uo pipefail

usage() {
  echo "Usage: $0 --log PATH [--interval SECONDS] -- COMMAND [ARG ...]" >&2
}

log_path=""
interval="0.2"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --log) log_path="${2:-}"; shift 2 ;;
    --interval) interval="${2:-}"; shift 2 ;;
    --) shift; break ;;
    *) usage; exit 2 ;;
  esac
done

if [[ -z "$log_path" || $# -eq 0 ]]; then
  usage
  exit 2
fi

if [[ -n "${BPU_RATIO_FILE:-}" ]]; then
  ratio_file="$BPU_RATIO_FILE"
elif [[ -r /sys/devices/system/bpu/bpu0/ratio ]]; then
  ratio_file=/sys/devices/system/bpu/bpu0/ratio
else
  ratio_file=/sys/devices/system/bpu/ratio
fi

if [[ ! -r "$ratio_file" ]]; then
  echo "BPU ratio file is not readable: $ratio_file" >&2
  exit 1
fi

mkdir -p "$(dirname "$log_path")"
: > "$log_path"

run_session() {
  local command_pid monitor_pid command_status
  "$@" &
  command_pid=$!

  (
    while kill -0 "$command_pid" 2>/dev/null; do
      ratio="$(tr -d '[:space:]' < "$ratio_file")"
      if [[ "$ratio" =~ ^[0-9]+$ ]]; then
        printf '%s [BPU] ratio=%s%%\n' "$(date --iso-8601=milliseconds)" "$ratio"
      fi
      sleep "$interval"
    done
  ) &
  monitor_pid=$!

  trap 'kill "$monitor_pid" 2>/dev/null || true' INT TERM EXIT
  wait "$command_pid"
  command_status=$?
  wait "$monitor_pid" 2>/dev/null || true
  trap - INT TERM EXIT
  return "$command_status"
}

run_session "$@" 2>&1 | tee "$log_path"
command_status=${PIPESTATUS[0]}

awk -F'ratio=|%' '
  /\[BPU\] ratio=/ {
    value = $2 + 0
    sum += value
    count += 1
    if (count == 1 || value > peak) peak = value
  }
  END {
    if (count > 0)
      printf "[BPU_SUMMARY] samples=%d average=%.1f%% peak=%d%%\n", count, sum / count, peak
    else
      print "[BPU_SUMMARY] samples=0 average=n/a peak=n/a"
  }
' "$log_path" | tee -a "$log_path"

exit "$command_status"
