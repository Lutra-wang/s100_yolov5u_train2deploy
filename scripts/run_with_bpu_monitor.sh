#!/usr/bin/env bash
set -uo pipefail

usage() {
  echo "Usage: $0 --log PATH [--interval SECONDS] -- COMMAND [ARG ...]" >&2
}

log_path=""
interval="0.2"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --log)
      if [[ $# -lt 2 || -z "$2" ]]; then usage; exit 2; fi
      log_path="$2"; shift 2 ;;
    --interval)
      if [[ $# -lt 2 || -z "$2" ]]; then usage; exit 2; fi
      interval="$2"; shift 2 ;;
    --) shift; break ;;
    *) usage; exit 2 ;;
  esac
done

if [[ -z "$log_path" || $# -eq 0 ]]; then
  usage
  exit 2
fi

if [[ ! "$interval" =~ ^([0-9]+([.][0-9]+)?|[.][0-9]+)$ ]] || ! awk -v value="$interval" 'BEGIN { exit !(value > 0) }'; then
  echo "Interval must be a positive number: $interval" >&2
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

samples_file="$(mktemp)"
output_file="$(mktemp)"
command_pid=""
monitor_pid=""
interrupted=0

cleanup() {
  if [[ -n "$command_pid" ]]; then kill "$command_pid" 2>/dev/null || true; fi
  if [[ -n "$monitor_pid" ]]; then kill "$monitor_pid" 2>/dev/null || true; fi
  rm -f "$samples_file" "$output_file"
}
trap cleanup EXIT

on_signal() {
  interrupted=1
  if [[ -n "$command_pid" ]]; then kill "$command_pid" 2>/dev/null || true; fi
  if [[ -n "$monitor_pid" ]]; then kill "$monitor_pid" 2>/dev/null || true; fi
}
trap on_signal INT TERM

run_session() {
  local command_status
  "$@" >"$output_file" 2>&1 &
  command_pid=$!

  (
    while kill -0 "$command_pid" 2>/dev/null; do
      ratio="$(tr -d '[:space:]' < "$ratio_file")"
      if [[ "$ratio" =~ ^[0-9]+$ ]]; then
        line="$(printf '%s [BPU] ratio=%s%%' "$(date --iso-8601=milliseconds)" "$ratio")"
        printf '%s\n' "$line" >> "$samples_file"
        printf '%s\n' "$line" | tee -a "$log_path"
      fi
      sleep "$interval"
    done
  ) &
  monitor_pid=$!

  trap 'kill "$monitor_pid" 2>/dev/null || true' INT TERM EXIT
  wait "$command_pid"
  command_status=$?
  wait "$monitor_pid" 2>/dev/null || true
  command_pid=""
  monitor_pid=""
  return "$command_status"
}

run_session "$@"
command_status=$?
cat "$output_file" | tee -a "$log_path"

awk -F'ratio=|%' '
  {
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
' "$samples_file" | tee -a "$log_path"

if [[ "$interrupted" -eq 1 ]]; then
  exit 143
fi

exit "$command_status"
