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

temp_dir="$(mktemp -d)" || { echo "Unable to create temporary directory" >&2; exit 1; }
samples_file="$temp_dir/samples"
stream_fifo="$temp_dir/stream"
if ! mkfifo "$stream_fifo"; then
  echo "Unable to create stream FIFO: $stream_fifo" >&2
  rm -rf "$temp_dir"
  exit 1
fi
command_pid=""
monitor_pid=""
aggregator_pid=""
interrupted=0

kill_command() {
  local child_pid
  [[ -z "$command_pid" ]] && return 0
  while read -r child_pid; do
    [[ -n "$child_pid" ]] || continue
    kill -KILL "$child_pid" 2>/dev/null || true
  done < <(pgrep -P "$command_pid" 2>/dev/null || true)
  kill -KILL "$command_pid" 2>/dev/null || true
}

cleanup() {
  kill_command
  if [[ -n "$monitor_pid" ]]; then kill "$monitor_pid" 2>/dev/null || true; fi
  if [[ -n "$aggregator_pid" ]]; then kill "$aggregator_pid" 2>/dev/null || true; fi
  if [[ -n "$command_pid" ]]; then wait "$command_pid" 2>/dev/null || true; fi
  if [[ -n "$monitor_pid" ]]; then wait "$monitor_pid" 2>/dev/null || true; fi
  if [[ -n "$aggregator_pid" ]]; then wait "$aggregator_pid" 2>/dev/null || true; fi
  rm -rf "$temp_dir"
}
trap cleanup EXIT

on_signal() {
  interrupted=1
  kill_command
  if [[ -n "$monitor_pid" ]]; then kill "$monitor_pid" 2>/dev/null || true; fi
  if [[ -n "$aggregator_pid" ]]; then kill "$aggregator_pid" 2>/dev/null || true; fi
  exit 143
}
trap on_signal INT TERM

tee -a "$log_path" < "$stream_fifo" &
aggregator_pid=$!

run_session() {
  local command_status
  "$@" >"$stream_fifo" 2>&1 &
  command_pid=$!

  (
    while kill -0 "$command_pid" 2>/dev/null; do
      ratio="$(tr -d '[:space:]' < "$ratio_file")"
      if [[ "$ratio" =~ ^[0-9]+$ ]]; then
        line="$(printf '%s [BPU] ratio=%s%%' "$(date --iso-8601=milliseconds)" "$ratio")"
        printf '%s\n' "$line" >> "$samples_file"
        printf '%s\n' "$line" > "$stream_fifo"
      fi
      sleep "$interval"
    done
  ) &
  monitor_pid=$!

  wait "$command_pid"
  command_status=$?
  command_pid=""
  kill "$monitor_pid" 2>/dev/null || true
  wait "$monitor_pid" 2>/dev/null || true
  monitor_pid=""
  wait "$aggregator_pid" 2>/dev/null || true
  aggregator_pid=""
  return "$command_status"
}

run_session "$@"
command_status=$?

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
