#!/usr/bin/env bash
# A small job queue for an unattended runtime. QUEUE is a text file with one job per line,
#     name | shell command
# run in order, each once. A job that succeeds leaves $LOGS/queue_<q>/<name>.done, one that fails
# leaves <name>.failed (and the queue moves on); its output goes to $LOGS/<name>.log. Lines added
# to the file while the queue runs are picked up. The queue exits once every job is done or failed
# AND the file contains a line "END"; otherwise it waits for new lines.
#
#   nohup bash scripts/queue.sh configs/queues/a.txt > /content/work/logs/queue_a.log 2>&1 &
set -uo pipefail
q="$1"
qname="$(basename "$q" .txt)"
LOGS="${LOGS:-/content/work/logs}"
state="$LOGS/queue_$qname"
mkdir -p "$state"
trim() { local s="$1"; s="${s#"${s%%[![:space:]]*}"}"; s="${s%"${s##*[![:space:]]}"}"; printf '%s' "$s"; }

while true; do
  next_job=""; next_cmd=""; has_end=0
  while IFS= read -r line || [[ -n "$line" ]]; do
    line="$(trim "$line")"
    [[ -z "$line" || "$line" == \#* ]] && continue
    if [[ "$line" == END ]]; then has_end=1; continue; fi
    [[ "$line" != *"|"* ]] && continue
    job="$(trim "${line%%|*}")"
    [[ -e "$state/$job.done" || -e "$state/$job.failed" ]] && continue
    if [[ -z "$next_job" ]]; then next_job="$job"; next_cmd="$(trim "${line#*|}")"; fi
  done < "$q"
  if [[ -z "$next_job" ]]; then
    if (( has_end )); then echo "[$(date +%H:%M:%S)] QUEUE $qname FINISHED"; exit 0; fi
    sleep 60; continue
  fi
  echo "[$(date +%H:%M:%S)] START $next_job: $next_cmd"
  if bash -c "$next_cmd" > "$LOGS/$next_job.log" 2>&1; then
    touch "$state/$next_job.done"; echo "[$(date +%H:%M:%S)] DONE $next_job"
  else
    touch "$state/$next_job.failed"; echo "[$(date +%H:%M:%S)] FAILED $next_job (see $LOGS/$next_job.log)"
  fi
done
