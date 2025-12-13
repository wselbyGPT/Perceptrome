#!/usr/bin/env bash
set -euo pipefail

# -----------------------------
# Defaults (override via flags)
# -----------------------------
ACCESSIONS_FILE="plasmid_accessions.txt"
SIZES="10 25 50 100"
STEPS_LIST="5"              # you can set "0 5" to isolate overhead vs training
BATCH_SIZE=16
WINDOW_SIZE=512
STRIDE=256
MAX_EPOCHS=1
DELETE_CACHE=1              # 1 => pass --delete-cache
RESET_EACH_RUN=1            # 1 => wipe checkpoint/state/cache/logs before each run
GPU_SAMPLE=1                # 1 => poll nvidia-smi during run
GPU_INTERVAL=1              # seconds between nvidia-smi samples

usage() {
  cat <<EOF
Usage: ./bench_minimal.sh [options]

Options:
  --accessions FILE      Source accession list (default: plasmid_accessions.txt)
  --sizes "A B C"        Sizes to test (default: "10 25 50 100")
  --steps "X Y"          Steps-per-plasmid values (default: "5")
  --batch N              Batch size (default: 16)
  --window N             Window size (default: 512)
  --stride N             Stride (default: 256)
  --epochs N             Max epochs (default: 1)
  --delete-cache 0|1     Delete per-accession cache after training (default: 1)
  --reset 0|1            Reset checkpoint/state/cache/logs each run (default: 1)
  --gpu-sample 0|1       Poll nvidia-smi during run (default: 1)
  --gpu-interval SEC     GPU poll interval seconds (default: 1)

Examples:
  ./bench_minimal.sh
  ./bench_minimal.sh --sizes "10 50 100" --steps "0 5" --delete-cache 1
  ./bench_minimal.sh --accessions plasmid_accessions.txt --sizes "200 500" --steps "5"
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --accessions) ACCESSIONS_FILE="$2"; shift 2;;
    --sizes) SIZES="$2"; shift 2;;
    --steps) STEPS_LIST="$2"; shift 2;;
    --batch) BATCH_SIZE="$2"; shift 2;;
    --window) WINDOW_SIZE="$2"; shift 2;;
    --stride) STRIDE="$2"; shift 2;;
    --epochs) MAX_EPOCHS="$2"; shift 2;;
    --delete-cache) DELETE_CACHE="$2"; shift 2;;
    --reset) RESET_EACH_RUN="$2"; shift 2;;
    --gpu-sample) GPU_SAMPLE="$2"; shift 2;;
    --gpu-interval) GPU_INTERVAL="$2"; shift 2;;
    -h|--help) usage; exit 0;;
    *) echo "Unknown arg: $1"; usage; exit 1;;
  esac
done

# -----------------------------
# Pre-flight checks
# -----------------------------
if [[ -z "${VIRTUAL_ENV:-}" ]]; then
  echo "[bench] WARNING: VIRTUAL_ENV not set. Run this inside your (venv) for correct torch/cuda."
fi

if [[ ! -f "$ACCESSIONS_FILE" ]]; then
  echo "[bench] ERROR: accessions file not found: $ACCESSIONS_FILE"
  exit 1
fi

if [[ ! -x /usr/bin/time ]]; then
  echo "[bench] ERROR: /usr/bin/time not found."
  exit 1
fi

has_nvidia_smi=0
if command -v nvidia-smi >/dev/null 2>&1; then
  has_nvidia_smi=1
fi

# -----------------------------
# Helpers
# -----------------------------
time_to_seconds() {
  # input like "1:46.61" or "0:12.34" or "1:02:03.45"
  local t="$1"
  awk -v t="$t" 'BEGIN{
    n=split(t,a,":");
    if(n==1){print a[1];}
    else if(n==2){print a[1]*60 + a[2];}
    else if(n==3){print a[1]*3600 + a[2]*60 + a[3];}
    else {print a[1];}
  }'
}

trim() {
  sed -E 's/^[[:space:]]+|[[:space:]]+$//g'
}

# -----------------------------
# Output folder
# -----------------------------
ts="$(date +%Y%m%d_%H%M%S)"
OUTDIR="bench_runs/$ts"
mkdir -p "$OUTDIR"

RESULTS_CSV="$OUTDIR/results.csv"
echo "timestamp,size,steps_per_plasmid,batch,window,stride,epochs,delete_cache,elapsed_s,user_s,sys_s,cpu_pct,max_rss_kb,gpu_avg_util,gpu_avg_mem_util,gpu_max_mem_used,gpu_mem_total,fetch_count" \
  > "$RESULTS_CSV"

echo "[bench] Output: $OUTDIR"
echo "[bench] Sizes: $SIZES"
echo "[bench] Steps: $STEPS_LIST"

# -----------------------------
# Main loop
# -----------------------------
for size in $SIZES; do
  catalog="config/plasmids_${size}.txt"
  mkdir -p config
  head -n "$size" "$ACCESSIONS_FILE" > "$catalog"

  for steps in $STEPS_LIST; do
    run_id="N${size}_S${steps}_B${BATCH_SIZE}_W${WINDOW_SIZE}_R${STRIDE}"
    run_dir="$OUTDIR/$run_id"
    mkdir -p "$run_dir"

    echo
    echo "[bench] ==== RUN $run_id ===="

    if [[ "$RESET_EACH_RUN" == "1" ]]; then
      rm -f model/checkpoints/latest.pt
      rm -f state/progress.json
      rm -rf cache/fasta cache/encoded
      rm -f logs/training.log logs/fetch.log logs/encode.log logs/scope.log
      python3 stream_train.py --config config/stream_config.yaml init > /dev/null
    fi

    # Start GPU sampling in background (optional)
    gpu_pid=""
    gpu_log="$run_dir/nvidia_smi.csv"
    if [[ "$GPU_SAMPLE" == "1" && "$has_nvidia_smi" == "1" ]]; then
      (
        # header
        echo "timestamp,util_gpu,util_mem,mem_used,mem_total" > "$gpu_log"
        while true; do
          nvidia-smi --query-gpu=timestamp,utilization.gpu,utilization.memory,memory.used,memory.total \
            --format=csv,noheader,nounits >> "$gpu_log" 2>/dev/null || true
          sleep "$GPU_INTERVAL"
        done
      ) &
      gpu_pid="$!"
      # ensure it dies
      trap '[[ -n "${gpu_pid:-}" ]] && kill "${gpu_pid}" 2>/dev/null || true' EXIT
    fi

    time_txt="$run_dir/time.txt"
    cmd=(python3 stream_train.py --config config/stream_config.yaml stream
      --catalog "$catalog"
      --max-epochs "$MAX_EPOCHS"
      --steps-per-plasmid "$steps"
      --batch-size "$BATCH_SIZE"
      --window-size "$WINDOW_SIZE"
      --stride "$STRIDE"
    )
    if [[ "$DELETE_CACHE" == "1" ]]; then
      cmd+=(--delete-cache)
    fi

    # Run + capture /usr/bin/time output
    /usr/bin/time -v "${cmd[@]}" 2> "$time_txt"

    # Stop GPU sampler
    if [[ -n "${gpu_pid:-}" ]]; then
      kill "$gpu_pid" 2>/dev/null || true
      gpu_pid=""
      trap - EXIT
    fi

    # Save logs/state for this run (if present)
    [[ -f logs/training.log ]] && cp logs/training.log "$run_dir/training.log" || true
    [[ -f logs/fetch.log ]] && cp logs/fetch.log "$run_dir/fetch.log" || true
    [[ -f state/progress.json ]] && cp state/progress.json "$run_dir/progress.json" || true
    cp "$catalog" "$run_dir/catalog.txt"

    # Parse time metrics
    user_s="$(grep -F 'User time (seconds):' "$time_txt" | awk -F: '{print $2}' | trim)"
    sys_s="$(grep -F 'System time (seconds):' "$time_txt" | awk -F: '{print $2}' | trim)"
    cpu_pct="$(grep -F 'Percent of CPU this job got:' "$time_txt" | awk -F: '{print $2}' | tr -d ' %' | trim)"
    elapsed_raw="$(grep -F 'Elapsed (wall clock) time' "$time_txt" | sed -E 's/.*: //')"
    elapsed_s="$(time_to_seconds "$elapsed_raw")"
    max_rss_kb="$(grep -F 'Maximum resident set size (kbytes):' "$time_txt" | awk -F: '{print $2}' | trim)"

    # Count actual NCBI fetches from fetch.log (rough signal)
    fetch_count="0"
    if [[ -f "$run_dir/fetch.log" ]]; then
      fetch_count="$(grep -c 'fetching from NCBI' "$run_dir/fetch.log" || true)"
    fi

    # Parse GPU stats if present
    gpu_avg_util=""
    gpu_avg_mem_util=""
    gpu_max_mem_used=""
    gpu_mem_total=""
    if [[ -f "$gpu_log" ]]; then
      # skip header
      read gpu_avg_util gpu_avg_mem_util gpu_max_mem_used gpu_mem_total < <(
        awk -F',' '
          NR==1{next}
          {
            for(i=2;i<=5;i++){gsub(/^[ \t]+|[ \t]+$/, "", $i)}
            util += $2; utilm += $3; n++
            if($4 > maxmem) maxmem = $4
            memtotal = $5
          }
          END{
            if(n>0){
              printf "%.2f %.2f %.0f %.0f\n", util/n, utilm/n, maxmem, memtotal
            } else {
              printf "  \n"
            }
          }' "$gpu_log"
      )
    fi

    # Append to results CSV
    echo "$(date +%Y-%m-%dT%H:%M:%S),$size,$steps,$BATCH_SIZE,$WINDOW_SIZE,$STRIDE,$MAX_EPOCHS,$DELETE_CACHE,$elapsed_s,$user_s,$sys_s,$cpu_pct,$max_rss_kb,$gpu_avg_util,$gpu_avg_mem_util,$gpu_max_mem_used,$gpu_mem_total,$fetch_count" \
      >> "$RESULTS_CSV"

    echo "[bench] wrote: $run_dir"
  done
done

echo
echo "[bench] DONE"
echo "[bench] Results: $RESULTS_CSV"
echo "[bench] Quick view:"
column -s, -t "$RESULTS_CSV" | sed -n '1,12p'
