#!/bin/bash
# run_parallel.sh
# 在单张 GPU 上并行运行 4 个训练任务

set -u

# ============ 配置 ============
MAX_PARALLEL=1                   # 同时运行的任务数
GPU_ID=0                          # 使用的 GPU 编号
LOG_DIR="./logs/$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"

# 任务列表:每个元素是 "模型名 数据集名"
TASKS=(
    "HimNet METR-LA"
)

# ============ 运行单个任务的函数 ============
run_task() {
    local model=$1
    local dataset=$2
    local log_file="$LOG_DIR/${model}_${dataset}.log"

    echo "[$(date +%H:%M:%S)] START: $model / $dataset (PID $$)"
    CUDA_VISIBLE_DEVICES=$GPU_ID \
        python -u experiments/train.py -c "baselines/${model}/${dataset}.py" \
        > "$log_file" 2>&1
    local status=$?
    if [ $status -eq 0 ]; then
        echo "[$(date +%H:%M:%S)] DONE : $model / $dataset"
    else
        echo "[$(date +%H:%M:%S)] FAIL : $model / $dataset (exit $status, see $log_file)"
    fi
    return $status
}

# ============ 并发调度 ============
echo "Total tasks: ${#TASKS[@]}, parallel: $MAX_PARALLEL, GPU: $GPU_ID"
echo "Logs -> $LOG_DIR"
echo "================================================"

# Ctrl+C 时杀掉所有子进程
trap 'echo "Interrupted, killing children..."; kill 0; exit 130' INT TERM

for task in "${TASKS[@]}"; do
    # 等待直到正在运行的子进程数 < MAX_PARALLEL
    while [ "$(jobs -rp | wc -l)" -ge "$MAX_PARALLEL" ]; do
        # wait -n 等待任意一个子进程结束 (bash 4.3+)
        wait -n
    done

    read -r model dataset <<< "$task"
    run_task "$model" "$dataset" &
done

# 等待剩余任务完成
wait
echo "================================================"
echo "All tasks finished. Logs in $LOG_DIR"




