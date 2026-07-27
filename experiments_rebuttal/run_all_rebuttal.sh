#!/bin/bash
# ================================================================
# Run all 4 rebuttal experiments
# Usage: bash experiments_rebuttal/run_all_rebuttal.sh [GPU_ID]
# ================================================================
GPU=${1:-0}
cd "$(dirname "$0")/.."

echo "================================================="
echo "AdaST Rebuttal Experiments  (GPU=$GPU)"
echo "================================================="

# ── EXP 1: Noise Robustness ──
echo ""
echo "=== EXP 1: Noise Robustness ==="
echo "  (checkpoints auto-discovered)"
for DATASET in PurpleAir PEMS08; do
    if [ -f "datasets/${DATASET}/data.dat" ]; then
        python experiments_rebuttal/exp1_noise_robustness.py \
            --dataset $DATASET \
            --noise_levels 0.5 1.0  \
            --noise_types spatial temporal both \
            --gpus $GPU \
            --output_json experiments_rebuttal/results_exp1_${DATASET}.json \
            | tee experiments_rebuttal/log_exp1.txt
    else
        echo "  SKIP $DATASET: dataset not found"
    fi
done

# ── EXP 2: Alpha Sensitivity ──
echo ""
echo "=== EXP 2: Alpha Sensitivity (quick) ==="
# python experiments_rebuttal/exp2_alpha_sensitivity.py \
#     --mode quick \
#     --alpha_values 0.0 0.05 0.2 0.5 1.0 \
#     --layer_values 3 \
#     --gpus $GPU \
#     --output_json experiments_rebuttal/results_exp2_alpha.json

# Full alpha sweep on real data (uncomment when needed):
python experiments_rebuttal/exp2_alpha_sensitivity.py \
    --mode full --dataset PEMS08 \
    --alpha_values 0.0 0.05 0.1 0.5 1.0 \
    --layer_values 3 --epochs 30 --gpus $GPU \
    | tee experiments_rebuttal/log_exp2.txt
bash experiments_rebuttal/run_alpha_sweep.sh

# ── EXP 3: Weather ──
echo ""
echo "=== EXP 3: Weather Dataset ==="
if [ -f "datasets/Weather/data.dat" ]; then
    python experiments_rebuttal/exp3_weather_baselines.py \
        --models AdaST STAEformer DLinear \
        --epochs 30 --gpus $GPU \
        --output_json experiments_rebuttal/results_exp3_weather.json \
        | tee experiments_rebuttal/log_exp3.txt
else
    echo "  Generating configs (run data prep first)"
    python experiments_rebuttal/exp3_weather_baselines.py \
        --generate_script_only --models AdaST STAEformer DLinear \
        --epochs 30 --gpus $GPU \
        | tee experiments_rebuttal/log_exp3_config.txt
    echo "  Then: bash experiments_rebuttal/run_weather_exp.sh"
fi

# ── EXP 4: Dimension-Controlled Ablation ──
echo ""
echo "=== EXP 4: Dimension-Controlled Ablation ==="
python experiments_rebuttal/exp4_dim_controlled_ablation.py --mode verify
if [ -f "datasets/PEMS08/data.dat" ]; then
    python experiments_rebuttal/exp4_dim_controlled_ablation.py \
        --mode train \
        --datasets PEMS08 PurpleAir \
        --ablations none tod dow spatial adaptive \
        --epochs 30 --gpus $GPU \
        --output_json experiments_rebuttal/results_exp4_dim_ablation.json \ 
        | tee experiments_rebuttal/log_exp4.txt
else
    python experiments_rebuttal/exp4_dim_controlled_ablation.py \
        --mode generate --datasets PEMS08 PurpleAir \
        --epochs 30 --gpus $GPU \
        | tee experiments_rebuttal/log_exp4_config.txt
    echo "  Then: bash experiments_rebuttal/run_dim_ablation.sh"
fi

echo ""
echo "================================================="
echo "Done! Results in experiments_rebuttal/results_*.json"
echo "================================================="
