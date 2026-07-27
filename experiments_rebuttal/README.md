# AdaST Rebuttal Experiments

Four experiments designed to address reviewer concerns for ICML 2026 submission #6232.

---

## Prerequisites

1. Ensure the repository is set up and datasets are prepared:
   ```bash
   cd <repo_root>
   # Install dependencies
   pip install -r requirements.txt
   # Prepare datasets (example: PEMS04, PEMS07, PurpleAir, Weather)
   python scripts/data_preparation/PEMS04/generate_training_data.py
   python scripts/data_preparation/PEMS07/generate_training_data.py
   python scripts/data_preparation/PurpleAir/generate_training_data.py
   python scripts/data_preparation/Weather/generate_training_data.py
   ```
2. Train the base AdaST models first (to get checkpoints for Exp1):
   ```bash
   python experiments/train.py -c baselines/AdaST/PEMS04.py -g 0
   python experiments/train.py -c baselines/AdaST/PurpleAir.py -g 0
   python experiments/train.py -c baselines/STAEformer/PEMS04.py -g 0
   python experiments/train.py -c baselines/STAEformer/PurpleAir.py -g 0
   ```

---

## Experiment 1: Noise Robustness (Spurious Correlation)

**Addresses**: Reviewer CBef Q1 & Q4
> "The paper argues that AdaST mitigates spurious dependencies, but the authors
>  do not directly establish that the model is removing spurious correlations."

**Design**: Inject Gaussian noise into spatial/temporal dimensions at test time.
AdaST should be MORE robust than baselines because its adaptive gating
suppresses irrelevant components.

- On **PurpleAir** (temporal-dominated, low spatial gate):
  → AdaST should degrade LESS under spatial noise
- On **PEMS04/07/08** (spatial-dominated, low temporal gate):
  → AdaST should degrade LESS under temporal noise

```bash
# PurpleAir (temporal-dominated dataset)
python experiments_rebuttal/exp1_noise_robustness.py \
    --dataset PurpleAir \
    --adast_ckpt checkpoints/AdaST/AdaST_PurpleAir_50_336_336_best_val_MAE.pt \
    --stae_ckpt  checkpoints/STAEformer/STAEformer_PurpleAir_50_336_336_best_val_MAE.pt \
    --noise_levels 0.0 0.5 1.0 2.0 \
    --noise_types spatial temporal both \
    --gpus 0 \
    --output_json experiments_rebuttal/results_exp1_noise_PurpleAir.json

# PEMS04 (spatial-dominated dataset)
python experiments_rebuttal/exp1_noise_robustness.py \
    --dataset PEMS04 \
    --adast_ckpt checkpoints/AdaST/AdaST_PEMS04_50_12_12_best_val_MAE.pt \
    --stae_ckpt  checkpoints/STAEformer/STAEformer_PEMS04_50_12_12_best_val_MAE.pt \
    --noise_levels 0.0 0.5 1.0 2.0 \
    --gpus 0 \
    --output_json experiments_rebuttal/results_exp1_noise_PEMS04.json
```

**Finding to report**: Relative MAE degradation under noise:
| Dataset   | Noise   | AdaST Δ% | STAEformer Δ% | Interpretation          |
|-----------|---------|-----------|----------------|-------------------------|
| PurpleAir | spatial | small     | large          | Low spatial gate helps  |
| PEMS04    | temporal| small     | large          | Low temporal gate helps |

---

## Experiment 2: Alpha Hyperparameter Sensitivity

**Addresses**: Reviewer cknJ W1/Q2, Reviewer 1Uz9 W3
> "α is fixed at 0.1 without justification. Sensitivity studies on α and
>  the number of layers L are needed."

### Quick sanity check (synthetic data, fast, ~10 min):
```bash
python experiments_rebuttal/exp2_alpha_sensitivity.py \
    --mode quick \
    --alpha_values 0.0 0.01 0.05 0.1 0.2 0.5 1.0 \
    --layer_values 1 2 3 4 5 \
    --gpus 0
```

### Full sweep (real datasets, ~several hours per config):
```bash
# Step 1: Generate config files and shell script
python experiments_rebuttal/exp2_alpha_sensitivity.py \
    --mode full \
    --dataset PEMS07 \
    --alpha_values 0.0 0.01 0.05 0.1 0.2 0.5 1.0 \
    --layer_values 1 2 3 4 5 \
    --epochs 50 \
    --gpus 0

# Step 2: Run all configs
bash experiments_rebuttal/run_alpha_sweep.sh
```

**Expected finding**: MAE is relatively stable for α ∈ [0.05, 0.2],
confirming robustness to the hyperparameter choice. α=0.1 sits in this
stable range, justifying the default.

---

## Experiment 3: Weather Dataset Evaluation

**Addresses**: Reviewer CBef Q2, Reviewer cknJ W3
> "I expect more experimental datasets except the traffic ones. For example,
>  weather or electricity data."

**Models**: AdaST, STAEformer, DLinear

```bash
# Option A: Launch training directly via Python
python experiments_rebuttal/exp3_weather_baselines.py \
    --models AdaST STAEformer DLinear \
    --epochs 50 \
    --gpus 0 \
    --output_json experiments_rebuttal/results_exp3_weather.json

# Option B: Generate individual .py config files for manual runs
python experiments_rebuttal/exp3_weather_baselines.py \
    --generate_script_only \
    --models AdaST STAEformer DLinear \
    --epochs 50 \
    --gpus 0

bash experiments_rebuttal/run_weather_exp.sh
```

**Alternative**: The existing `baselines/AdaST/Weather.py` can be run directly:
```bash
python experiments/train.py -c baselines/AdaST/Weather.py -g 0
python experiments/train.py -c baselines/STAEformer/Weather.py -g 0    # if exists
python experiments/train.py -c baselines/DLinear/Weather.py -g 0
```

---

## Experiment 4: Dimension-Controlled Ablation

**Addresses**: Reviewer cknJ Q1
> "In the ablation study, removing one expert (w/o Eh) will reduce the
>  dimensionality of H(·). How do you ensure the performance drop is caused
>  by removing the expert rather than by dimension change?"

**Design**: When removing expert Ek (dim=Dk), REPLACE it with an EXTRA COPY
of another expert (independent weights, same Dk), keeping model_dim = DH constant.

```bash
# Step 1: Verify dimension consistency (sanity check)
python experiments_rebuttal/exp4_dim_controlled_ablation.py --mode verify

# Step 2: Generate run scripts
python experiments_rebuttal/exp4_dim_controlled_ablation.py \
    --mode generate \
    --datasets PEMS07 PurpleAir \
    --ablations none tod dow spatial adaptive \
    --epochs 50 \
    --gpus 0

# Step 3: Run
bash experiments_rebuttal/run_dim_ablation.sh

# Or train directly:
python experiments_rebuttal/exp4_dim_controlled_ablation.py \
    --mode train \
    --datasets PEMS07 PurpleAir \
    --ablations none tod dow spatial adaptive \
    --epochs 50 \
    --gpus 0
```

**Expected result table** (compare with original Table 3):

| Variant              | PEMS07 MAE | PurpleAir MAE | Interpretation                    |
|----------------------|------------|----------------|-----------------------------------|
| Full AdaST           | 19.16      | 0.489          | baseline                          |
| w/o Eh (std ablat.)  | 20.39      | 0.495          | includes dim reduction effect     |
| w/o Eh (dim-ctrl)    | ~20.2      | ~0.493         | pure information effect (smaller Δ)|
| w/o Ew (std ablat.)  | 19.21      | 0.498          |                                   |
| w/o Ew (dim-ctrl)    | ~19.1      | ~0.495         |                                   |

The key claim: **dimension-controlled ablation still shows degradation**,
confirming the drop is due to INFORMATION loss, not capacity reduction.

---

## Summary of Results Files

| Experiment | Output File                                    |
|-----------|------------------------------------------------|
| Exp 1     | `results_exp1_noise_{dataset}.json`            |
| Exp 2     | `results_exp2_alpha.json`                      |
| Exp 3     | `results_exp3_weather.json`                    |
| Exp 4     | `results_exp4_dim_ablation.json`               |

All results files use the format:
```json
{
  "ModelName": {
    "metric_name": value,
    ...
  }
}
```
