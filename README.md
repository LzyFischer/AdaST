# AdaST

**AdaST** is a spatio-temporal forecasting model that adaptively balances *temporal*, *spatial*,
and *joint spatio-temporal* modeling for each input. It is implemented on top of the
[BasicTS](https://github.com/GestaltCogTeam/BasicTS) benchmark framework, so training,
evaluation, data loading and all baselines share the same pipeline.

## Model overview

The implementation lives in [`baselines/AdaST/arch/adast_arch.py`](baselines/AdaST/arch/adast_arch.py).

1. **Embedding experts.** The input series is projected and concatenated with four
   context embeddings: time-of-day, day-of-week, a per-node spatial embedding, and a learnable
   adaptive (time × node) embedding. Each expert can be switched off by setting its dimension to `0`.
2. **Three-way projection.** Every expert is linearly split into three views that feed three branches:
   - **ST branch** – temporal self-attention followed by a spatial mixer (joint modeling),
   - **T branch** – temporal self-attention only,
   - **S branch** – spatial mixer only (a learnable dense `N × N` node-mixing matrix).
3. **Correlation-aware aggregation.** A gating network scores the three branch outputs; the
   gates are modulated by the measured temporal/spatial cosine self-similarity of each branch
   and softmax-normalised, so the model leans on whichever view the current sample favours.
4. **Decoder.** The aggregated representation is flattened over the input window and mapped
   to all output steps at once (`use_mixed_proj=True`).

## Repository layout

```
AdaST/
├── baselines/
│   ├── AdaST/                  # the proposed model
│   │   ├── arch/
│   │   │   ├── adast_arch.py       # AdaST (main model)
│   │   │   └── adast_ablation.py   # AdaST with ablation switches
│   │   ├── <DATASET>.py            # main configs, one per dataset
│   │   ├── ablation/               # ablation configs (PEMS03/04/07/08, PurpleAir)
│   │   └── analysis/               # configs that dump embeddings & gate weights
│   └── <Baseline>/             # 20 comparison baselines (STAEformer, STID, GWNet, …)
├── basicts/                    # BasicTS framework (runners, data, scalers, metrics)
├── experiments/                # train.py / evaluate.py / inference.py entry points
├── scripts/
│   ├── data_preparation/       # raw data → datasets/<NAME>/ converters
│   └── data_visualization/     # notebooks for data, prediction and gate visualisation
├── datasets/                   # dataset storage (see datasets/README.md)
├── docs/                       # BasicTS design docs (config, runner, dataset, …)
├── examples/                   # annotated BasicTS config templates
└── tests/                      # unit tests for the framework
```

Baselines kept for comparison: AGCRN, CATS, D2STGNN, DCRNN, DeepAR, DLinear, GTS, GWNet, HI,
HimNet, MTGNN, NBeats, PatchTST, STAEformer, STDN, STGCN, STID, STNorm, StemGNN, TimeMixer.

## Installation

Python 3.9+ and a CUDA build of PyTorch are recommended.

```bash
# 1. Install PyTorch for your CUDA version, e.g.
pip install torch --index-url https://download.pytorch.org/whl/cu121
# 2. Install the remaining dependencies
pip install -r requirements.txt
```

Some baselines need extra packages (e.g. `torch_geometric`); install them only if you run those models.

## Data preparation

1. Download the raw data following [`datasets/README.md`](datasets/README.md) and put it under
   `datasets/raw_data/<DATASET>/`.
   For **PurpleAir**, place `PurpleAir.csv` (timestamp index × sensor columns) in
   `datasets/raw_data/PurpleAir/`, and build the adjacency matrix with
   `scripts/data_preparation/PurpleAir/generate_adj_mx.py`.
2. Convert the raw data into the BasicTS format:

```bash
python scripts/data_preparation/PEMS08/generate_training_data.py
# or all supported datasets at once
bash scripts/data_preparation/run.sh
```

Supported datasets: PEMS03, PEMS04, PEMS07, PEMS08, METR-LA, PEMS-BAY, PurpleAir,
BeijingAirQuality, ETTh1/ETTh2/ETTm1/ETTm2, Electricity, Weather, ExchangeRate, Illness, Pulse.

## Usage

All commands are run from the repository root.

### Train

```bash
python experiments/train.py -c baselines/AdaST/PEMS08.py -g 0
```

Checkpoints and logs are written to `checkpoints/AdaST/<DATASET>_<EPOCHS>_<IN>_<OUT>/<hash>/`.
The same command trains any baseline, e.g. `-c baselines/STAEformer/PEMS08.py`.

### Evaluate a checkpoint

```bash
python experiments/evaluate.py \
    -cfg baselines/AdaST/PEMS08.py \
    -ckpt checkpoints/AdaST/PEMS08_50_12_12/<hash>/AdaST_best_val_MAE.pt \
    -g 0
```

### Ablation studies

Configs in `baselines/AdaST/ablation/` use `adast_ablation.py`, which exposes extra switches:

| Config suffix             | What changes                                                   |
|---------------------------|----------------------------------------------------------------|
| `wo_tod` / `wo_dow`       | drop the time-of-day / day-of-week expert                      |
| `wo_spatial` / `wo_adaptive` | drop the node / adaptive embedding expert                   |
| `wo_all_experts`          | drop all four context embeddings                               |
| `wo_correlation`          | gated aggregation without correlation modulation               |
| `simple_average`          | average the three branches                                     |
| `correlation_average`     | weight branches by correlation only (no learned gate)          |
| `spatial_attention`       | replace the spatial mixer with spatial self-attention          |

```bash
python experiments/train.py -c baselines/AdaST/ablation/PEMS08_wo_correlation.py -g 0
```

### Embedding and gate analysis

Configs in `baselines/AdaST/analysis/` use `EmbTimeSeriesForecastingRunner`, which saves the
ST/S/T branch embeddings, gate weights and correlation features during testing. The notebooks in
`scripts/data_visualization/` visualise them.

```bash
python experiments/train.py -c baselines/AdaST/analysis/PEMS08_emb.py -g 0
```

## Key hyper-parameters

Set in `MODEL_PARAM` of each config:

| Parameter | Meaning | Default |
|---|---|---|
| `input_embedding_dim` | projection size of the raw input | 24 |
| `tod_embedding_dim`, `dow_embedding_dim` | time-of-day / day-of-week expert size (0 = off) | 24 |
| `spatial_embedding_dim` | per-node embedding size (0 = off) | 24 |
| `adaptive_embedding_dim` | adaptive time×node embedding size (0 = off) | 80 |
| `num_layers`, `num_heads`, `feed_forward_dim` | depth / attention heads / FFN width | 3 / 4 / 256 |
| `steps_per_day` | number of time steps per day of the dataset | 288 for PEMS |

## Framework documentation

AdaST uses BasicTS unchanged apart from one extra runner
(`basicts/runners/runner_zoo/emb_tsf_runner.py`). For details on configs, runners, datasets,
scalers and metrics, see [`docs/`](docs/getting_started.md) and the annotated
[`examples/complete_config.py`](examples/complete_config.py).

## Acknowledgements

This code is built on [BasicTS](https://github.com/GestaltCogTeam/BasicTS) (Apache-2.0).
The baseline implementations are adapted from BasicTS and the original authors' releases.

## License

Apache-2.0, see [LICENSE](LICENSE).
