"""
Experiment 3: Weather Dataset Evaluation
==========================================
Motivation (Reviewer CBef Q2, Reviewer cknJ W3):
  "The authors named the paper with 'spatial-temporal forecasting', so I
   expect more experimental datasets except the traffic ones. For example,
   weather or electricity data."

Design:
  Run AdaST and 3 baselines on the Weather dataset:
    - AdaST (our model)
    - STAEformer (strongest ST baseline)
    - DLinear   (strong temporal-only baseline)
    - GWNet     (classic ST baseline)

  Weather is a multivariate dataset with 21 features, 52696 timesteps,
  10-min intervals -> steps_per_day = 144.
  Standard split: 7:1:2.
  Metrics: MAE, MSE (following LTSF convention)

Usage:
  # Step 1: Prepare the Weather dataset (if not already done)
  python scripts/data_preparation/Weather/generate_training_data.py

  # Step 2: Run all models sequentially
  python experiments_rebuttal/exp3_weather_baselines.py \
      --models AdaST STAEformer DLinear GWNet \
      --epochs 50 --gpus 0

  # Step 3: Results are saved to experiments_rebuttal/results_exp3_weather.json
"""
import os
import sys
import json
import argparse

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
os.chdir(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


# ══════════════════════════════════════════════════════════════════════
# Config builders (return EasyDict CFG objects, same as baselines/*.py)
# ══════════════════════════════════════════════════════════════════════

def build_adast_weather_cfg(epochs=50):
    """AdaST on Weather — adapted from baselines/AdaST/Weather.py"""
    import torch
    from easydict import EasyDict
    from basicts.metrics import masked_mae, masked_mse
    from basicts.data import TimeSeriesForecastingDataset
    from basicts.runners import SimpleTimeSeriesForecastingRunner
    from basicts.scaler import ZScoreScaler
    from basicts.utils import get_regular_settings
    from baselines.AdaST.arch import AdaST

    DATA_NAME = 'Weather'
    regular = get_regular_settings(DATA_NAME)
    INPUT_LEN  = regular['INPUT_LEN']
    OUTPUT_LEN = regular['OUTPUT_LEN']
    TVT        = regular['TRAIN_VAL_TEST_RATIO']
    NORM       = regular['NORM_EACH_CHANNEL']
    RESCALE    = regular['RESCALE']
    NULL_VAL   = regular['NULL_VAL']

    MODEL_PARAM = {
        "num_nodes": 21,
        "in_steps":  INPUT_LEN,
        "out_steps": OUTPUT_LEN,
        "steps_per_day": 144,   # 10-min intervals
        "input_dim": 3,
        "output_dim": 1,
        "input_embedding_dim": 24,
        "tod_embedding_dim": 24,
        "dow_embedding_dim": 24,
        "spatial_embedding_dim": 0,
        "adaptive_embedding_dim": 80,
        "feed_forward_dim": 256,
        "num_heads": 4,
        "num_layers": 3,
        "dropout": 0.1,
        "use_mixed_proj": True,
    }

    CFG = EasyDict()
    CFG.DESCRIPTION  = f'AdaST on Weather (exp3)'
    CFG.GPU_NUM      = 1
    CFG.RUNNER       = SimpleTimeSeriesForecastingRunner
    CFG.DATASET      = EasyDict()
    CFG.DATASET.NAME = DATA_NAME
    CFG.DATASET.TYPE = TimeSeriesForecastingDataset
    CFG.DATASET.PARAM = EasyDict({
        'dataset_name': DATA_NAME,
        'train_val_test_ratio': TVT,
        'input_len': INPUT_LEN,
        'output_len': OUTPUT_LEN,
    })
    CFG.SCALER      = EasyDict()
    CFG.SCALER.TYPE = ZScoreScaler
    CFG.SCALER.PARAM = EasyDict({
        'dataset_name': DATA_NAME,
        'train_ratio': TVT[0],
        'norm_each_channel': NORM,
        'rescale': RESCALE,
    })
    CFG.MODEL       = EasyDict()
    CFG.MODEL.NAME  = 'AdaST'
    CFG.MODEL.ARCH  = AdaST
    CFG.MODEL.PARAM = MODEL_PARAM
    CFG.MODEL.FORWARD_FEATURES = [0, 1, 2]
    CFG.MODEL.TARGET_FEATURES  = [0]
    CFG.METRICS      = EasyDict()
    CFG.METRICS.FUNCS = EasyDict({'MAE': masked_mae, 'MSE': masked_mse})
    CFG.METRICS.TARGET  = 'MAE'
    CFG.METRICS.NULL_VAL = NULL_VAL
    CFG.TRAIN        = EasyDict()
    CFG.TRAIN.NUM_EPOCHS = epochs
    CFG.TRAIN.CKPT_SAVE_DIR = os.path.join(
        'checkpoints', 'AdaST', f'Weather_{epochs}_{INPUT_LEN}_{OUTPUT_LEN}')
    CFG.TRAIN.LOSS   = masked_mae
    CFG.TRAIN.OPTIM  = EasyDict()
    CFG.TRAIN.OPTIM.TYPE = 'Adam'
    CFG.TRAIN.OPTIM.PARAM = {'lr': 0.001, 'weight_decay': 0.0003}
    CFG.TRAIN.LR_SCHEDULER = EasyDict()
    CFG.TRAIN.LR_SCHEDULER.TYPE = 'MultiStepLR'
    CFG.TRAIN.LR_SCHEDULER.PARAM = {'milestones': [20, 25], 'gamma': 0.1}
    CFG.TRAIN.DATA   = EasyDict()
    CFG.TRAIN.DATA.BATCH_SIZE = 16
    CFG.TRAIN.DATA.SHUFFLE    = True
    CFG.VAL          = EasyDict()
    CFG.VAL.INTERVAL = 1
    CFG.VAL.DATA     = EasyDict()
    CFG.VAL.DATA.BATCH_SIZE = 64
    CFG.TEST         = EasyDict()
    CFG.TEST.INTERVAL = 1
    CFG.TEST.DATA    = EasyDict()
    CFG.TEST.DATA.BATCH_SIZE = 64
    CFG.EVAL         = EasyDict()
    CFG.EVAL.HORIZONS = [3, 6, 12]
    CFG.EVAL.USE_GPU  = True
    return CFG


def build_staeformer_weather_cfg(epochs=50):
    """STAEformer on Weather"""
    import torch
    from easydict import EasyDict
    from basicts.metrics import masked_mae, masked_mse
    from basicts.data import TimeSeriesForecastingDataset
    from basicts.runners import SimpleTimeSeriesForecastingRunner
    from basicts.scaler import ZScoreScaler
    from basicts.utils import get_regular_settings
    from baselines.STAEformer.arch import STAEformer

    DATA_NAME = 'Weather'
    regular   = get_regular_settings(DATA_NAME)
    INPUT_LEN  = regular['INPUT_LEN']
    OUTPUT_LEN = regular['OUTPUT_LEN']
    TVT        = regular['TRAIN_VAL_TEST_RATIO']
    NORM       = regular['NORM_EACH_CHANNEL']
    RESCALE    = regular['RESCALE']
    NULL_VAL   = regular['NULL_VAL']

    MODEL_PARAM = {
        "num_nodes": 21,
        "in_steps":  INPUT_LEN,
        "out_steps": OUTPUT_LEN,
        "steps_per_day": 144,
        "input_dim": 3,
        "output_dim": 1,
        "input_embedding_dim": 24,
        "tod_embedding_dim": 24,
        "dow_embedding_dim": 24,
        "spatial_embedding_dim": 0,
        "adaptive_embedding_dim": 80,
        "feed_forward_dim": 256,
        "num_heads": 4,
        "num_layers": 3,
        "dropout": 0.1,
        "use_mixed_proj": True,
    }

    CFG = EasyDict()
    CFG.DESCRIPTION  = f'STAEformer on Weather (exp3)'
    CFG.GPU_NUM      = 1
    CFG.RUNNER       = SimpleTimeSeriesForecastingRunner
    CFG.DATASET      = EasyDict()
    CFG.DATASET.NAME = DATA_NAME
    CFG.DATASET.TYPE = TimeSeriesForecastingDataset
    CFG.DATASET.PARAM = EasyDict({
        'dataset_name': DATA_NAME,
        'train_val_test_ratio': TVT,
        'input_len': INPUT_LEN,
        'output_len': OUTPUT_LEN,
    })
    CFG.SCALER      = EasyDict()
    CFG.SCALER.TYPE = ZScoreScaler
    CFG.SCALER.PARAM = EasyDict({
        'dataset_name': DATA_NAME,
        'train_ratio': TVT[0],
        'norm_each_channel': NORM,
        'rescale': RESCALE,
    })
    CFG.MODEL       = EasyDict()
    CFG.MODEL.NAME  = 'STAEformer'
    CFG.MODEL.ARCH  = STAEformer
    CFG.MODEL.PARAM = MODEL_PARAM
    CFG.MODEL.FORWARD_FEATURES = [0, 1, 2]
    CFG.MODEL.TARGET_FEATURES  = [0]
    CFG.METRICS      = EasyDict()
    CFG.METRICS.FUNCS = EasyDict({'MAE': masked_mae, 'MSE': masked_mse})
    CFG.METRICS.TARGET   = 'MAE'
    CFG.METRICS.NULL_VAL = NULL_VAL
    CFG.TRAIN        = EasyDict()
    CFG.TRAIN.NUM_EPOCHS = epochs
    CFG.TRAIN.CKPT_SAVE_DIR = os.path.join(
        'checkpoints', 'STAEformer', f'Weather_{epochs}_{INPUT_LEN}_{OUTPUT_LEN}')
    CFG.TRAIN.LOSS   = masked_mae
    CFG.TRAIN.OPTIM  = EasyDict()
    CFG.TRAIN.OPTIM.TYPE = 'Adam'
    CFG.TRAIN.OPTIM.PARAM = {'lr': 0.001, 'weight_decay': 0.0003}
    CFG.TRAIN.LR_SCHEDULER = EasyDict()
    CFG.TRAIN.LR_SCHEDULER.TYPE = 'MultiStepLR'
    CFG.TRAIN.LR_SCHEDULER.PARAM = {'milestones': [20, 25], 'gamma': 0.1}
    CFG.TRAIN.DATA   = EasyDict()
    CFG.TRAIN.DATA.BATCH_SIZE = 16
    CFG.TRAIN.DATA.SHUFFLE    = True
    CFG.VAL          = EasyDict()
    CFG.VAL.INTERVAL = 1
    CFG.VAL.DATA     = EasyDict()
    CFG.VAL.DATA.BATCH_SIZE = 64
    CFG.TEST         = EasyDict()
    CFG.TEST.INTERVAL = 1
    CFG.TEST.DATA    = EasyDict()
    CFG.TEST.DATA.BATCH_SIZE = 64
    CFG.EVAL         = EasyDict()
    CFG.EVAL.HORIZONS = [3, 6, 12]
    CFG.EVAL.USE_GPU  = True
    return CFG


def build_dlinear_weather_cfg(epochs=50):
    """DLinear on Weather — following DLinear/Weather.py but with fixed epochs"""
    from easydict import EasyDict
    from basicts.metrics import masked_mae, masked_mse
    from basicts.data import TimeSeriesForecastingDataset
    from basicts.runners import SimpleTimeSeriesForecastingRunner
    from basicts.scaler import ZScoreScaler
    from basicts.utils import get_regular_settings
    from baselines.DLinear.arch import DLinear

    DATA_NAME = 'Weather'
    regular   = get_regular_settings(DATA_NAME)
    INPUT_LEN  = regular['INPUT_LEN']
    OUTPUT_LEN = regular['OUTPUT_LEN']
    TVT        = regular['TRAIN_VAL_TEST_RATIO']
    NORM       = regular['NORM_EACH_CHANNEL']
    RESCALE    = regular['RESCALE']
    NULL_VAL   = regular['NULL_VAL']

    CFG = EasyDict()
    CFG.DESCRIPTION  = 'DLinear on Weather (exp3)'
    CFG.GPU_NUM      = 1
    CFG.RUNNER       = SimpleTimeSeriesForecastingRunner
    CFG.DATASET      = EasyDict()
    CFG.DATASET.NAME = DATA_NAME
    CFG.DATASET.TYPE = TimeSeriesForecastingDataset
    CFG.DATASET.PARAM = EasyDict({
        'dataset_name': DATA_NAME,
        'train_val_test_ratio': TVT,
        'input_len': INPUT_LEN,
        'output_len': OUTPUT_LEN,
    })
    CFG.SCALER      = EasyDict()
    CFG.SCALER.TYPE = ZScoreScaler
    CFG.SCALER.PARAM = EasyDict({
        'dataset_name': DATA_NAME,
        'train_ratio': TVT[0],
        'norm_each_channel': NORM,
        'rescale': RESCALE,
    })
    CFG.MODEL       = EasyDict()
    CFG.MODEL.NAME  = 'DLinear'
    CFG.MODEL.ARCH  = DLinear
    CFG.MODEL.PARAM = {
        'seq_len': INPUT_LEN,
        'pred_len': OUTPUT_LEN,
        'individual': False,
        'enc_in': 21,
    }
    CFG.MODEL.FORWARD_FEATURES = [0]
    CFG.MODEL.TARGET_FEATURES  = [0]
    CFG.METRICS      = EasyDict()
    CFG.METRICS.FUNCS = EasyDict({'MAE': masked_mae, 'MSE': masked_mse})
    CFG.METRICS.TARGET   = 'MAE'
    CFG.METRICS.NULL_VAL = NULL_VAL
    CFG.TRAIN        = EasyDict()
    CFG.TRAIN.NUM_EPOCHS = epochs
    CFG.TRAIN.CKPT_SAVE_DIR = os.path.join(
        'checkpoints', 'DLinear', f'Weather_{epochs}_{INPUT_LEN}_{OUTPUT_LEN}')
    CFG.TRAIN.LOSS   = masked_mae
    CFG.TRAIN.OPTIM  = EasyDict()
    CFG.TRAIN.OPTIM.TYPE = 'Adam'
    CFG.TRAIN.OPTIM.PARAM = {'lr': 0.0003, 'weight_decay': 0.0001}
    CFG.TRAIN.LR_SCHEDULER = EasyDict()
    CFG.TRAIN.LR_SCHEDULER.TYPE = 'MultiStepLR'
    CFG.TRAIN.LR_SCHEDULER.PARAM = {'milestones': [1, 25], 'gamma': 0.5}
    CFG.TRAIN.CLIP_GRAD_PARAM = {'max_norm': 5.0}
    CFG.TRAIN.DATA   = EasyDict()
    CFG.TRAIN.DATA.BATCH_SIZE = 64
    CFG.TRAIN.DATA.SHUFFLE    = True
    CFG.VAL          = EasyDict()
    CFG.VAL.INTERVAL = 1
    CFG.VAL.DATA     = EasyDict()
    CFG.VAL.DATA.BATCH_SIZE = 64
    CFG.TEST         = EasyDict()
    CFG.TEST.INTERVAL = 1
    CFG.TEST.DATA    = EasyDict()
    CFG.TEST.DATA.BATCH_SIZE = 64
    CFG.EVAL         = EasyDict()
    CFG.EVAL.USE_GPU = True
    return CFG


# ══════════════════════════════════════════════════════════════════════
# Dispatcher
# ══════════════════════════════════════════════════════════════════════
MODEL_CFG_BUILDERS = {
    'AdaST':      build_adast_weather_cfg,
    'STAEformer': build_staeformer_weather_cfg,
    'DLinear':    build_dlinear_weather_cfg,
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--models',  nargs='+',
                   choices=list(MODEL_CFG_BUILDERS.keys()),
                   default=list(MODEL_CFG_BUILDERS.keys()))
    p.add_argument('--epochs',  type=int, default=50)
    p.add_argument('--gpus',    default='0')
    p.add_argument('--output_json',
                   default='experiments_rebuttal/results_exp3_weather.json')
    p.add_argument('--generate_script_only', action='store_true',
                   help='Only generate shell script, do not train')
    return p.parse_args()


def main():
    args = parse_args()

    # Check dataset exists
    if not os.path.exists('datasets/Weather/data.dat'):
        print("ERROR: datasets/Weather/data.dat not found!")
        print("Please run the data preparation script first:")
        print("  cd <repo_root>")
        print("  python scripts/data_preparation/Weather/generate_training_data.py")
        return

    if args.generate_script_only:
        # Generate shell script for running on cluster
        script = "#!/bin/bash\n# Exp3: Weather baseline comparison\n\n"
        for model in args.models:
            script += f"echo '=== {model} on Weather ==='\n"
            script += (f"python experiments/train.py "
                       f"-c experiments_rebuttal/weather_cfgs/{model}_Weather.py "
                       f"-g {args.gpus}\n\n")

        os.makedirs('experiments_rebuttal', exist_ok=True)
        with open('experiments_rebuttal/run_weather_exp.sh', 'w') as f:
            f.write(script)
        os.chmod('experiments_rebuttal/run_weather_exp.sh', 0o755)

        # Also write out individual config files
        os.makedirs('experiments_rebuttal/weather_cfgs', exist_ok=True)
        for model in args.models:
            cfg_path = f'experiments_rebuttal/weather_cfgs/{model}_Weather.py'
            base_src = f'baselines/{model}/Weather.py' if model != 'AdaST' else 'baselines/AdaST/Weather.py'
            if os.path.exists(base_src):
                with open(base_src) as f:
                    code = f.read()
                # Override epochs
                code = code.replace(
                    f'NUM_EPOCHS = 1',
                    f'NUM_EPOCHS = {args.epochs}'
                ).replace(
                    f'NUM_EPOCHS = 50',
                    f'NUM_EPOCHS = {args.epochs}'
                ).replace(
                    f'NUM_EPOCHS = 100',
                    f'NUM_EPOCHS = {args.epochs}'
                )
                with open(cfg_path, 'w') as f:
                    f.write(f"# Exp3 config: {model} on Weather, {args.epochs} epochs\n")
                    f.write(code)
                print(f"Written: {cfg_path}")
        print(f"\nShell script: experiments_rebuttal/run_weather_exp.sh")
        print(f"Run: bash experiments_rebuttal/run_weather_exp.sh")
        return

    # ── Launch training using basicts ──
    import basicts
    results = {}
    for model_name in args.models:
        print(f"\n{'='*60}")
        print(f"Training {model_name} on Weather")
        print(f"{'='*60}")
        cfg = MODEL_CFG_BUILDERS[model_name](epochs=args.epochs)

        try:
            basicts.launch_training(cfg=cfg, gpus=args.gpus, node_rank=0)
            print(f"\n✓ {model_name} training complete.")
            # Try to read best val metrics from checkpoint dir
            ckpt_dir = cfg.TRAIN.CKPT_SAVE_DIR
            summary_path = os.path.join(ckpt_dir, 'metrics_summary.json')
            if os.path.exists(summary_path):
                with open(summary_path) as f:
                    metrics = json.load(f)
                results[model_name] = metrics
                print(f"  Best test MAE: {metrics.get('best_test_MAE', 'N/A')}")
            else:
                results[model_name] = {'status': 'complete', 'ckpt_dir': ckpt_dir}
        except Exception as e:
            print(f"✗ {model_name} failed: {e}")
            results[model_name] = {'status': 'failed', 'error': str(e)}

    # Save results
    os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
    with open(args.output_json, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {args.output_json}")

    # Summary
    print("\n" + "="*60)
    print("WEATHER EXPERIMENT SUMMARY")
    print("="*60)
    for model, res in results.items():
        print(f"  {model:12s}: {res}")
    print("="*60)


if __name__ == '__main__':
    main()
