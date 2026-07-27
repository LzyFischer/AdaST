"""
Experiment 2: Alpha Hyperparameter Sensitivity Analysis
=========================================================
Motivation (Reviewer cknJ W1/Q2, Reviewer 1Uz9 W3):
  "α is fixed at 0.1 without justification. Sensitivity studies on α
   and the number of layers L are needed."

Design:
  Modify the CombineCorrelationAggregation to accept alpha as a parameter.
  Train AdaST from scratch with different alpha values and report test MAE.

  Alpha sweep:  [0.0, 0.01, 0.05, 0.1, 0.2, 0.5, 1.0]
    alpha=0.0 means correlation has NO influence (gate_net only)
    alpha=1.0 means linear correlation weighting

  L sweep:      [1, 2, 3, 4, 5]

  This script patches the arch on-the-fly (no need to edit arch file),
  trains a mini version for quick sensitivity check, then launches
  full training on real data using the basicts framework.

Usage:
  # Quick mode (small synthetic data, fast):
  python experiments_rebuttal/exp2_alpha_sensitivity.py \
      --mode quick --dataset PEMS04 --gpus 0

  # Full mode (real dataset, full epochs):
  python experiments_rebuttal/exp2_alpha_sensitivity.py \
      --mode full --dataset PEMS04 --alpha_values 0.01 0.05 0.1 0.2 0.5 \
      --layer_values 2 3 4 --epochs 50 --gpus 0
"""
import os
import sys
import json
import copy
import argparse
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
os.chdir(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


# ══════════════════════════════════════════════════════════════════════
# Step 1: Patch AdaST arch to accept alpha
# ══════════════════════════════════════════════════════════════════════
def patch_adast_with_alpha():
    """
    Monkey-patches the CombineCorrelationAggregation and AdaST classes
    so that alpha becomes an explicit constructor argument.

    Call this ONCE before importing the arch.
    """
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class CombineCorrelationAggregationAlpha(nn.Module):
        """
        Same as original CombineCorrelationAggregation but with
        configurable alpha (power applied to correlation features).
        Paper eq. 19: g_tilde = g * C^alpha
        """
        def __init__(self, model_dim: int, alpha: float = 0.1, reduction: int = 4):
            super().__init__()
            self.alpha = alpha
            hidden = max(model_dim // reduction, 8)
            self.gate_net = nn.Sequential(
                nn.Linear(model_dim * 3, hidden),
                nn.ReLU(inplace=True),
                nn.Linear(hidden, 3)
            )

        def forward(self, x, x_t, x_s):
            B, T, S, D = x.shape

            # Temporal correlation of x_t branch
            x_t_reshaped = x_t.permute(0, 2, 1, 3).reshape(-1, x_t.shape[1], x_t.shape[3])
            x_t_norm = F.normalize(x_t_reshaped, p=2, dim=-1)
            similarity_t = torch.sum(x_t_norm[:, :-1, :] * x_t_norm[:, 1:, :], dim=-1)
            corr_x_t = similarity_t.reshape(B, -1).mean(dim=-1)

            # Spatial correlation of x_s branch
            x_s_reshaped = x_s.permute(0, 1, 2, 3).reshape(-1, x_s.shape[2], x_s.shape[3])
            x_s_norm = F.normalize(x_s_reshaped, p=2, dim=-1)
            similarity_s = torch.sum(x_s_norm[:, :-1, :] * x_s_norm[:, 1:, :], dim=-1)
            corr_x_s = similarity_s.reshape(B, -1).mean(dim=-1)

            # Mixed correlation of x (ST branch)
            x_r1 = x.permute(0, 2, 1, 3).reshape(-1, x.shape[2], x.shape[3])
            x_n1 = F.normalize(x_r1, p=2, dim=-1)
            sim_xt = torch.sum(x_n1[:, :-1, :] * x_n1[:, 1:, :], dim=-1)
            x_r2 = x.permute(0, 1, 2, 3).reshape(-1, x.shape[2], x.shape[3])
            x_n2 = F.normalize(x_r2, p=2, dim=-1)
            sim_xs = torch.sum(x_n2[:, :-1, :] * x_n2[:, 1:, :], dim=-1)
            corr_x = (sim_xt.reshape(B, -1).mean(dim=-1) +
                      sim_xs.reshape(B, -1).mean(dim=-1)) / 2.0

            corr_features = torch.stack([corr_x, corr_x_t, corr_x_s], dim=-1)  # (B, 3)

            # Gate computation
            combined = torch.cat([x, x_t, x_s], dim=-1)  # (B, T, N, 3D)
            gates = self.gate_net(combined)               # (B, T, N, 3)

            # Apply alpha: g_tilde = g * C^alpha
            if self.alpha != 0.0:
                corr_mod = corr_features.view(B, 1, 1, -1).clamp(min=1e-8) ** self.alpha
            else:
                corr_mod = torch.ones_like(corr_features.view(B, 1, 1, -1))

            gates = gates * corr_mod
            gates = torch.softmax(gates, dim=-1).unsqueeze(-2)  # (B, T, N, 1, 3)

            stacked = torch.stack([x, x_t, x_s], dim=-1)        # (B, T, N, D, 3)
            out = (stacked * gates).sum(dim=-1)
            return out, gates.squeeze(-2), corr_features

    return CombineCorrelationAggregationAlpha


# ══════════════════════════════════════════════════════════════════════
# Step 2: Build AdaST with alpha parameter
# ══════════════════════════════════════════════════════════════════════
def build_adast_model_with_alpha(model_params: dict, alpha: float,
                                  num_layers: int = None) -> 'nn.Module':
    """
    Imports AdaST and replaces its aggregation module with alpha-aware version.
    """
    import torch
    import baselines.AdaST.arch.adast_arch as arch_module

    AlphaAggClass = patch_adast_with_alpha()

    params = copy.deepcopy(model_params)
    if num_layers is not None:
        params['num_layers'] = num_layers

    # Build model
    from baselines.AdaST.arch import AdaST
    model = AdaST(**params)

    # Replace aggregation module
    model.aggregation = AlphaAggClass(
        model_dim=model.aggregation.gate_net[0].in_features // 3,
        alpha=alpha
    )
    return model


# ══════════════════════════════════════════════════════════════════════
# Step 3a: Quick mode — train on mini data and sweep alpha/L
# ══════════════════════════════════════════════════════════════════════
def quick_sweep(alpha_values, layer_values, device_str='cpu', epochs=10,
                mini_dataset='MiniPEMS'):
    """
    Train AdaST variants on a tiny in-memory dataset.
    Useful for quickly verifying alpha/L sensitivity before full runs.
    """
    import torch
    import torch.nn as nn
    from torch.utils.data import TensorDataset, DataLoader
    from basicts.metrics import masked_mae

    print("\n" + "="*60)
    print("QUICK MODE: Synthetic mini-dataset sweep")
    print("="*60)

    # Generate tiny synthetic data
    torch.manual_seed(42)
    B_train, B_val = 500, 100
    T, N, C = 12, 20, 3
    steps_per_day = 288

    def gen_batch(n):
        # Simple AR(1) process
        x = torch.randn(n, T, N, C)
        # time of day and day of week as fractions
        x[:, :, :, 1] = torch.arange(T).float().view(1, T, 1).expand(n, T, N) / steps_per_day
        x[:, :, :, 2] = torch.zeros(n, T, N)
        y = x[:, -1:, :, 0:1] * 0.8 + torch.randn(n, T, N, 1) * 0.1
        return x, y

    X_train, Y_train = gen_batch(B_train)
    X_val, Y_val = gen_batch(B_val)

    train_loader = DataLoader(
        TensorDataset(X_train, Y_train), batch_size=32, shuffle=True)
    val_loader = DataLoader(
        TensorDataset(X_val, Y_val), batch_size=32, shuffle=False)

    device = torch.device(device_str if torch.cuda.is_available() or device_str == 'cpu' else 'cpu')

    base_model_params = dict(
        num_nodes=N, in_steps=T, out_steps=T,
        steps_per_day=steps_per_day,
        input_dim=3, output_dim=1,
        input_embedding_dim=16,
        tod_embedding_dim=16, dow_embedding_dim=16,
        spatial_embedding_dim=0, adaptive_embedding_dim=32,
        feed_forward_dim=64, num_heads=4, num_layers=3,
        dropout=0.1, use_mixed_proj=True,
    )

    results = {'alpha': {}, 'layers': {}}

    # ── Alpha sweep ──
    print("\n[Alpha Sweep]  (L=3 fixed)")
    for alpha in alpha_values:
        model = build_adast_model_with_alpha(base_model_params, alpha=alpha)
        model = model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        for ep in range(epochs):
            model.train()
            for xb, yb in train_loader:
                xb, yb = xb.to(device), yb.to(device)
                pred = model(history_data=xb, future_data=xb,
                             batch_seen=0, epoch=ep, train=True)
                if isinstance(pred, dict):
                    pred = pred['prediction']
                pred = pred[:, :, :, 0:1]
                loss = nn.L1Loss()(pred, yb)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        # Validate
        model.eval()
        val_losses = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                pred = model(history_data=xb, future_data=xb,
                             batch_seen=0, epoch=0, train=False)
                if isinstance(pred, dict):
                    pred = pred['prediction']
                pred = pred[:, :, :, 0:1]
                val_losses.append(nn.L1Loss()(pred, yb).item())
        val_mae = np.mean(val_losses)
        results['alpha'][str(alpha)] = round(val_mae, 6)
        print(f"  alpha={alpha:.3f}  =>  Val MAE = {val_mae:.6f}")

    # ── Layer sweep ──
    print("\n[Layer Sweep]  (alpha=0.1 fixed)")
    for L in layer_values:
        model = build_adast_model_with_alpha(base_model_params, alpha=0.1,
                                              num_layers=L)
        model = model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        for ep in range(epochs):
            model.train()
            for xb, yb in train_loader:
                xb, yb = xb.to(device), yb.to(device)
                pred = model(history_data=xb, future_data=xb,
                             batch_seen=0, epoch=ep, train=True)
                if isinstance(pred, dict):
                    pred = pred['prediction']
                pred = pred[:, :, :, 0:1]
                loss = nn.L1Loss()(pred, yb)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        model.eval()
        val_losses = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                pred = model(history_data=xb, future_data=xb,
                             batch_seen=0, epoch=0, train=False)
                if isinstance(pred, dict):
                    pred = pred['prediction']
                pred = pred[:, :, :, 0:1]
                val_losses.append(nn.L1Loss()(pred, yb).item())
        val_mae = np.mean(val_losses)
        results['layers'][str(L)] = round(val_mae, 6)
        print(f"  L={L}  =>  Val MAE = {val_mae:.6f}")

    return results


# ══════════════════════════════════════════════════════════════════════
# Step 3b: Full mode — generate basicts-compatible config and launch
# ══════════════════════════════════════════════════════════════════════
def generate_full_configs(dataset, alpha_values, layer_values, epochs,
                           output_dir='experiments_rebuttal/alpha_configs'):
    """
    Generates one .py config file per (alpha, L) combination,
    ready to be run with: python experiments/train.py -c <config.py>
    """
    os.makedirs(output_dir, exist_ok=True)

    # Read base config file as text and patch it
    base_cfg_path = f'baselines/AdaST/{dataset}.py'
    if not os.path.exists(base_cfg_path):
        print(f"Warning: {base_cfg_path} not found, skipping full config generation")
        return []

    with open(base_cfg_path, 'r') as f:
        base_code = f.read()

    generated = []

    # Alpha sweep
    for alpha in alpha_values:
        tag = f"alpha{str(alpha).replace('.','p')}"
        cfg_path = f"{output_dir}/{dataset}_{tag}.py"

        # We need to inject the alpha-aware aggregation into the arch
        # Strategy: add a post-build hook by subclassing in the config file
        config_code = f'''# AUTO-GENERATED by exp2_alpha_sensitivity.py
# Alpha = {alpha}, L = 3 (default)
import os, sys
sys.path.insert(0, os.path.abspath(__file__ + "/../../.."))
os.chdir(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

# ── inject alpha into aggregation ──
from experiments_rebuttal.exp2_alpha_sensitivity import patch_adast_with_alpha, build_adast_model_with_alpha
_AlphaAgg = patch_adast_with_alpha()

# ── load base config ──
exec(open("{base_cfg_path}").read())

# ── override: alpha={alpha} ──
NUM_EPOCHS = {epochs}
_alpha = {alpha}

# Wrap original arch to inject alpha
import baselines.AdaST.arch.adast_arch as _arch_module
_OrigAgg = _arch_module.CombineCorrelationAggregation

class _AdaSTWithAlpha(CFG.MODEL.ARCH):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.aggregation = _AlphaAgg(
            model_dim=self.aggregation.gate_net[0].in_features // 3,
            alpha=_alpha
        )

CFG.MODEL.ARCH = _AdaSTWithAlpha
CFG.TRAIN.NUM_EPOCHS = NUM_EPOCHS
CFG.TRAIN.CKPT_SAVE_DIR = os.path.join(
    "checkpoints", "AdaST_alpha_sweep",
    "{dataset}_{tag}_E{{}}".format(NUM_EPOCHS))
CFG.DESCRIPTION = "AdaST alpha={alpha} on {dataset}"
'''
        with open(cfg_path, 'w') as f:
            f.write(config_code)
        generated.append((f'alpha={alpha}', cfg_path))

    # Layer sweep
    for L in layer_values:
        tag = f"L{L}"
        cfg_path = f"{output_dir}/{dataset}_{tag}.py"

        config_code = f'''# AUTO-GENERATED by exp2_alpha_sensitivity.py
# Alpha = 0.1 (default), L = {L}
import os, sys
sys.path.insert(0, os.path.abspath(__file__ + "/../../.."))
os.chdir(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

exec(open("{base_cfg_path}").read())

NUM_EPOCHS = {epochs}
CFG.MODEL.PARAM["num_layers"] = {L}
CFG.TRAIN.NUM_EPOCHS = NUM_EPOCHS
CFG.TRAIN.CKPT_SAVE_DIR = os.path.join(
    "checkpoints", "AdaST_layer_sweep",
    "{dataset}_{tag}_E{{}}".format(NUM_EPOCHS))
CFG.DESCRIPTION = "AdaST L={L} on {dataset}"
'''
        with open(cfg_path, 'w') as f:
            f.write(config_code)
        generated.append((f'L={L}', cfg_path))

    return generated


# ══════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--mode',         choices=['quick', 'full'], default='quick')
    p.add_argument('--dataset',      default='PEMS04')
    p.add_argument('--alpha_values', nargs='+', type=float,
                   default=[0.0, 0.01, 0.05, 0.1, 0.2, 0.5, 1.0])
    p.add_argument('--layer_values', nargs='+', type=int,
                   default=[1, 2, 3, 4, 5])
    p.add_argument('--epochs',       type=int, default=50)
    p.add_argument('--gpus',         default='0')
    p.add_argument('--output_json',  default='experiments_rebuttal/results_exp2_alpha.json')
    return p.parse_args()


def main():
    args = parse_args()

    if args.mode == 'quick':
        device_str = f'cuda:{args.gpus}' if args.gpus else 'cpu'
        results = quick_sweep(
            alpha_values=args.alpha_values,
            layer_values=args.layer_values,
            device_str=device_str,
            epochs=15,
        )

        os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
        with open(args.output_json, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nQuick results saved to {args.output_json}")

        # Summary table
        print("\n" + "="*50)
        print("ALPHA SENSITIVITY (Val MAE, synthetic data)")
        print("-"*50)
        for alpha, mae in results['alpha'].items():
            marker = " <-- paper default" if alpha == "0.1" else ""
            print(f"  alpha={alpha:5s}  MAE={mae:.6f}{marker}")
        print("\nLAYER SENSITIVITY (Val MAE, synthetic data)")
        print("-"*50)
        for L, mae in results['layers'].items():
            marker = " <-- paper default" if L == "3" else ""
            print(f"  L={L:2s}        MAE={mae:.6f}{marker}")
        print("="*50)

    else:  # full mode
        configs = generate_full_configs(
            dataset=args.dataset,
            alpha_values=args.alpha_values,
            layer_values=args.layer_values,
            epochs=args.epochs,
        )
        print(f"\nGenerated {len(configs)} config files.")
        print("\nTo run all experiments, execute:")
        print("-"*60)
        for tag, cfg_path in configs:
            print(f"  python experiments/train.py -c {cfg_path} -g {args.gpus}  # {tag}")
        print("-"*60)
        print("\nOr run them all sequentially with:")
        print("  bash experiments_rebuttal/run_alpha_sweep.sh")

        # Generate a shell script
        script_path = 'experiments_rebuttal/run_alpha_sweep.sh'
        with open(script_path, 'w') as f:
            f.write("#!/bin/bash\n")
            f.write("# Auto-generated: alpha/layer sweep for AdaST\n\n")
            for tag, cfg_path in configs:
                f.write(f"echo '=== Running {tag} ==='\n")
                f.write(f"python experiments/train.py -c {cfg_path} -g {args.gpus}\n\n")
        os.chmod(script_path, 0o755)
        print(f"\nShell script written to {script_path}")
        print("Run: bash experiments_rebuttal/run_alpha_sweep.sh")


if __name__ == '__main__':
    main()
