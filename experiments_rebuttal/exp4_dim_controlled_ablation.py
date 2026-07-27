"""
Experiment 4: Dimension-Controlled Ablation
=============================================
Motivation (Reviewer cknJ Q1):
  "In the ablation study, removing one expert (i.e., w/o Eh) will reduce
   the dimensionality of H(·). How is this handled, and how do you ensure
   that the performance drop is caused by removing the expert rather than
   by the dimension change?"

Design:
  When removing expert Ek (dim=Dk), we REPLACE it with a DOUBLED copy
  of another expert (dim=2*Dk) so that the total dimension DH remains
  constant. This isolates the contribution of the expert's INFORMATION
  content from its capacity effect.

  For example, when removing Eh (tod expert, dim=24):
    - Standard ablation:  H = [Zn(16) | Zw(8)  | Za(20)]  (dim reduced)
    - Dimension-controlled: H = [Zn(16) | Zn'(8) | Za(20)]  (dim same)
      where Zn' is an independent copy of Zn with a separate projection.

  We test this on PEMS07 and PurpleAir to match the paper's Table 3.

  IMPORTANT IMPLEMENTATION NOTE:
    The doubled expert uses SEPARATE projection weights — it is not
    merely duplicated features. This gives it the same parameter budget
    while lacking the original expert's specialized embedding.

Usage:
  python experiments_rebuttal/exp4_dim_controlled_ablation.py \
      --dataset PEMS07 \
      --epochs 50 \
      --gpus 0

  Results are written to:
      experiments_rebuttal/results_exp4_dim_ablation.json
  And config files for basicts training to:
      experiments_rebuttal/dim_ablation_cfgs/
"""
import os
import sys
import json
import copy
import argparse
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
os.chdir(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.nn as nn
import torch.nn.functional as F


# ══════════════════════════════════════════════════════════════════════
# Dimension-controlled AdaST variants
# ══════════════════════════════════════════════════════════════════════

class AdaSTDimControlled(nn.Module):
    """
    AdaST with dimension-controlled expert ablation.

    Instead of REMOVING an expert and reducing dimension, we REPLACE
    the ablated expert with an additional copy of another expert
    (using separate, independent weights), keeping DH constant.

    Args:
        ablate_expert (str): Which expert to ablate:
            'none'  -> full model (baseline)
            'tod'   -> remove time-of-day expert (Eh), replace with extra spatial
            'dow'   -> remove day-of-week expert (Ew), replace with extra spatial
            'spatial'  -> remove spatial expert (En), replace with extra tod
            'adaptive' -> remove adaptive expert (Ea), replace with extra tod
        All other args: same as original AdaST.
    """

    def __init__(
        self,
        num_nodes: int,
        in_steps: int = 12,
        out_steps: int = 12,
        steps_per_day: int = 288,
        input_dim: int = 3,
        output_dim: int = 1,
        input_embedding_dim: int = 24,
        tod_embedding_dim: int = 24,
        dow_embedding_dim: int = 24,
        spatial_embedding_dim: int = 24,
        adaptive_embedding_dim: int = 80,
        feed_forward_dim: int = 256,
        num_layers: int = 3,
        dropout: float = 0.1,
        num_heads: int = 4,
        use_mixed_proj: bool = True,
        ablate_expert: str = 'none',   # NEW
    ):
        super().__init__()
        self.num_nodes = num_nodes
        self.in_steps = in_steps
        self.out_steps = out_steps
        self.steps_per_day = steps_per_day
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.ablate_expert = ablate_expert
        self.use_mixed_proj = use_mixed_proj

        # ── base input projection ──
        self.input_proj = nn.Linear(input_dim, input_embedding_dim)

        # ── expert embeddings ──
        # Each expert produces its own contextualized representation.
        # When an expert is ablated, we give its "slot" to a replacement expert
        # (same TYPE as another expert but with independent weights).
        # This keeps DH identical to the full model.

        dim_x = input_embedding_dim

        # Determine which experts are active and their dims
        # We use a 4-expert scheme: [spatial, tod, dow, adaptive]
        # Each expert i has embedding dim Di and produces hidden dim (dim_x + Di)
        # Final model_dim = sum of all 4 expert output dims

        # ── Define expert slot assignments ──
        # slot_configs: list of (type, embedding_dim) for each of 4 slots
        # 'type' is what kind of expert fills that slot
        if ablate_expert == 'none':
            # Standard: [spatial(En), tod(Eh), dow(Ew), adaptive(Ea)]
            slot_types = ['spatial', 'tod', 'dow', 'adaptive']
            slot_dims  = [spatial_embedding_dim, tod_embedding_dim,
                          dow_embedding_dim, adaptive_embedding_dim]
        elif ablate_expert == 'tod':
            # Remove Eh, replace with extra Spatial expert (independent weights)
            # Eh slot gets spatial_embedding_dim (==24) to preserve same dim
            slot_types = ['spatial', 'spatial_extra', 'dow', 'adaptive']
            slot_dims  = [spatial_embedding_dim, tod_embedding_dim,
                          dow_embedding_dim, adaptive_embedding_dim]
        elif ablate_expert == 'dow':
            # Remove Ew, replace with extra ToD expert
            slot_types = ['spatial', 'tod', 'tod_extra', 'adaptive']
            slot_dims  = [spatial_embedding_dim, tod_embedding_dim,
                          dow_embedding_dim, adaptive_embedding_dim]
        elif ablate_expert == 'spatial':
            # Remove En, replace with extra ToD expert (dim=spatial_embedding_dim)
            slot_types = ['tod_extra2', 'tod', 'dow', 'adaptive']
            slot_dims  = [spatial_embedding_dim, tod_embedding_dim,
                          dow_embedding_dim, adaptive_embedding_dim]
        elif ablate_expert == 'adaptive':
            # Remove Ea, replace with extra ToD expert
            slot_types = ['spatial', 'tod', 'dow', 'tod_extra3']
            slot_dims  = [spatial_embedding_dim, tod_embedding_dim,
                          dow_embedding_dim, adaptive_embedding_dim]
        else:
            raise ValueError(f"Unknown ablate_expert: {ablate_expert}")

        self.slot_types = slot_types
        self.slot_dims  = slot_dims

        # ── Create embeddings for each slot ──
        for i, (stype, sdim) in enumerate(zip(slot_types, slot_dims)):
            if 'spatial' in stype:
                emb = nn.Parameter(torch.empty(num_nodes, sdim))
                nn.init.xavier_uniform_(emb)
                setattr(self, f'emb_{i}', emb)
            elif 'tod' in stype:
                emb = nn.Embedding(steps_per_day, sdim)
                setattr(self, f'emb_{i}', emb)
            elif stype == 'dow':
                emb = nn.Embedding(7, sdim)
                setattr(self, f'emb_{i}', emb)

        # adaptive embedding is special: shape (in_steps, num_nodes, adaptive_dim)
        if 'adaptive' in slot_types:
            adp_idx = slot_types.index('adaptive')
            adp_dim = slot_dims[adp_idx]
            adp = nn.Parameter(torch.empty(in_steps, num_nodes, adp_dim))
            nn.init.xavier_uniform_(adp)
            self.adaptive_embedding = adp
        # replace adaptive slot with a tod-type extra expert
        elif 'tod_extra3' in slot_types:
            # No adaptive embedding; tod_extra3 is a tod embedding with larger dim
            adp_idx = slot_types.index('tod_extra3')
            adp_dim = slot_dims[adp_idx]
            adp_emb = nn.Embedding(steps_per_day, adp_dim)
            setattr(self, f'emb_{adp_idx}', adp_emb)

        # ── Compute slot output dims ──
        slot_out_dims = [dim_x + d for d in slot_dims]
        self.model_dim = sum(slot_out_dims)

        # ── 3-way projection heads ──
        self.proj_heads = nn.ModuleList([
            nn.Linear(d, d * 3) for d in slot_out_dims
        ])

        # ── Spatial-Temporal modules ──
        from baselines.AdaST.arch.adast_arch import (
            SpatialMixerLayer, SelfAttentionLayer, CombineCorrelationAggregation
        )
        self.mixer_layers_s = nn.ModuleList([
            SpatialMixerLayer(num_nodes, self.model_dim, feed_forward_dim, dropout)
            for _ in range(num_layers)
        ])
        self.mixer_layers_s_specific = nn.ModuleList([
            SpatialMixerLayer(num_nodes, self.model_dim, feed_forward_dim, dropout)
            for _ in range(num_layers)
        ])
        self.attn_layers_t = nn.ModuleList([
            SelfAttentionLayer(self.model_dim, feed_forward_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])
        self.attn_layers_t_specific = nn.ModuleList([
            SelfAttentionLayer(self.model_dim, feed_forward_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])
        self.aggregation = CombineCorrelationAggregation(self.model_dim)

        # ── Output projection ──
        if use_mixed_proj:
            self.output_proj = nn.Linear(
                in_steps * self.model_dim, out_steps * output_dim)
        else:
            self.temporal_proj = nn.Linear(in_steps, out_steps)
            self.output_proj   = nn.Linear(self.model_dim, output_dim)

    def _get_slot_embedding(self, i, tod_idx, dow_idx, B):
        """Return the contextualized embedding for slot i."""
        stype = self.slot_types[i]
        emb_module = getattr(self, f'emb_{i}', None)

        if 'spatial' in stype:
            # (N, D) -> expand to (B, T, N, D)
            return emb_module.expand(B, self.in_steps, *emb_module.shape)
        elif 'tod' in stype:
            if hasattr(emb_module, 'weight'):  # nn.Embedding
                return emb_module(tod_idx)  # (B, T, N, D)
            else:
                # tod_extra3: also an embedding
                return emb_module(tod_idx)
        elif stype == 'dow':
            return emb_module(dow_idx)  # (B, T, N, D)
        elif stype == 'adaptive':
            return self.adaptive_embedding.expand(
                B, *self.adaptive_embedding.shape)  # (B, T, N, D)
        return None

    def forward(self, history_data, future_data, batch_seen, epoch, train, **kwargs):
        x_full = history_data
        B = x_full.shape[0]

        # Extract tod/dow indices
        tod_idx = (x_full[..., 1] * self.steps_per_day).clamp(
            0, self.steps_per_day - 1).long()
        dow_idx = (x_full[..., 2] * 7).clamp(0, 6).long()

        # Project input
        x = self.input_proj(x_full[..., :self.input_dim])  # (B, T, N, dim_x)

        # ── Build per-slot contextualized inputs ──
        slot_contexts = []
        for i in range(len(self.slot_types)):
            slot_emb = self._get_slot_embedding(i, tod_idx, dow_idx, B)
            if slot_emb is not None:
                ctx = torch.cat([x, slot_emb], dim=-1)
            else:
                ctx = x
            slot_contexts.append(ctx)

        # ── 3-way split for each slot ──
        mains, temporals, spatials = [], [], []
        for i, ctx in enumerate(slot_contexts):
            out = self.proj_heads[i](ctx)  # (B, T, N, 3*slot_dim)
            m, t, s = out.chunk(3, dim=-1)
            mains.append(m)
            temporals.append(t)
            spatials.append(s)

        # ── Concatenate across slots ──
        x_main = torch.cat(mains,    dim=-1)  # (B, T, N, model_dim)
        x_t    = torch.cat(temporals, dim=-1)
        x_s    = torch.cat(spatials,  dim=-1)

        # ── ST modeling ──
        x = x_main
        for attn in self.attn_layers_t:
            x = attn(x, dim=1)
        for mixer in self.mixer_layers_s:
            x = mixer(x)

        xt = x_t
        for attn in self.attn_layers_t_specific:
            xt = attn(xt, dim=1)

        xs = x_s
        for mixer in self.mixer_layers_s_specific:
            xs = mixer(xs)

        # ── Aggregation ──
        x_agg, gate_weights, corr_features = self.aggregation(x, xt, xs)

        # ── Decode ──
        if self.use_mixed_proj:
            out = x_agg.transpose(1, 2)  # (B, N, T, D)
            out = out.reshape(B, self.num_nodes, self.in_steps * self.model_dim)
            out = self.output_proj(out).view(
                B, self.num_nodes, self.out_steps, self.output_dim)
            out = out.transpose(1, 2)  # (B, T, N, output_dim)
        else:
            out = x_agg.transpose(1, 3)
            out = self.temporal_proj(out)
            out = self.output_proj(out.transpose(1, 3))
        return out


# ══════════════════════════════════════════════════════════════════════
# Config builder
# ══════════════════════════════════════════════════════════════════════

def build_dim_ablation_cfg(dataset: str, ablate_expert: str,
                            epochs: int = 50) -> dict:
    """
    Returns a basicts CFG for a dimension-controlled ablation run.
    """
    from easydict import EasyDict
    from basicts.metrics import masked_mae, masked_mape, masked_rmse
    from basicts.data import TimeSeriesForecastingDataset
    from basicts.runners import SimpleTimeSeriesForecastingRunner
    from basicts.scaler import ZScoreScaler
    from basicts.utils import get_regular_settings

    regular   = get_regular_settings(dataset)
    INPUT_LEN  = regular['INPUT_LEN']
    OUTPUT_LEN = regular['OUTPUT_LEN']
    TVT        = regular['TRAIN_VAL_TEST_RATIO']
    NORM       = regular['NORM_EACH_CHANNEL']
    RESCALE    = regular['RESCALE']
    NULL_VAL   = regular['NULL_VAL']

    # Dataset-specific node counts
    NODE_COUNTS = {
        'PEMS04': 307, 'PEMS07': 883, 'PEMS08': 170,
        'PEMS03': 358, 'PurpleAir': 55,
    }
    num_nodes = NODE_COUNTS.get(dataset, 100)
    steps_per_day = 288 if 'PEMS' in dataset else 240

    MODEL_PARAM = {
        'num_nodes': num_nodes,
        'in_steps':  INPUT_LEN,
        'out_steps': OUTPUT_LEN,
        'steps_per_day': steps_per_day,
        'input_dim': 3, 'output_dim': 1,
        'input_embedding_dim': 24,
        'tod_embedding_dim': 24,
        'dow_embedding_dim': 24,
        'spatial_embedding_dim': 24,
        'adaptive_embedding_dim': 80,
        'feed_forward_dim': 256,
        'num_heads': 4,
        'num_layers': 3,
        'dropout': 0.1,
        'use_mixed_proj': True,
        'ablate_expert': ablate_expert,  # KEY PARAM
    }

    tag = f'DimCtrl_{ablate_expert}'
    CFG = EasyDict()
    CFG.DESCRIPTION  = f'AdaST dim-controlled ablation ({ablate_expert}) on {dataset}'
    CFG.GPU_NUM      = 1
    CFG.RUNNER       = SimpleTimeSeriesForecastingRunner
    CFG.DATASET      = EasyDict()
    CFG.DATASET.NAME = dataset
    CFG.DATASET.TYPE = TimeSeriesForecastingDataset
    CFG.DATASET.PARAM = EasyDict({
        'dataset_name': dataset,
        'train_val_test_ratio': TVT,
        'input_len': INPUT_LEN,
        'output_len': OUTPUT_LEN,
    })
    CFG.SCALER      = EasyDict()
    CFG.SCALER.TYPE = ZScoreScaler
    CFG.SCALER.PARAM = EasyDict({
        'dataset_name': dataset,
        'train_ratio': TVT[0],
        'norm_each_channel': NORM,
        'rescale': RESCALE,
    })
    CFG.MODEL       = EasyDict()
    CFG.MODEL.NAME  = f'AdaST_{tag}'
    CFG.MODEL.ARCH  = AdaSTDimControlled
    CFG.MODEL.PARAM = MODEL_PARAM
    CFG.MODEL.FORWARD_FEATURES = [0, 1, 2]
    CFG.MODEL.TARGET_FEATURES  = [0]
    CFG.METRICS      = EasyDict()
    CFG.METRICS.FUNCS = EasyDict({
        'MAE': masked_mae, 'MAPE': masked_mape, 'RMSE': masked_rmse})
    CFG.METRICS.TARGET   = 'MAE'
    CFG.METRICS.NULL_VAL = NULL_VAL
    CFG.TRAIN        = EasyDict()
    CFG.TRAIN.NUM_EPOCHS = epochs
    CFG.TRAIN.CKPT_SAVE_DIR = os.path.join(
        'checkpoints', f'AdaST_{tag}',
        f'{dataset}_{epochs}_{INPUT_LEN}_{OUTPUT_LEN}')
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


# ══════════════════════════════════════════════════════════════════════
# Quick sanity check: verify dimension stays constant
# ══════════════════════════════════════════════════════════════════════
def verify_dimension_consistency():
    """
    Verify that removing any expert keeps model_dim constant.
    """
    print("\n[Sanity Check] Verifying dimension consistency...")
    base_params = dict(
        num_nodes=50, in_steps=12, out_steps=12, steps_per_day=288,
        input_dim=3, output_dim=1,
        input_embedding_dim=24, tod_embedding_dim=24,
        dow_embedding_dim=24, spatial_embedding_dim=24,
        adaptive_embedding_dim=80, feed_forward_dim=64,
        num_layers=2, dropout=0.1, num_heads=4, use_mixed_proj=True,
    )

    models = {}
    for ablate in ['none', 'tod', 'dow', 'spatial', 'adaptive']:
        m = AdaSTDimControlled(**base_params, ablate_expert=ablate)
        models[ablate] = m.model_dim

    print(f"{'Ablation':15s}  model_dim  {'Same as full?':15s}")
    print("-" * 45)
    full_dim = models['none']
    all_ok = True
    for ablate, dim in models.items():
        ok = (dim == full_dim)
        all_ok = all_ok and ok
        marker = "✓" if ok else "✗ MISMATCH!"
        print(f"  {ablate:13s}  {dim:9d}  {marker}")

    if all_ok:
        print("\n✓ All variants have identical model_dim = ", full_dim)
    else:
        print("\n✗ Dimension mismatch detected!")
    return all_ok


# ══════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════
ABLATION_VARIANTS = ['none', 'tod', 'dow', 'spatial', 'adaptive']


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--datasets',   nargs='+', default=['PEMS07', 'PurpleAir'])
    p.add_argument('--ablations',  nargs='+', default=ABLATION_VARIANTS)
    p.add_argument('--epochs',     type=int, default=50)
    p.add_argument('--gpus',       default='0')
    p.add_argument('--mode',       choices=['verify', 'generate', 'train'],
                   default='generate',
                   help='verify: sanity check | generate: write configs | train: run training')
    p.add_argument('--output_json',
                   default='experiments_rebuttal/results_exp4_dim_ablation.json')
    return p.parse_args()


def main():
    args = parse_args()

    if args.mode == 'verify':
        verify_dimension_consistency()
        return

    # Always run verify first
    if not verify_dimension_consistency():
        print("ERROR: Dimension mismatch! Please fix before running experiments.")
        return

    if args.mode == 'generate':
        # Write shell scripts and config summaries
        os.makedirs('experiments_rebuttal', exist_ok=True)
        script_lines = ["#!/bin/bash", "# Exp4: Dimension-controlled ablation\n"]

        for dataset in args.datasets:
            script_lines.append(f"\necho '=== Dataset: {dataset} ==='")
            for ablate in args.ablations:
                tag = f"DimCtrl_{ablate}"
                script_lines.append(f"echo '  Ablation: {ablate}'")
                # Since CFG is a Python object, launch directly
                script_lines.append(
                    f"python experiments_rebuttal/exp4_dim_controlled_ablation.py "
                    f"--mode train --datasets {dataset} "
                    f"--ablations {ablate} "
                    f"--epochs {args.epochs} "
                    f"--gpus {args.gpus}"
                )

        script_path = 'experiments_rebuttal/run_dim_ablation.sh'
        with open(script_path, 'w') as f:
            f.write('\n'.join(script_lines))
        os.chmod(script_path, 0o755)
        print(f"\nGenerated: {script_path}")
        print("Run: bash experiments_rebuttal/run_dim_ablation.sh")

        # Print expected results table format
        print("\nExpected results table (fill in after running):")
        print("="*80)
        header = f"{'Variant':20s}"
        for ds in args.datasets:
            header += f"  {ds:>12s}(MAE)  {ds:>12s}(RMSE)"
        print(header)
        print("-"*80)
        variant_names = {
            'none':     'Full AdaST',
            'tod':      'w/o Eh (dim-ctrl)',
            'dow':      'w/o Ew (dim-ctrl)',
            'spatial':  'w/o En (dim-ctrl)',
            'adaptive': 'w/o Ea (dim-ctrl)',
        }
        for ablate in args.ablations:
            print(f"  {variant_names.get(ablate, ablate):18s}  [run experiment to fill]")
        print("="*80)

    elif args.mode == 'train':
        import basicts
        all_results = {}

        for dataset in args.datasets:
            all_results[dataset] = {}
            data_path = f'datasets/{dataset}/data.dat'
            if not os.path.exists(data_path):
                print(f"WARNING: {data_path} not found, skipping {dataset}")
                continue

            for ablate in args.ablations:
                print(f"\n{'='*60}")
                print(f"Dataset={dataset}, Ablation='{ablate}'")
                print(f"{'='*60}")

                cfg = build_dim_ablation_cfg(
                    dataset=dataset,
                    ablate_expert=ablate,
                    epochs=args.epochs
                )

                try:
                    basicts.launch_training(cfg=cfg, gpus=args.gpus, node_rank=0)
                    print(f"✓ {dataset}/{ablate} training complete.")
                    ckpt_dir = cfg.TRAIN.CKPT_SAVE_DIR
                    summary_path = os.path.join(ckpt_dir, 'metrics_summary.json')
                    if os.path.exists(summary_path):
                        with open(summary_path) as f:
                            metrics = json.load(f)
                        all_results[dataset][ablate] = metrics
                    else:
                        all_results[dataset][ablate] = {
                            'status': 'complete', 'ckpt_dir': ckpt_dir}
                except Exception as e:
                    print(f"✗ {dataset}/{ablate} failed: {e}")
                    all_results[dataset][ablate] = {'status': 'failed', 'error': str(e)}

        # Save
        os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
        with open(args.output_json, 'w') as f:
            json.dump(all_results, f, indent=2)
        print(f"\nAll results saved to {args.output_json}")

        # Summary table
        print("\n" + "="*80)
        print("DIMENSION-CONTROLLED ABLATION RESULTS")
        print("="*80)
        for dataset, ablation_results in all_results.items():
            print(f"\n  Dataset: {dataset}")
            print(f"  {'Variant':22s}  {'MAE':>8s}  {'RMSE':>8s}  {'MAPE':>8s}")
            print(f"  {'-'*55}")
            for ablate, res in ablation_results.items():
                if isinstance(res, dict) and 'MAE' in res:
                    mae  = res.get('best_test_MAE',  res.get('MAE',  'N/A'))
                    rmse = res.get('best_test_RMSE', res.get('RMSE', 'N/A'))
                    mape = res.get('best_test_MAPE', res.get('MAPE', 'N/A'))
                    vname = {'none': 'Full AdaST', 'tod': 'w/o Eh (dim-ctrl)',
                             'dow': 'w/o Ew (dim-ctrl)',
                             'spatial': 'w/o En (dim-ctrl)',
                             'adaptive': 'w/o Ea (dim-ctrl)'}.get(ablate, ablate)
                    print(f"  {vname:22s}  {str(mae):>8s}  {str(rmse):>8s}  {str(mape):>8s}")
        print("="*80)


if __name__ == '__main__':
    main()
