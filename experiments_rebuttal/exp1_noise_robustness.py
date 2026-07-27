"""
Experiment 1: Noise Robustness Analysis (Spurious Correlation Response)
=========================================================================
Usage:
  # Auto-find checkpoints and run:
  python experiments_rebuttal/exp1_noise_robustness.py \
      --dataset PurpleAir --gpus 0

  # Or specify checkpoints explicitly:
  python experiments_rebuttal/exp1_noise_robustness.py \
      --dataset PurpleAir \
      --adast_ckpt "checkpoints/AdaST/PurpleAir_50_336_336/<hash>/AdaST_best_val_MAE.pt" \
      --gpus 0
"""
import os, sys, json, glob, argparse
import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
os.chdir(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from basicts.metrics import masked_mae, masked_rmse, masked_mape
from basicts.data import TimeSeriesForecastingDataset
from basicts.scaler import ZScoreScaler
from basicts.utils import get_regular_settings
from torch.utils.data import DataLoader

DATASET_NODE_PARAMS = {
    'PurpleAir':        dict(num_nodes=55,  steps_per_day=240),
    'PEMS03':           dict(num_nodes=358, steps_per_day=288),
    'PEMS04':           dict(num_nodes=307, steps_per_day=288),
    'PEMS07':           dict(num_nodes=883, steps_per_day=288),
    'PEMS08':           dict(num_nodes=170, steps_per_day=288),
    'BeijingAirQuality':dict(num_nodes=36,  steps_per_day=24),
    'ETTh1':            dict(num_nodes=7,   steps_per_day=24),
    'ETTh2':            dict(num_nodes=7,   steps_per_day=24),
    'ETTm1':            dict(num_nodes=7,   steps_per_day=96),
    'ETTm2':            dict(num_nodes=7,   steps_per_day=96),
    'Weather':          dict(num_nodes=21,  steps_per_day=144),
    'METR-LA':          dict(num_nodes=207, steps_per_day=288),
    'PEMS-BAY':         dict(num_nodes=325, steps_per_day=288),
}


def auto_find_checkpoint(model_name, dataset_name, ckpt_root='checkpoints'):
    patterns = [
        f'{ckpt_root}/{model_name}/{dataset_name}*/**/*best_val_MAE.pt',
        f'{ckpt_root}/{model_name}/{dataset_name}*/**/*best_val*.pt',
        f'{ckpt_root}/{model_name}/{dataset_name}*/**/*best*.pt',
        f'{ckpt_root}/{model_name}/{dataset_name}*/**/*.pt',
    ]
    for pat in patterns:
        found = glob.glob(pat, recursive=True)
        if found:
            for f in found:
                if 'best_val_MAE' in f: return f
            return sorted(found, key=lambda p: os.path.getmtime(p), reverse=True)[0]
    return None


def load_model(arch_class, ckpt_path, dataset_name, device):
    """Load model, auto-inferring params from state dict if needed."""
    print(f"  Loading from: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    # Extract state dict
    if isinstance(ckpt, dict):
        state_dict = ckpt.get('model_state_dict',
                    ckpt.get('state_dict',
                    ckpt.get('model', ckpt)))
        saved_params = ckpt.get('param', ckpt.get('model_param', None))
    else:
        state_dict = ckpt
        saved_params = None

    # Infer params from state dict keys
    base = DATASET_NODE_PARAMS.get(dataset_name,
                                    dict(num_nodes=100, steps_per_day=288))
    regular = get_regular_settings(dataset_name)

    # Read actual dims from weights
    input_emb_dim = state_dict['input_proj.weight'].shape[0] \
        if 'input_proj.weight' in state_dict else 24
    input_dim = state_dict['input_proj.weight'].shape[1] \
        if 'input_proj.weight' in state_dict else 3
    tod_dim = state_dict['tod_embedding.weight'].shape[1] \
        if 'tod_embedding.weight' in state_dict else 24
    steps_per_day = state_dict['tod_embedding.weight'].shape[0] \
        if 'tod_embedding.weight' in state_dict else base['steps_per_day']
    dow_dim = state_dict['dow_embedding.weight'].shape[1] \
        if 'dow_embedding.weight' in state_dict else 24
    adp_dim = state_dict['adaptive_embedding'].shape[2] \
        if 'adaptive_embedding' in state_dict else 80
    num_nodes = state_dict['adaptive_embedding'].shape[1] \
        if 'adaptive_embedding' in state_dict else base['num_nodes']

    params = dict(
        num_nodes=num_nodes,
        in_steps=regular['INPUT_LEN'],
        out_steps=regular['OUTPUT_LEN'],
        steps_per_day=steps_per_day,
        input_dim=input_dim, output_dim=1,
        input_embedding_dim=input_emb_dim,
        tod_embedding_dim=tod_dim,
        dow_embedding_dim=dow_dim,
        spatial_embedding_dim=0,
        adaptive_embedding_dim=adp_dim,
        feed_forward_dim=256,
        num_heads=4, num_layers=3, dropout=0.1, use_mixed_proj=True,
    )
    print(f"    num_nodes={num_nodes}, steps_per_day={steps_per_day}, "
          f"in_steps={params['in_steps']}")

    model = arch_class(**params).to(device)
    try:
        model.load_state_dict(state_dict, strict=True)
        print(f"    ✓ strict=True OK")
    except RuntimeError as e:
        print(f"    strict=False (reason: {str(e)[:60]}...)")
        model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model


def build_test_loader(dataset_name, batch_size=64):
    regular = get_regular_settings(dataset_name)
    ds = TimeSeriesForecastingDataset(
        dataset_name=dataset_name,
        train_val_test_ratio=regular['TRAIN_VAL_TEST_RATIO'],
        mode='test',
        input_len=regular['INPUT_LEN'],
        output_len=regular['OUTPUT_LEN'],
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=2)


def inject_noise(x, noise_type, noise_std):
    if noise_std == 0.0:
        return x
    x = x.clone()
    B, T, N, C = x.shape
    if noise_type in ('spatial', 'both'):
        x[..., 0:1] += torch.randn(B, 1, N, 1, device=x.device) * noise_std
    if noise_type in ('temporal', 'both'):
        x[..., 0:1] += torch.randn(B, T, 1, 1, device=x.device) * noise_std
    return x


@torch.no_grad()
def evaluate(model, loader, scaler, device, fw_feats, tgt_feats, null_val,
             noise_type, noise_std):
    model.eval()
    preds, tgts = [], []
    for batch in loader:
        inp = batch['inputs'].to(device)
        tgt = batch['target'].to(device)
        inp_n = scaler.transform(inp)
        tgt_n = scaler.transform(tgt)
        hist = inp_n[:, :, :, fw_feats]
        fut  = tgt_n[:, :, :, fw_feats].clone()
        fut[..., 0] = 0.0
        hist = inject_noise(hist, noise_type, noise_std)
        out  = model(history_data=hist, future_data=fut,
                     batch_seen=0, epoch=0, train=False)
        pred = (out['prediction'] if isinstance(out, dict) else out)[:, :, :, tgt_feats]
        preds.append(scaler.inverse_transform(pred).cpu())
        tgts.append(tgt[:, :, :, tgt_feats].cpu())
    P = torch.cat(preds); T = torch.cat(tgts)
    nv = torch.tensor(null_val)
    return (masked_mae(P,T,nv).item(),
            masked_rmse(P,T,nv).item(),
            masked_mape(P,T,nv).item())


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset',      default='PurpleAir')
    p.add_argument('--adast_ckpt',   default=None,
                   help='Path to AdaST .pt  (auto-searched if not given)')
    p.add_argument('--stae_ckpt',    default=None,
                   help='Path to STAEformer .pt (auto-searched if not given)')
    p.add_argument('--ckpt_root',    default='checkpoints')
    p.add_argument('--noise_levels', nargs='+', type=float,
                   default=[0.0, 0.5, 1.0, 2.0])
    p.add_argument('--noise_types',  nargs='+',
                   default=['spatial', 'temporal', 'both'])
    p.add_argument('--batch_size',   type=int, default=64)
    p.add_argument('--gpus',         default='0')
    p.add_argument('--output_json',  default=None)
    return p.parse_args()


def main():
    args = parse_args()
    if args.output_json is None:
        args.output_json = f'experiments_rebuttal/results_exp1_{args.dataset}.json'
    device = torch.device(
        f'cuda:{args.gpus}' if torch.cuda.is_available() else 'cpu')
    print(f"\nDataset={args.dataset}  Device={device}")
    print("="*65)

    # Auto-find checkpoints
    if args.adast_ckpt is None:
        args.adast_ckpt = auto_find_checkpoint('AdaST', args.dataset, args.ckpt_root)
    if args.stae_ckpt is None:
        args.stae_ckpt  = auto_find_checkpoint('STAEformer', args.dataset, args.ckpt_root)

    if args.adast_ckpt is None:
        print(f"ERROR: No AdaST checkpoint found under {args.ckpt_root}/AdaST/{args.dataset}*/")
        print("Available checkpoint dirs:")
        for d in sorted(glob.glob(f'{args.ckpt_root}/AdaST/*')):
            print(f"  {d}")
        sys.exit(1)
    else:
        print(f"AdaST  ckpt: {args.adast_ckpt}")
    if args.stae_ckpt:
        print(f"STAEformer  ckpt: {args.stae_ckpt}")
    else:
        print("STAEformer: not found (optional)")

    # Data
    regular  = get_regular_settings(args.dataset)
    null_val = regular['NULL_VAL']
    loader   = build_test_loader(args.dataset, args.batch_size)
    scaler   = ZScoreScaler(dataset_name=args.dataset,
                             train_ratio=regular['TRAIN_VAL_TEST_RATIO'][0],
                             norm_each_channel=regular['NORM_EACH_CHANNEL'],
                             rescale=regular['RESCALE'])
    print(f"Test samples: {len(loader.dataset)}")

    # Load models
    models = {}
    from baselines.AdaST.arch import AdaST as AdaSTArch
    models['AdaST'] = {
        'model': load_model(AdaSTArch, args.adast_ckpt, args.dataset, device),
        'fw': [0,1,2], 'tgt': [0],
    }
    if args.stae_ckpt and os.path.exists(args.stae_ckpt):
        try:
            from baselines.STAEformer.arch import STAEformer
            models['STAEformer'] = {
                'model': load_model(STAEformer, args.stae_ckpt, args.dataset, device),
                'fw': [0,1,2], 'tgt': [0],
            }
        except Exception as e:
            print(f"STAEformer load failed: {e}")

    # Sweep
    print(f"\nSweeping {args.noise_types} x σ={args.noise_levels} ...")
    results = {}
    for mname, info in models.items():
        results[mname] = {}
        for ntype in args.noise_types:
            results[mname][ntype] = {}
            for std in args.noise_levels:
                mae, rmse, mape = evaluate(
                    info['model'], loader, scaler, device,
                    info['fw'], info['tgt'], null_val, ntype, std)
                results[mname][ntype][str(std)] = {
                    'MAE': round(mae,4), 'RMSE': round(rmse,4),
                    'MAPE%': round(mape*100,4)}
                print(f"  {mname:12s}|{ntype:8s}|σ={std:.2f}| "
                      f"MAE={mae:.4f} RMSE={rmse:.4f} MAPE={mape*100:.2f}%")

    os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
    with open(args.output_json, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved → {args.output_json}")

    # Summary
    print("\n" + "="*70)
    print(f"SUMMARY  ({args.dataset})")
    print("="*70)
    for ntype in args.noise_types:
        print(f"\n  [{ntype.upper()} noise]  relative MAE degradation vs σ=0")
        for mname in results:
            nd = results[mname].get(ntype, {})
            mae0 = nd.get(str(args.noise_levels[0]),{}).get('MAE',0)
            row  = f"  {mname:12s}"
            for std in args.noise_levels[1:]:
                mae_s = nd.get(str(std),{}).get('MAE', float('nan'))
                pct   = (mae_s-mae0)/mae0*100 if mae0>0 else float('nan')
                row  += f"  σ={std}: +{pct:.1f}%"
            print(row)

    if 'AdaST' in results and 'STAEformer' in results:
        print("\nKEY FINDING:")
        for ntype in ['spatial','temporal']:
            if ntype not in results['AdaST']: continue
            max_s = str(max(args.noise_levels))
            a_deg = (results['AdaST'][ntype][max_s]['MAE'] -
                     results['AdaST'][ntype][str(args.noise_levels[0])]['MAE']) / \
                     results['AdaST'][ntype][str(args.noise_levels[0])]['MAE']*100
            s_deg = (results['STAEformer'][ntype][max_s]['MAE'] -
                     results['STAEformer'][ntype][str(args.noise_levels[0])]['MAE']) / \
                     results['STAEformer'][ntype][str(args.noise_levels[0])]['MAE']*100
            winner = 'AdaST' if a_deg < s_deg else 'STAEformer'
            print(f"  {ntype:8s} σ={max_s}: AdaST +{a_deg:.1f}% vs STAEformer +{s_deg:.1f}% "
                  f"→ {winner} more robust")


if __name__ == '__main__':
    main()
