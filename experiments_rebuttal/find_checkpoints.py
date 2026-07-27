"""
find_checkpoints.py
====================
Scans the checkpoints/ directory and prints:
  1. All available checkpoints with their full paths
  2. Ready-to-run exp1 commands for matching dataset pairs

Usage:
    python experiments_rebuttal/find_checkpoints.py
    python experiments_rebuttal/find_checkpoints.py --dataset PurpleAir
    python experiments_rebuttal/find_checkpoints.py --generate_script
"""
import os
import sys
import glob
import argparse
import json

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
os.chdir(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


def find_all_checkpoints(ckpt_root='checkpoints'):
    """
    Recursively find all .pt files under checkpoints/.
    Returns a dict:
      {model_name: {dataset_name: [list of .pt paths]}}
    """
    results = {}
    if not os.path.exists(ckpt_root):
        print(f"ERROR: {ckpt_root}/ directory not found.")
        return results

    for pt_path in glob.glob(f'{ckpt_root}/**/*.pt', recursive=True):
        parts = pt_path.replace('\\', '/').split('/')
        # parts[0] = 'checkpoints', parts[1] = model_name
        if len(parts) < 3:
            continue
        model_name = parts[1]

        # Dataset name: look at the subdirectory name after model_name
        # Format variants:
        #   checkpoints/AdaST/PurpleAir_50_12_12/<hash>/best.pt
        #   checkpoints/AdaST/PurpleAir_1_336_336/<hash>/AdaST_best_val_MAE.pt
        #   checkpoints/AdaST/<hash>/best.pt   (older format)
        subdir = parts[2] if len(parts) > 2 else ''

        # Try to extract dataset name from subdir
        # Pattern: DatasetName_epochs_inlen_outlen  OR  DatasetName
        dataset_name = subdir.split('_')[0] if subdir else 'unknown'

        # Skip if it's a hash (all hex chars)
        if all(c in '0123456789abcdefABCDEF' for c in dataset_name) and len(dataset_name) >= 8:
            # The immediate subdir IS the hash — dataset name is in parent
            dataset_name = 'unknown'

        if model_name not in results:
            results[model_name] = {}
        if dataset_name not in results[model_name]:
            results[model_name][dataset_name] = []
        results[model_name][dataset_name].append(pt_path)

    return results


def find_best_checkpoint(model_name, dataset_name, ckpt_root='checkpoints'):
    """
    Find the best checkpoint for a given model+dataset combination.
    
    Search priority:
      1. *best_val_MAE.pt
      2. *best*.pt
      3. Most recently modified .pt
    """
    search_patterns = [
        # Standard basicts naming
        f'{ckpt_root}/{model_name}/{dataset_name}*/**/*best_val_MAE.pt',
        f'{ckpt_root}/{model_name}/{dataset_name}*/**/*best_val*.pt',
        f'{ckpt_root}/{model_name}/{dataset_name}*/**/*best*.pt',
        f'{ckpt_root}/{model_name}/{dataset_name}*/**/*.pt',
        # Also try without dataset suffix
        f'{ckpt_root}/{model_name}/**/{dataset_name}*best*.pt',
        f'{ckpt_root}/{model_name}/**/*best_val_MAE.pt',
    ]

    candidates = []
    for pattern in search_patterns:
        found = glob.glob(pattern, recursive=True)
        # Filter: must contain dataset name somewhere in path
        filtered = [p for p in found if dataset_name.lower() in p.lower()]
        if filtered:
            candidates = filtered
            break

    if not candidates:
        return None

    # Prefer "best_val_MAE" over others
    for c in candidates:
        if 'best_val_MAE' in c:
            return c
    for c in candidates:
        if 'best' in c:
            return c

    # Fall back to most recently modified
    candidates.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return candidates[0]


def print_all_checkpoints(ckpt_root='checkpoints'):
    """Print a structured view of all checkpoints."""
    print(f"\n{'='*65}")
    print(f"CHECKPOINT INVENTORY  ({ckpt_root}/)")
    print(f"{'='*65}")

    if not os.path.exists(ckpt_root):
        print(f"  Directory not found: {ckpt_root}/")
        return

    for model_dir in sorted(os.listdir(ckpt_root)):
        model_path = os.path.join(ckpt_root, model_dir)
        if not os.path.isdir(model_path):
            continue
        print(f"\n  [{model_dir}]")
        all_pts = glob.glob(f'{model_path}/**/*.pt', recursive=True)
        if not all_pts:
            print(f"    (no .pt files found)")
            continue
        # Group by top-level subdir
        subdirs = {}
        for pt in all_pts:
            rel = os.path.relpath(pt, model_path)
            top = rel.split(os.sep)[0]
            subdirs.setdefault(top, []).append(pt)
        for subdir, pts in sorted(subdirs.items()):
            print(f"    {subdir}/")
            for pt in sorted(pts, key=lambda x: os.path.basename(x)):
                size_mb = os.path.getsize(pt) / 1e6
                fname = os.path.basename(pt)
                print(f"      {fname}  ({size_mb:.1f} MB)  -> {pt}")


def generate_exp1_commands(datasets, gpus='0', ckpt_root='checkpoints'):
    """Generate ready-to-run exp1 commands for each dataset."""
    print(f"\n{'='*65}")
    print(f"EXP1 COMMANDS")
    print(f"{'='*65}")

    adast_datasets = []
    for ds in datasets:
        adast_ckpt = find_best_checkpoint('AdaST', ds, ckpt_root)
        stae_ckpt  = find_best_checkpoint('STAEformer', ds, ckpt_root)

        print(f"\n  Dataset: {ds}")
        print(f"    AdaST  ckpt: {adast_ckpt or 'NOT FOUND'}")
        print(f"    STAEformer ckpt: {stae_ckpt or 'NOT FOUND (optional)'}")

        if adast_ckpt:
            stae_arg = f'--stae_ckpt "{stae_ckpt}"' if stae_ckpt else ''
            cmd = (
                f"python experiments_rebuttal/exp1_noise_robustness.py \\\n"
                f"    --dataset {ds} \\\n"
                f"    --adast_ckpt \"{adast_ckpt}\" \\\n"
                f"    {stae_arg} \\\n"
                f"    --noise_levels 0.0 0.5 1.0 2.0 \\\n"
                f"    --noise_types spatial temporal both \\\n"
                f"    --gpus {gpus} \\\n"
                f"    --output_json experiments_rebuttal/results_exp1_{ds}.json"
            )
            adast_datasets.append((ds, adast_ckpt, stae_ckpt))
            print(f"\n  Command:\n    {cmd}")
        else:
            print(f"  SKIP: AdaST checkpoint for {ds} not found.")
            print(f"  Run training first:")
            print(f"    python experiments/train.py -c baselines/AdaST/{ds}.py -g {gpus}")

    return adast_datasets


def generate_run_script(datasets, gpus='0', ckpt_root='checkpoints',
                         out='experiments_rebuttal/run_exp1_auto.sh'):
    """Write a shell script with auto-detected checkpoint paths."""
    lines = [
        '#!/bin/bash',
        '# Auto-generated by find_checkpoints.py',
        f'# GPU: {gpus}',
        '',
        'set -e',
        f'cd "$(dirname "$0")/.."',
        '',
    ]

    any_found = False
    for ds in datasets:
        adast_ckpt = find_best_checkpoint('AdaST', ds, ckpt_root)
        stae_ckpt  = find_best_checkpoint('STAEformer', ds, ckpt_root)

        if adast_ckpt:
            any_found = True
            stae_line = f'    --stae_ckpt "{stae_ckpt}" \\' if stae_ckpt else \
                        f'    # --stae_ckpt "NOT FOUND" \\'
            lines += [
                f'echo "=== Exp1: {ds} ==="',
                f'python experiments_rebuttal/exp1_noise_robustness.py \\',
                f'    --dataset {ds} \\',
                f'    --adast_ckpt "{adast_ckpt}" \\',
                stae_line,
                f'    --noise_levels 0.0 0.5 1.0 2.0 \\',
                f'    --noise_types spatial temporal both \\',
                f'    --gpus {gpus} \\',
                f'    --output_json experiments_rebuttal/results_exp1_{ds}.json',
                '',
            ]
        else:
            lines += [
                f'# SKIP {ds}: AdaST checkpoint not found',
                f'# Train first: python experiments/train.py -c baselines/AdaST/{ds}.py -g {gpus}',
                '',
            ]

    if not any_found:
        lines += ['echo "No checkpoints found. Please train models first."']

    with open(out, 'w') as f:
        f.write('\n'.join(lines))
    os.chmod(out, 0o755)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset',          default=None,
                   help='Specific dataset to look up. Default: show all.')
    p.add_argument('--ckpt_root',        default='checkpoints')
    p.add_argument('--gpus',             default='0')
    p.add_argument('--generate_script',  action='store_true',
                   help='Write run_exp1_auto.sh with correct checkpoint paths')
    p.add_argument('--datasets_for_exp1', nargs='+',
                   default=['PurpleAir', 'PEMS04', 'PEMS07', 'PEMS08',
                             'BeijingAirQuality', 'ETTh1', 'ETTh2',
                             'ETTm1', 'ETTm2', 'Weather'])
    args = p.parse_args()

    # 1. Print full inventory
    print_all_checkpoints(args.ckpt_root)

    # 2. Specific lookup
    if args.dataset:
        print(f"\n  Looking for '{args.dataset}':")
        for model in ['AdaST', 'STAEformer', 'GWNet', 'DLinear', 'NBeats']:
            ckpt = find_best_checkpoint(model, args.dataset, args.ckpt_root)
            print(f"    {model:15s}: {ckpt or 'not found'}")

    # 3. Exp1 commands
    generate_exp1_commands(args.datasets_for_exp1, args.gpus, args.ckpt_root)

    # 4. Generate script
    if args.generate_script:
        script = generate_run_script(
            args.datasets_for_exp1, args.gpus, args.ckpt_root)
        print(f"\n  Script written to: {script}")
        print(f"  Run: bash {script}")


if __name__ == '__main__':
    main()
