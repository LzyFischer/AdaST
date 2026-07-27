"""
Creates a minimal synthetic dataset for testing experiments.
Format matches the basicts TimeSeriesForecastingDataset format.
"""
import os
import json
import numpy as np

def create_mini_dataset(
    name="MiniPEMS",
    n_nodes=50,
    n_timesteps=3000,
    steps_per_day=288,
    save_dir="datasets"
):
    """Create a small synthetic traffic-like dataset."""
    os.makedirs(f"{save_dir}/{name}", exist_ok=True)
    
    np.random.seed(42)
    # Generate spatial-temporal correlated data
    # Base signal: AR(1) process with spatial diffusion
    data = np.zeros((n_timesteps, n_nodes, 1), dtype=np.float32)
    
    # Initialize
    data[0] = np.random.randn(n_nodes, 1) * 2 + 50
    
    # Simple adjacency for spatial correlation
    A = np.zeros((n_nodes, n_nodes))
    for i in range(n_nodes):
        for j in range(max(0, i-2), min(n_nodes, i+3)):
            if i != j:
                A[i, j] = 1.0
    row_sums = A.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    A = A / row_sums
    
    # Generate time series
    for t in range(1, n_timesteps):
        tod = (t % steps_per_day) / steps_per_day
        daily_pattern = 20 * np.sin(2 * np.pi * tod) + 50
        
        temporal_part = 0.7 * data[t-1, :, 0]
        spatial_part = 0.2 * (A @ data[t-1, :, 0])
        noise = np.random.randn(n_nodes) * 2
        
        data[t, :, 0] = temporal_part + spatial_part + 0.1 * daily_pattern + noise
    
    # Add time of day feature
    tod = np.array([i % steps_per_day / steps_per_day for i in range(n_timesteps)])
    tod_tiled = np.tile(tod.reshape(-1, 1, 1), [1, n_nodes, 1]).astype(np.float32)
    
    # Add day of week feature
    dow = np.array([(i // steps_per_day) % 7 / 7 for i in range(n_timesteps)])
    dow_tiled = np.tile(dow.reshape(-1, 1, 1), [1, n_nodes, 1]).astype(np.float32)
    
    # Stack features: [flow, tod, dow]
    full_data = np.concatenate([data, tod_tiled, dow_tiled], axis=-1)  # (T, N, 3)
    print(f"Dataset shape: {full_data.shape}")
    
    # Save as memmap
    fp = np.memmap(f"{save_dir}/{name}/data.dat", dtype='float32', mode='w+', shape=full_data.shape)
    fp[:] = full_data[:]
    fp.flush()
    del fp
    
    # Compute mean/std for scaler
    train_end = int(n_timesteps * 0.6)
    mean = full_data[:train_end, :, 0:1].mean()
    std = full_data[:train_end, :, 0:1].std()
    
    # Save description
    desc = {
        "name": name,
        "domain": "traffic flow",
        "shape": list(full_data.shape),
        "num_time_steps": n_timesteps,
        "num_nodes": n_nodes,
        "num_features": 3,
        "feature_description": ["traffic flow", "time of day", "day of week"],
        "regular_settings": {
            "INPUT_LEN": 12,
            "OUTPUT_LEN": 12,
            "TRAIN_VAL_TEST_RATIO": [0.6, 0.2, 0.2],
            "NORM_EACH_CHANNEL": False,
            "RESCALE": True,
            "METRICS": ["MAE", "RMSE", "MAPE"],
            "NULL_VAL": 0.0
        },
        "scaler": {
            "mean": float(mean),
            "std": float(std)
        }
    }
    with open(f"{save_dir}/{name}/desc.json", 'w') as f:
        json.dump(desc, f, indent=2)
    
    print(f"Dataset '{name}' created at {save_dir}/{name}/")
    print(f"  Mean: {mean:.3f}, Std: {std:.3f}")
    return full_data.shape

if __name__ == "__main__":
    import sys
    os.chdir(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    create_mini_dataset("MiniPEMS", n_nodes=50, n_timesteps=3000)
    print("Done!")
