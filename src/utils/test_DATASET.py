import os
import numpy as np
from numpy.ma import count
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import matplotlib.pyplot as plt

from src.datasets.multi_res_normalized_sir_dataset import MultiResSIRDataset
from src.models.multi_res_forecast_net import ResolutionForecastNet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# --------------------------------------------------
# Hyperparameters
# --------------------------------------------------
K = 32
T_out = 16
stride = 8
epochs = 100
lr = 1e-3

os.makedirs("outputs/models", exist_ok=True)

# --------------------------------------------------
# Extract runs and windows_per_run
# --------------------------------------------------
states_df = pd.read_csv("data/raw/states.csv")

run_ids = sorted(states_df["run_id"].unique())
runs = len(run_ids)

first_run = states_df[states_df["run_id"] == run_ids[0]]
time_length = len(first_run)

windows_per_run = (time_length - K - T_out) // stride + 1

print("Runs:", runs)
print("Time length per run:", time_length)
print("Windows per run:", windows_per_run)

# --------------------------------------------------
# Split 64 / 16 / 20
# --------------------------------------------------
train_runs = int(0.64 * runs)
val_runs   = int(0.16 * runs)
test_runs  = runs - train_runs - val_runs

train_run_ids = run_ids[:train_runs]
val_run_ids   = run_ids[train_runs:train_runs+val_runs]
test_run_ids  = run_ids[train_runs+val_runs:]

# --------------------------------------------------
# Compute normalization stats ONLY from train runs
# --------------------------------------------------
params_df = pd.read_csv("data/raw/params.csv")

train_params = params_df[params_df["run_id"].isin(train_run_ids)][[
    "transmissionProbability",
    "meanTimeToRecover",
    "meanImmunityDuration"
]].values.astype(np.float32)

param_mean = train_params.mean(axis=0)
param_std  = train_params.std(axis=0) + 1e-8

norm_stats = {
    "param_mean": param_mean,
    "param_std": param_std
}

norm_path = "outputs/normalization_stats.pt"
torch.save(norm_stats, norm_path)

print("Saved normalization stats to:", norm_path)

# --------------------------------------------------
# Datasets (3 resolutions)
# --------------------------------------------------
dataset_Aj = MultiResSIRDataset(
    "data/raw/params.csv",
    "data/raw/states.csv",
    resolution="Aj",
    K=K,
    T_out=T_out,
    stride=stride,
    norm_stats_path=norm_path
)

dataset_Aj1 = MultiResSIRDataset(
    "data/raw/params.csv",
    "data/raw/states.csv",
    resolution="Aj1",
    K=K,
    T_out=T_out,
    stride=stride,
    norm_stats_path=norm_path
)

dataset_Aj2 = MultiResSIRDataset(
    "data/raw/params.csv",
    "data/raw/states.csv",
    resolution="Aj2",
    K=K,
    T_out=T_out,
    stride=stride,
    norm_stats_path=norm_path
)

# --------------------------------------------------
# Convert run -> dataset indices
# --------------------------------------------------
def run_to_indices(start_run, num_runs):
    idx = []
    for r in range(start_run, start_run + num_runs):
        start = r * windows_per_run
        end   = start + windows_per_run
        idx.extend(range(start, end))
    return idx

#train_idx = run_to_indices(0, train_runs)
#val_idx   = run_to_indices(train_runs, val_runs)
test_idx  = run_to_indices(train_runs + val_runs, 1)
trajectory_run = states_df[states_df["run_id"] == run_ids[train_runs + val_runs]]
# Subsets
#train_Aj  = Subset(dataset_Aj, train_idx)
#val_Aj    = Subset(dataset_Aj, val_idx)
#test_Aj   = Subset(dataset_Aj, test_idx)

#train_Aj1 = Subset(dataset_Aj1, train_idx)
#val_Aj1   = Subset(dataset_Aj1, val_idx)
#test_Aj1  = Subset(dataset_Aj1, test_idx)

#train_Aj2 = Subset(dataset_Aj2, train_idx)
#val_Aj2   = Subset(dataset_Aj2, val_idx)
test_Aj2  = Subset(dataset_Aj2, test_idx)

# --------------------------------------------------
# DataLoaders
# --------------------------------------------------
#train_loader_Aj  = DataLoader(train_Aj,  batch_size=windows_per_run, shuffle=False)
#train_loader_Aj1 = DataLoader(train_Aj1, batch_size=windows_per_run, shuffle=False)
#train_loader_Aj2 = DataLoader(train_Aj2, batch_size=windows_per_run, shuffle=False)

#val_loader_Aj  = DataLoader(val_Aj,  batch_size=windows_per_run, shuffle=False)
#val_loader_Aj1 = DataLoader(val_Aj1, batch_size=windows_per_run, shuffle=False)
#val_loader_Aj2 = DataLoader(val_Aj2, batch_size=windows_per_run, shuffle=False)

#test_loader_Aj  = DataLoader(test_Aj,  batch_size=windows_per_run, shuffle=False)
#test_loader_Aj1 = DataLoader(test_Aj1, batch_size=windows_per_run, shuffle=False)
test_loader_Aj2 = DataLoader(test_Aj2, batch_size=windows_per_run, shuffle=False)

# --------------------------------------------------
# Shape check
# --------------------------------------------------
batch = next(iter(test_loader_Aj2))
print("x_in shape :", batch["x_in"].shape)
print("x_out shape:", batch["x_out"].shape)


# --------------------------------------------------
# Get data helper
# --------------------------------------------------
def get_data(loader):
    with torch.no_grad():
        for batch in loader:
            x = batch["x_in"].to(device)
            y = batch["x_out"].to(device)
            
    return x, y



def reconstruct(loader):

    batch = next(iter(loader))

    x = batch["x_in"].to(device)

    windows = x.shape[0]

    T_total = stride*(windows-1) + T_out


    true_traj = np.zeros((T_total,3))
    count = np.zeros((T_total,3))

    for w in range(windows):

        start = w*stride
        end   = start + T_out

        true_traj[start:end] += x[w][:T_out, :3].cpu().numpy()
        count[start:end] += 1
    count[count==0] = 1
    true_traj /= count

    return true_traj


trajectory = reconstruct(test_loader_Aj2)

true_states = trajectory_run[["S","I","R"]].values
N = true_states[0].sum()
#N = 1
true_states = true_states / N

# ------------------------
# Plot
# ------------------------
time = np.arange(trajectory.shape[0])

fig, axes = plt.subplots(3, 1, figsize=(10, 8))

labels = ["S", "I", "R"]

for dim in range(3):

    axes[dim].plot(time, trajectory[:, dim] , label="Original", linewidth=2)

    
    axes[dim].plot(time, true_states[:len(time), dim], label="trajectory_run")
    


    axes[dim].set_title(labels[dim])
    axes[dim].legend()

plt.tight_layout()
plt.show()
