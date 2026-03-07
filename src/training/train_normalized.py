import os
import numpy as np
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
K = 8
T_out = 88
stride = 64
epochs = 60
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

train_idx = run_to_indices(0, train_runs)
val_idx   = run_to_indices(train_runs, val_runs)
test_idx  = run_to_indices(train_runs + val_runs, test_runs)

# Subsets
train_Aj  = Subset(dataset_Aj, train_idx)
val_Aj    = Subset(dataset_Aj, val_idx)
test_Aj   = Subset(dataset_Aj, test_idx)

train_Aj1 = Subset(dataset_Aj1, train_idx)
val_Aj1   = Subset(dataset_Aj1, val_idx)
test_Aj1  = Subset(dataset_Aj1, test_idx)

train_Aj2 = Subset(dataset_Aj2, train_idx)
val_Aj2   = Subset(dataset_Aj2, val_idx)
test_Aj2  = Subset(dataset_Aj2, test_idx)

# --------------------------------------------------
# DataLoaders
# --------------------------------------------------
train_loader_Aj  = DataLoader(train_Aj,  batch_size=windows_per_run, shuffle=False)
train_loader_Aj1 = DataLoader(train_Aj1, batch_size=windows_per_run, shuffle=False)
train_loader_Aj2 = DataLoader(train_Aj2, batch_size=windows_per_run, shuffle=False)

val_loader_Aj  = DataLoader(val_Aj,  batch_size=windows_per_run, shuffle=False)
val_loader_Aj1 = DataLoader(val_Aj1, batch_size=windows_per_run, shuffle=False)
val_loader_Aj2 = DataLoader(val_Aj2, batch_size=windows_per_run, shuffle=False)

test_loader_Aj  = DataLoader(test_Aj,  batch_size=windows_per_run, shuffle=False)
test_loader_Aj1 = DataLoader(test_Aj1, batch_size=windows_per_run, shuffle=False)
test_loader_Aj2 = DataLoader(test_Aj2, batch_size=windows_per_run, shuffle=False)

# --------------------------------------------------
# Shape check
# --------------------------------------------------
batch = next(iter(train_loader_Aj))
print("x_in shape :", batch["x_in"].shape)
print("x_out shape:", batch["x_out"].shape)

# --------------------------------------------------
# Models
# --------------------------------------------------
model_Aj  = ResolutionForecastNet(K, T_out).to(device)
model_Aj1 = ResolutionForecastNet(K, T_out).to(device)
model_Aj2 = ResolutionForecastNet(K, T_out).to(device)

opt_Aj  = torch.optim.Adam(model_Aj.parameters(), lr=lr)
opt_Aj1 = torch.optim.Adam(model_Aj1.parameters(), lr=lr)
opt_Aj2 = torch.optim.Adam(model_Aj2.parameters(), lr=lr)

criterion = nn.MSELoss()

# --------------------------------------------------
# Evaluation helper
# --------------------------------------------------
def eval_model(model, loader):
    model.eval()
    total = 0
    with torch.no_grad():
        for batch in loader:
            x = batch["x_in"].to(device)
            y = batch["x_out"].to(device)
            pred = model(x)
            total += criterion(pred, y).item()
    return total / len(loader)

# --------------------------------------------------
# Training
# --------------------------------------------------
best_val = float("inf")

history = {
    "train_Aj": [],
    "val_Aj": [],
    "train_Aj1": [],
    "val_Aj1": [],
    "train_Aj2": [],
    "val_Aj2": []
}

for epoch in range(epochs):

    model_Aj.train()
    model_Aj1.train()
    model_Aj2.train()

    total_Aj = 0
    total_Aj1 = 0
    total_Aj2 = 0
    yy = 0
    for batch_Aj, batch_Aj1, batch_Aj2 in zip(
        train_loader_Aj, train_loader_Aj1, train_loader_Aj2):

        # Aj
        x = batch_Aj["x_in"].to(device)
        y = batch_Aj["x_out"].to(device)
        opt_Aj.zero_grad()
        loss = criterion(model_Aj(x), y)
        loss.backward()
        opt_Aj.step()
        total_Aj += loss.item()
        if yy == len(train_loader_Aj) - 2:
            final_loss_Aj = loss.item()
   

        # Aj1
        x = batch_Aj1["x_in"].to(device)
        y = batch_Aj1["x_out"].to(device)
        opt_Aj1.zero_grad()
        loss = criterion(model_Aj1(x), y)
        loss.backward()
        opt_Aj1.step()
        total_Aj1 += loss.item()
        if yy == len(train_loader_Aj1) - 2:
            final_loss_Aj1 = loss.item()

        # Aj2
        x = batch_Aj2["x_in"].to(device)
        y = batch_Aj2["x_out"].to(device)
        opt_Aj2.zero_grad()
        loss = criterion(model_Aj2(x), y)
        loss.backward()
        opt_Aj2.step()
        total_Aj2 += loss.item()
        if yy == len(train_loader_Aj2) - 2:
            final_loss_Aj2 = loss.item()

        yy += 1

    avg_train_Aj  = total_Aj  / len(train_loader_Aj)
    avg_train_Aj1 = total_Aj1 / len(train_loader_Aj1)
    avg_train_Aj2 = total_Aj2 / len(train_loader_Aj2)

    val_Aj  = eval_model(model_Aj,  val_loader_Aj)
    val_Aj1 = eval_model(model_Aj1, val_loader_Aj1)
    val_Aj2 = eval_model(model_Aj2, val_loader_Aj2)

    val_sum = val_Aj + val_Aj1 + val_Aj2

    print(f"\nEpoch {epoch+1}")
    print(f"Train: Aj={avg_train_Aj:.4f}  Aj1={avg_train_Aj1:.4f}  Aj2={avg_train_Aj2:.4f}")
    print(f"Val  : Aj={val_Aj:.4f}  Aj1={val_Aj1:.4f}  Aj2={val_Aj2:.4f}")

    if val_sum < best_val:
        best_val = val_sum
        torch.save(model_Aj.state_dict(),  "outputs/models/best_Aj.pt")
        torch.save(model_Aj1.state_dict(), "outputs/models/best_Aj1.pt")
        torch.save(model_Aj2.state_dict(), "outputs/models/best_Aj2.pt")
        print("Saved best models")

    history["train_Aj"].append(final_loss_Aj)
    history["val_Aj"].append(val_Aj)
    history["train_Aj1"].append(final_loss_Aj1)
    history["val_Aj1"].append(val_Aj1)
    history["train_Aj2"].append(final_loss_Aj2)
    history["val_Aj2"].append(val_Aj2)
   
# --------------------------------------------------
# Plot losses
# --------------------------------------------------
"""
plt.figure(figsize=(12,4))

plt.plot(history["train_Aj"], label="Train Aj")
plt.plot(history["val_Aj"], label="Val Aj")

plt.plot(history["train_Aj1"], label="Train Aj1")
plt.plot(history["val_Aj1"], label="Val Aj1")

plt.plot(history["train_Aj2"], label="Train Aj2")
plt.plot(history["val_Aj2"], label="Val Aj2")

plt.legend()
plt.title("Training and Validation Loss")
plt.savefig("outputs/losses.png")

"""

import matplotlib.pyplot as plt

plt.figure(figsize=(5,4))

plt.plot(history["train_Aj"], label="Train Aj")
plt.plot(history["val_Aj"], label="Val Aj")

plt.plot(history["train_Aj1"], label="Train Aj1")
plt.plot(history["val_Aj1"], label="Val Aj1")

plt.plot(history["train_Aj2"], label="Train Aj2")
plt.plot(history["val_Aj2"], label="Val Aj2")

plt.xlabel("Epoch Number")
plt.ylabel("Loss")

plt.legend()
plt.title("Training and Validation Loss")

plt.savefig("outputs/figures/losses.png")

# --------------------------------------------------
# Test
# --------------------------------------------------
model_Aj.load_state_dict(torch.load("outputs/models/best_Aj.pt", map_location=device))
model_Aj1.load_state_dict(torch.load("outputs/models/best_Aj1.pt", map_location=device))
model_Aj2.load_state_dict(torch.load("outputs/models/best_Aj2.pt", map_location=device))

test_Aj  = eval_model(model_Aj,  test_loader_Aj)
test_Aj1 = eval_model(model_Aj1, test_loader_Aj1)
test_Aj2 = eval_model(model_Aj2, test_loader_Aj2)

print("\nFinal Test")
print(f"Aj={test_Aj:.4f}  Aj1={test_Aj1:.4f}  Aj2={test_Aj2:.4f}")

plt.show()