import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd
import os

from src.datasets.LSTMDataset import SimpleSIRDataset
from src.models.GRUForecast import GRUForecast


# --------------------------------------------------
# Device
# --------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# --------------------------------------------------
# Hyperparameters
# --------------------------------------------------
K = 8
T_out = 16
stride = 16

batch_size = 128
epochs = 100
lr = 1e-3

# --------------------------------------------------
# Model Save Path
# --------------------------------------------------
model_save_path = "outputs/models/best_gru_Osc_16.pt"


# --------------------------------------------------
# Load data
# --------------------------------------------------
states_df = pd.read_csv("data/raw/states.csv")
params_df = pd.read_csv("data/raw/params.csv")

run_ids = sorted(states_df["run_id"].unique())
n = len(run_ids)


# --------------------------------------------------
# Split (64 / 16 / 20)
# --------------------------------------------------
train_ids = run_ids[:int(0.64 * n)]
val_ids   = run_ids[int(0.64 * n):int(0.80 * n)]
test_ids  = run_ids[int(0.80 * n):]

print(f"Train runs: {len(train_ids)}")
print(f"Val runs: {len(val_ids)}")
print(f"Test runs: {len(test_ids)}")


# --------------------------------------------------
# NORMALIZATION (TRAIN ONLY)
# --------------------------------------------------
train_states = states_df[states_df["run_id"].isin(train_ids)]
train_params = params_df[params_df["run_id"].isin(train_ids)]

state_values = train_states[["S","I","R"]].values.astype(float)
param_values = train_params[[
    "transmissionProbability",
    "meanTimeToRecover",
    "meanImmunityDuration"
]].values.astype(float)

state_mean = state_values.mean(axis=0)
state_std  = state_values.std(axis=0) + 1e-8

param_mean = param_values.mean(axis=0)
param_std  = param_values.std(axis=0) + 1e-8


# save normalization
os.makedirs("outputs", exist_ok=True)

norm_path = "outputs/normalization_stats.pt"

torch.save({
    "state_mean": state_mean,
    "state_std": state_std,
    "param_mean": param_mean,
    "param_std": param_std
}, norm_path)

print("Normalization stats saved!")


# --------------------------------------------------
# Datasets
# --------------------------------------------------
train_dataset = SimpleSIRDataset(train_ids, norm_path, K, T_out, stride)
val_dataset   = SimpleSIRDataset(val_ids,   norm_path, K, T_out, stride)


train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,  num_workers=0)
val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False, num_workers=0)


# --------------------------------------------------
# Model (فقط این خط عوض شده)
# --------------------------------------------------
model = GRUForecast(T_out=T_out).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=lr)
loss_fn = nn.MSELoss()


# --------------------------------------------------
# Training
# --------------------------------------------------
best_val_loss = float("inf")

for epoch in range(epochs):

    # ------------------------
    # Train
    # ------------------------
    model.train()
    train_loss = 0

    for x, y, p in train_loader:

        x = x.to(device)
        y = y.to(device)
        p = p.to(device)

        pred = model(x, p)
        loss = loss_fn(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    train_loss /= len(train_loader)


    # ------------------------
    # Validation
    # ------------------------
    model.eval()
    val_loss = 0

    with torch.no_grad():
        for x, y, p in val_loader:

            x = x.to(device)
            y = y.to(device)
            p = p.to(device)

            pred = model(x, p)
            loss = loss_fn(pred, y)

            val_loss += loss.item()

    val_loss /= len(val_loader)


    print(f"Epoch {epoch} | Train: {train_loss:.4f} | Val: {val_loss:.4f}")


    # ------------------------
    # Save best model
    # ------------------------
    if val_loss < best_val_loss:
        best_val_loss = val_loss

        os.makedirs("outputs/models", exist_ok=True)

        torch.save(model.state_dict(), model_save_path)

        print("Best model saved!")


print("Training finished.")