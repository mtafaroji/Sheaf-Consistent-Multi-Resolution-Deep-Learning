import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import matplotlib.pyplot as plt

from src.datasets.all_nor_opti_dataset import MultiResSIRDataset
from src.models.multi_res_forecast_net import ResolutionForecastNet
from src.utils.wavelet_restriction import WaveletRestriction


import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# --------------------------------------------------
# Hyperparameters
# --------------------------------------------------

K = 8
T_out = 248
stride = 248
epochs = 100
lr = 1e-3
max_runs = 200
lambda_c01 = 0.3
lambda_c12 = 0.3
lambda_c10 = 0.3
lambda_c21 = 0.3

os.makedirs("outputs/models", exist_ok=True)
os.makedirs("outputs/figures", exist_ok=True)


# --------------------------------------------------
# Best model paths
# -------------------------------------------------
Aj_file_path = "outputs/models/best_Aj_248_cons_bidirection_noisy.pt"
Aj1_file_path = "outputs/models/best_Aj1_248_cons_bidirection_noisy.pt"
Aj2_file_path = "outputs/models/best_Aj2_248_cons_bidirection_noisy.pt"

# --------------------------------------------------
# Normalization stats path
# ------------------------------------------------
norm_path = "outputs/normalization_stats.pt"

# --------------------------------------------------
# Extract runs
# --------------------------------------------------

states_df = pd.read_csv("data/raw/states.csv")

run_ids = sorted(states_df["run_id"].unique())
runs = len(run_ids)
runs = min(runs, max_runs)

first_run = states_df[states_df["run_id"] == run_ids[0]]
time_length = len(first_run)

windows_per_run = (time_length - K - T_out) // stride + 1

print("Runs:", runs)
print("Time length per run:", time_length)
print("Windows per run:", windows_per_run)

# --------------------------------------------------
# Split dataset
# --------------------------------------------------

train_runs = int(0.64 * runs)
val_runs   = int(0.16 * runs)
test_runs  = runs - train_runs - val_runs

train_run_ids = run_ids[:train_runs]
val_run_ids   = run_ids[train_runs:train_runs+val_runs]
test_run_ids  = run_ids[train_runs+val_runs:]

# --------------------------------------------------
# Compute normalization stats only from train runs
# --------------------------------------------------

params_df = pd.read_csv("data/raw/params.csv")

train_params = params_df[params_df["run_id"].isin(train_run_ids)][[
    "transmissionProbability",
    "meanTimeToRecover",
    "meanImmunityDuration"
]].values.astype(np.float32)

param_mean = train_params.mean(axis=0)
param_std  = train_params.std(axis=0) + 1e-8

train_states = states_df[states_df["run_id"].isin(train_run_ids)][["S", "I", "R"]].values.astype(np.float32)

state_mean = train_states.mean(axis=0)
state_std  = train_states.std(axis=0) + 1e-8

norm_stats = {
    "param_mean": param_mean,
    "param_std": param_std,
    "state_mean": state_mean,
    "state_std": state_std
}


torch.save(norm_stats, norm_path)

print("Saved normalization stats to:", norm_path)
print("param_mean:", param_mean)
print("param_std :", param_std)
print("state_mean:", state_mean)
print("state_std :", state_std)

# --------------------------------------------------
# Dataset
# --------------------------------------------------

dataset = MultiResSIRDataset(
    "data/raw/params.csv",
    "data/raw/states.csv",
    K=K,
    T_out=T_out,
    stride=stride,
    norm_stats_path=norm_path,
    max_runs=max_runs
)

# --------------------------------------------------
# Convert run to indices
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

train_set = Subset(dataset, train_idx)
val_set   = Subset(dataset, val_idx)
test_set  = Subset(dataset, test_idx)

# --------------------------------------------------
# DataLoaders
# --------------------------------------------------

train_loader = DataLoader(train_set, batch_size=windows_per_run, shuffle=False)
val_loader   = DataLoader(val_set, batch_size=windows_per_run, shuffle=False)
test_loader  = DataLoader(test_set, batch_size=windows_per_run, shuffle=False)

# --------------------------------------------------
# Models
# --------------------------------------------------

model_Aj  = ResolutionForecastNet(K, T_out).to(device)
model_Aj1 = ResolutionForecastNet(K, T_out).to(device)
model_Aj2 = ResolutionForecastNet(K, T_out).to(device)


# --------------------------------------------------
# Load Last Training Model Parameters
# --------------------------------------------------
#model_Aj.load_state_dict(torch.load("outputs/models/best_Aj.pt", map_location=device))
#model_Aj1.load_state_dict(torch.load("outputs/models/best_Aj1.pt", map_location=device))
#model_Aj2.load_state_dict(torch.load("outputs/models/best_Aj2.pt", map_location=device))


opt_Aj  = torch.optim.Adam(model_Aj.parameters(), lr=lr)
opt_Aj1 = torch.optim.Adam(model_Aj1.parameters(), lr=lr)
opt_Aj2 = torch.optim.Adam(model_Aj2.parameters(), lr=lr)

criterion = nn.MSELoss()

# --------------------------------------------------
# Restriction operator
# --------------------------------------------------

restriction = WaveletRestriction("db4").to(device)

# --------------------------------------------------
# Reconstruction helper
# --------------------------------------------------

def reconstruct_traj(pred):

    windows = pred.shape[0]
    T_total = stride*(windows-1) + T_out

    traj = torch.zeros(T_total,3,device=device)
    count = torch.zeros(T_total,3,device=device)

    for w in range(windows):

        start = w * stride
        end   = start + T_out

        traj[start:end] += pred[w]
        count[start:end] += 1

    traj = traj / torch.clamp(count,min=1)

    return traj

# --------------------------------------------------
# Evaluation helper
# --------------------------------------------------

def eval_model(model, loader, res):

    model.eval()
    total = 0

    with torch.no_grad():

        for batch in loader:

            p = batch["params"].to(device)

            if res == "Aj":
                x = torch.cat([batch["Aj_in"].to(device),
                               p.unsqueeze(1).repeat(1,K,1)],dim=2)
                y = batch["Aj_out"].to(device)

            elif res == "Aj1":
                x = torch.cat([batch["Aj1_in"].to(device),
                               p.unsqueeze(1).repeat(1,K,1)],dim=2)
                y = batch["Aj1_out"].to(device)

            else:
                x = torch.cat([batch["Aj2_in"].to(device),
                               p.unsqueeze(1).repeat(1,K,1)],dim=2)
                y = batch["Aj2_out"].to(device)

            pred = model(x)

            total += criterion(pred,y).item()

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

    for batch in train_loader:

        p = batch["params"].to(device)

        # -------------------
        # Aj
        # -------------------

        x = torch.cat([batch["Aj_in"].to(device),
                       p.unsqueeze(1).repeat(1,K,1)],dim=2)

        y = batch["Aj_out"].to(device)

        pred_Aj = model_Aj(x)

        loss_Aj = criterion(pred_Aj,y)

        # -------------------
        # Aj1
        # -------------------

        x = torch.cat([batch["Aj1_in"].to(device),
                       p.unsqueeze(1).repeat(1,K,1)],dim=2)

        y1 = batch["Aj1_out"].to(device)

        pred_Aj1 = model_Aj1(x)

        loss_Aj1 = criterion(pred_Aj1,y1)

        # -------------------
        # Aj2
        # -------------------

        x = torch.cat([batch["Aj2_in"].to(device),
                       p.unsqueeze(1).repeat(1,K,1)],dim=2)

        y2 = batch["Aj2_out"].to(device)

        pred_Aj2 = model_Aj2(x)

        loss_Aj2 = criterion(pred_Aj2,y2)

        # ------------------------------------------------
        # Reconstruction
        # ------------------------------------------------

        traj_Aj  = reconstruct_traj(pred_Aj)
        traj_Aj1 = reconstruct_traj(pred_Aj1)
        traj_Aj2 = reconstruct_traj(pred_Aj2)

        # ------------------------------------------------
        # Restriction
        # ------------------------------------------------


        Aj1_from_Aj2 = restriction.remove_D1(traj_Aj2)

        Aj_from_Aj1  = restriction.remove_D2(traj_Aj1)

        # ------------------------------------------------
        # Consistency losses
        # ------------------------------------------------

        #loss_c2 = criterion(Aj1_from_Aj2 , traj_Aj1.detach())
        #loss_c1 = criterion(Aj_from_Aj1  , traj_Aj.detach())

        #loss_total_Aj  = loss_Aj
        #loss_total_Aj1 = loss_Aj1 + loss_c1
        #loss_total_Aj2 = loss_Aj2 + loss_c2


        loss_c12 = criterion(Aj1_from_Aj2 , traj_Aj1.detach())
        loss_c01 = criterion(Aj_from_Aj1  , traj_Aj.detach())


        loss_c21 = criterion(Aj1_from_Aj2.detach() , traj_Aj1)
        loss_c10 = criterion(Aj_from_Aj1.detach()  , traj_Aj)

        # ------------------------------------------------
        # Total losses
        # ------------------------------------------------

        loss_total_Aj  = loss_Aj  + lambda_c10 * loss_c10 
        loss_total_Aj1 = loss_Aj1 + lambda_c21 * loss_c21 + lambda_c01 * loss_c01
        loss_total_Aj2 = loss_Aj2 + lambda_c12 * loss_c12

        # ------------------------------------------------
        # Separate backward
        # ------------------------------------------------

        opt_Aj.zero_grad()
        loss_total_Aj.backward()
        opt_Aj.step()

        opt_Aj1.zero_grad()
        loss_total_Aj1.backward()
        opt_Aj1.step()

        opt_Aj2.zero_grad()
        loss_total_Aj2.backward()
        opt_Aj2.step()

        total_Aj  += loss_total_Aj.item()
        total_Aj1 += loss_total_Aj1.item()
        total_Aj2 += loss_total_Aj2.item()

    avg_train_Aj  = total_Aj  / len(train_loader)
    avg_train_Aj1 = total_Aj1 / len(train_loader)
    avg_train_Aj2 = total_Aj2 / len(train_loader)

    val_Aj  = eval_model(model_Aj , val_loader , "Aj")
    val_Aj1 = eval_model(model_Aj1, val_loader , "Aj1")
    val_Aj2 = eval_model(model_Aj2, val_loader , "Aj2")

    val_sum = val_Aj + val_Aj1 + val_Aj2

    print(f"\nEpoch {epoch+1}")
    print(f"Train: Aj={avg_train_Aj:.4f}  Aj1={avg_train_Aj1:.4f}  Aj2={avg_train_Aj2:.4f}")
    print(f"Val  : Aj={val_Aj:.4f}  Aj1={val_Aj1:.4f}  Aj2={val_Aj2:.4f}")

    if val_sum < best_val:

        best_val = val_sum

        torch.save(model_Aj.state_dict(),
                   Aj_file_path)

        torch.save(model_Aj1.state_dict(),
                   Aj1_file_path)

        torch.save(model_Aj2.state_dict(),
                   Aj2_file_path)

        print("Saved best models")

    history["train_Aj"].append(avg_train_Aj)
    history["val_Aj"].append(val_Aj)
    history["train_Aj1"].append(avg_train_Aj1)
    history["val_Aj1"].append(val_Aj1)
    history["train_Aj2"].append(avg_train_Aj2)
    history["val_Aj2"].append(val_Aj2)



# --------------------------------------------------
# Plot losses
# --------------------------------------------------
plt.figure(figsize=(12,10))

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

plt.savefig("outputs/figures/losses_consistency_16.png")
# --------------------------------------------------
# Test
# --------------------------------------------------

model_Aj.load_state_dict(torch.load(Aj_file_path,map_location=device))
model_Aj1.load_state_dict(torch.load(Aj1_file_path,map_location=device))
model_Aj2.load_state_dict(torch.load(Aj2_file_path,map_location=device))

test_Aj  = eval_model(model_Aj , test_loader , "Aj")
test_Aj1 = eval_model(model_Aj1, test_loader , "Aj1")
test_Aj2 = eval_model(model_Aj2, test_loader , "Aj2")

print("\nFinal Test")
print(f"Aj={test_Aj:.4f}  Aj1={test_Aj1:.4f}  Aj2={test_Aj2:.4f}")

plt.show()