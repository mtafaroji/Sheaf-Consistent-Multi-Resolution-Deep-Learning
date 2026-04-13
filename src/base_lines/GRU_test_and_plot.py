import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.datasets.GRUDataset import SimpleSIRDataset
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
M = 0   # index of test run for plotting

norm_path = "outputs/normalization_stats.pt"
model_path = "outputs/models/best_gru_Osc_16.pt"


# --------------------------------------------------
# Load normalization stats
# --------------------------------------------------
stats = torch.load(norm_path, map_location="cpu")
state_mean = stats["state_mean"]
state_std  = stats["state_std"]
param_mean = stats["param_mean"]
param_std  = stats["param_std"]


# --------------------------------------------------
# Load data
# --------------------------------------------------
states_df = pd.read_csv("data/raw/states.csv")
states_df = states_df.sort_values(["run_id", "t"])

params_df = pd.read_csv("data/raw/params.csv")

run_ids = sorted(states_df["run_id"].unique())
n = len(run_ids)

# split
test_ids = run_ids[int(0.80 * n):]


# --------------------------------------------------
# Dataset (for MSE)
# --------------------------------------------------
test_dataset = SimpleSIRDataset(
    test_ids,
    norm_path,
    K,
    T_out,
    stride=stride
)


# --------------------------------------------------
# Load model
# --------------------------------------------------
model = GRUForecast(T_out=T_out).to(device)
model.load_state_dict(torch.load(
    model_path,
    map_location=device,
    weights_only=True
))
model.eval()


# --------------------------------------------------
# Compute Test MSE
# --------------------------------------------------
mse_total = 0

for i in range(len(test_dataset)):
    x, y, p = test_dataset[i]

    x = x.unsqueeze(0).to(device)
    y = y.unsqueeze(0).to(device)
    p = p.unsqueeze(0).to(device)

    with torch.no_grad():
        pred = model(x, p)

    mse = ((pred - y) ** 2).mean().item()
    mse_total += mse

mse_avg = mse_total / len(test_dataset)

print("Test MSE:", mse_avg)


# --------------------------------------------------
# Plot one trajectory
# --------------------------------------------------
run_id = test_ids[M]
print("Plotting run_id:", run_id)

run_df = states_df[states_df["run_id"] == run_id].sort_values("t")
traj = run_df[["S", "I", "R"]].values.astype(np.float32)

# normalize states
traj_norm = (traj - state_mean) / (state_std + 1e-8)

# get params
p = params_df[params_df["run_id"] == run_id][
    ["transmissionProbability", "meanTimeToRecover", "meanImmunityDuration"]
].values[0].astype(np.float32)

p_norm = (p - param_mean) / (param_std + 1e-8)

T = len(traj_norm)

# reconstruction buffers
pred_sum = np.zeros((T, 3), dtype=np.float32)
pred_count = np.zeros((T, 3), dtype=np.float32)

# sliding window prediction
for i in range(0, T - K - T_out + 1, stride):
    x = traj_norm[i:i+K]

    x_t = torch.tensor(x, dtype=torch.float32).unsqueeze(0).to(device)
    p_t = torch.tensor(p_norm, dtype=torch.float32).unsqueeze(0).to(device)

    with torch.no_grad():
        pred = model(x_t, p_t).cpu().numpy().squeeze(0)

    start = i + K
    end = i + K + T_out

    pred_sum[start:end] += pred
    pred_count[start:end] += 1


# average overlapping predictions
mask = pred_count[:, 0] > 0
pred_full = np.zeros_like(traj_norm)
pred_full[mask] = pred_sum[mask] / pred_count[mask]


# --------------------------------------------------
# Plot
# --------------------------------------------------
valid_idx = np.where(mask)[0]
start = valid_idx.min()
end = min(start + 240, valid_idx.max() + 1)

t = np.arange(T)

titles = [
    "S predicted vs Original",
    "I predicted vs Original",
    "R predicted vs Original"
]


plt.figure(figsize=(10, 6))
names = ["S", "I", "R"]
colors = ["blue", "red", "green"]


for j in range(3):
    plt.plot(t[start:end], traj_norm[start:end, j],
             color=colors[j],label=f"True {names[j]}")
    
    plt.plot(t[start:end], pred_full[start:end, j], "--",
             color=colors[j], label=f"Pred {names[j]}")

plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
plt.grid()
plt.title("True vs Pred (All states)")
plt.xlabel("Time")
plt.ylabel("State")

plt.tight_layout()
plt.show()


np.savez("outputs/data/GRU.npz", pred_GRU=pred_full[start:end, :])

"""

fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=False)
names = ["S", "I", "R"]

for j in range(3):
    axes[j].plot(t[start:end], traj_norm[start:end, j], label=f"True {names[j]}")
    axes[j].plot(t[start:end], pred_full[start:end, j], "--", label=f"Pred {names[j]}")
    axes[j].legend(loc="center left", bbox_to_anchor=(1, 0.5))
    axes[j].grid()
    axes[j].set_title(titles[j])
    axes[j].set_ylabel("state")

axes[-1].set_xlabel("Time")

plt.tight_layout()
plt.show()

"""