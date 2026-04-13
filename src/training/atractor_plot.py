import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Subset
from itertools import islice

from src.datasets.all_nor_opti_dataset import MultiResSIRDataset
from src.models.multi_res_forecast_net import ResolutionForecastNet

from mpl_toolkits.mplot3d import Axes3D

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------------------------------
# Hyperparameters
# -------------------------------------------------
K = 8
T_out = 128
stride = 128
max_runs = 100
M = 10   # شماره ران تست

# ✅ بازه‌ای که میخوای رسم بشه
start = 0
#end   = 200


norm_stats_path = "outputs/normalization_stats.pt"

Aj_file_path  = "outputs/models/best_Aj_Chua_128_cons.pt"
Aj1_file_path = "outputs/models/best_Aj1_Chua_128_cons.pt"
Aj2_file_path = "outputs/models/best_Aj2_Chua_128_cons.pt"

# -------------------------------------------------
# Load dataset
# -------------------------------------------------
dataset = MultiResSIRDataset(
    "data/raw/params.csv",
    "data/raw/states.csv",
    K=K,
    T_out=T_out,
    stride=stride,
    norm_stats_path=norm_stats_path,
    max_runs=max_runs
)

states_df = pd.read_csv("data/raw/states.csv")

run_ids = sorted(states_df["run_id"].unique())
runs = min(len(run_ids), max_runs)

first_run = states_df[states_df["run_id"] == run_ids[0]].sort_values("t")
time_length = len(first_run)

windows_per_run = (time_length - K - T_out) // stride + 1

train_runs = int(0.64 * runs)
val_runs   = int(0.16 * runs)

def run_to_indices(start_run, num_runs):
    idx = []
    for r in range(start_run, start_run + num_runs):
        start = r * windows_per_run
        end = start + windows_per_run
        idx.extend(range(start, end))
    return idx

test_idx = run_to_indices(train_runs + val_runs, runs - train_runs - val_runs)
test_subset = Subset(dataset, test_idx)

plot_loader = DataLoader(test_subset, batch_size=windows_per_run, shuffle=False)

# -------------------------------------------------
# Load models
# -------------------------------------------------
model_Aj  = ResolutionForecastNet(K, T_out).to(device)
model_Aj1 = ResolutionForecastNet(K, T_out).to(device)
model_Aj2 = ResolutionForecastNet(K, T_out).to(device)

model_Aj.load_state_dict(torch.load(Aj_file_path, map_location=device, weights_only=True))
model_Aj1.load_state_dict(torch.load(Aj1_file_path, map_location=device, weights_only=True))
model_Aj2.load_state_dict(torch.load(Aj2_file_path, map_location=device, weights_only=True))

model_Aj.eval()
model_Aj1.eval()
model_Aj2.eval()

# -------------------------------------------------
# Reconstruction
# -------------------------------------------------
def reconstruct(loader, model, res):
    batch = next(islice(loader, M, None))
    p = batch["params"].to(device)

    if res == "Aj":
        x = torch.cat([batch["Aj_in"].to(device), p.unsqueeze(1).repeat(1, K, 1)], dim=2)
    elif res == "Aj1":
        x = torch.cat([batch["Aj1_in"].to(device), p.unsqueeze(1).repeat(1, K, 1)], dim=2)
    else:
        x = torch.cat([batch["Aj2_in"].to(device), p.unsqueeze(1).repeat(1, K, 1)], dim=2)

    with torch.no_grad():
        y_pred = model(x).cpu().numpy()

    windows = y_pred.shape[0]
    T_total = stride * (windows - 1) + T_out

    pred_traj = np.zeros((T_total, 3))
    count = np.zeros((T_total, 3))

    for w in range(windows):
        s = w * stride
        e = s + T_out
        pred_traj[s:e] += y_pred[w]
        count[s:e] += 1

    pred_traj = pred_traj / np.maximum(count, 1)
    return pred_traj

# -------------------------------------------------
# Predictions
# -------------------------------------------------
pred_Aj  = reconstruct(plot_loader, model_Aj, "Aj")
pred_Aj1 = reconstruct(plot_loader, model_Aj1, "Aj1")
pred_Aj2 = reconstruct(plot_loader, model_Aj2, "Aj2")

# -------------------------------------------------
# Load TRUE trajectory
# -------------------------------------------------
run_id = run_ids[train_runs + val_runs + M]
run_df = states_df[states_df["run_id"] == run_id].sort_values("t")

original = run_df[["S", "I", "R"]].values
original = original[K:]
original = (original - dataset.state_mean) / dataset.state_std

# -------------------------------------------------
# Match length
# -------------------------------------------------
T = min(len(original), len(pred_Aj))
original = original[:T]
pred_Aj  = pred_Aj[:T]
pred_Aj1 = pred_Aj1[:T]
pred_Aj2 = pred_Aj2[:T]

# -------------------------------------------------
# Select range (IMPORTANT)
# -------------------------------------------------
end = len(original)
end = min(end, T)


original = original[start:end]
pred_Aj  = pred_Aj[start:end]
pred_Aj1 = pred_Aj1[start:end]
pred_Aj2 = pred_Aj2[start:end]

# -------------------------------------------------
# 3D ATTRACTOR (LINE)
# -------------------------------------------------
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

ax.plot(original[:,0], original[:,1], original[:,2], linewidth=2, label="True")
ax.plot(pred_Aj[:,0], pred_Aj[:,1], pred_Aj[:,2], "--", label="Aj")
ax.plot(pred_Aj1[:,0], pred_Aj1[:,1], pred_Aj1[:,2], "--", label="Aj1")
ax.plot(pred_Aj2[:,0], pred_Aj2[:,1], pred_Aj2[:,2], "--", label="Aj2")

ax.set_xlabel("S")
ax.set_ylabel("I")
ax.set_zlabel("R")

ax.set_title(f"3D Attractor (Run {run_id})")

ax.legend()
plt.tight_layout()
plt.show()