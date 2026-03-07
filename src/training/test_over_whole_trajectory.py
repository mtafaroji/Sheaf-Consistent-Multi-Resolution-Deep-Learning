import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Subset
from itertools import islice

from src.datasets.multi_res_normalized_sir_dataset import MultiResSIRDataset

from src.models.multi_res_forecast_net import ResolutionForecastNet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------------------------------
# Hyperparameters (same as training)
# -------------------------------------------------

K = 8
T_out = 88
stride = 88

M = 11

norm_stats_path = "outputs/normalization_stats.pt"

# -------------------------------------------------
# Load datasets
# -------------------------------------------------

dataset_Aj = MultiResSIRDataset(
    "data/raw/params.csv",
    "data/raw/states.csv",
    resolution="Aj",
    K=K,
    T_out=T_out,
    stride=stride,
    norm_stats_path=norm_stats_path
)

dataset_Aj1 = MultiResSIRDataset(
    "data/raw/params.csv",
    "data/raw/states.csv",
    resolution="Aj1",
    K=K,
    T_out=T_out,
    stride=stride,
    norm_stats_path=norm_stats_path
)

dataset_Aj2 = MultiResSIRDataset(
    "data/raw/params.csv",
    "data/raw/states.csv",
    resolution="Aj2",
    K=K,
    T_out=T_out,
    stride=stride,
    norm_stats_path=norm_stats_path
)

# -------------------------------------------------
# Split runs
# -------------------------------------------------

states_df = pd.read_csv("data/raw/states.csv")

run_ids = sorted(states_df["run_id"].unique())
runs = len(run_ids)

first_run = states_df[states_df["run_id"] == run_ids[0]]

time_length = len(first_run)

windows_per_run = (time_length - K - T_out) // stride + 1

train_runs = int(0.64 * runs)
val_runs   = int(0.16 * runs)
test_runs  = runs - train_runs - val_runs


def run_to_indices(start_run, num_runs):

    idx = []

    for r in range(start_run, start_run + num_runs):

        start = r * windows_per_run
        end   = start + windows_per_run

        idx.extend(range(start, end))

    return idx


test_idx = run_to_indices(train_runs + val_runs, test_runs)

loader_Aj  = DataLoader(Subset(dataset_Aj, test_idx),  batch_size=windows_per_run)
loader_Aj1 = DataLoader(Subset(dataset_Aj1, test_idx), batch_size=windows_per_run)
loader_Aj2 = DataLoader(Subset(dataset_Aj2, test_idx), batch_size=windows_per_run)

# -------------------------------------------------
# Load trained models
# -------------------------------------------------

model_Aj  = ResolutionForecastNet(K, T_out).to(device)
model_Aj1 = ResolutionForecastNet(K, T_out).to(device)
model_Aj2 = ResolutionForecastNet(K, T_out).to(device)

model_Aj.load_state_dict(torch.load("outputs/models/best_Aj.pt", map_location=device))
model_Aj1.load_state_dict(torch.load("outputs/models/best_Aj1.pt", map_location=device))
model_Aj2.load_state_dict(torch.load("outputs/models/best_Aj2.pt", map_location=device))

model_Aj.eval()
model_Aj1.eval()
model_Aj2.eval()

# -------------------------------------------------
# Trajectory reconstruction from overlapping windows
# -------------------------------------------------

def reconstruct(loader, model):

    #batch = next(iter(loader))
    batch = next(islice(loader, M, None))

    x = batch["x_in"].to(device)

    with torch.no_grad():
        y_pred = model(x).cpu().numpy()

    windows = y_pred.shape[0]

    T_total = stride*(windows-1) + T_out

    pred_traj = np.zeros((T_total,3))
    count = np.zeros((T_total,3))

    for w in range(windows):

        start = w * stride
        end   = start + T_out

        pred_traj[start:end] += y_pred[w]
        count[start:end] += 1

    count[count == 0] = 1

    pred_traj = pred_traj / count

    return pred_traj


# -------------------------------------------------
# Reconstruct predictions
# -------------------------------------------------

pred_Aj  = reconstruct(loader_Aj,  model_Aj)
pred_Aj1 = reconstruct(loader_Aj1, model_Aj1)
pred_Aj2 = reconstruct(loader_Aj2, model_Aj2)
#print("shape of pred_Aj :", pred_Aj.shape)

# -------------------------------------------------
# Load and normalize original trajectory
# -------------------------------------------------

run_id = run_ids[train_runs + val_runs + M]

run_df = states_df[states_df["run_id"] == run_id].sort_values("t")

original = run_df[["S","I","R"]].values
N = original[0].sum()
original = original[K:]

# normalize with same rule used in dataset
original = original / N

T_plot = len(pred_Aj)

original = original[:T_plot]

t = range(T_plot)

# -------------------------------------------------
# Plot results
# -------------------------------------------------

fig, axs = plt.subplots(3,1, figsize=(12,10))

titles = [
    "Aj prediction vs Original",
    "Aj1 prediction vs Original",
    "Aj2 prediction vs Original"
]

preds = [pred_Aj, pred_Aj1, pred_Aj2]

fig, axs = plt.subplots(3, 1, figsize=(8, 12))   

for i in range(3):

    axs[i].plot(t, original[:,0], label="True S")
    axs[i].plot(t, preds[i][:,0], "--", label="Pred S")

    axs[i].plot(t, original[:,1], label="True I")
    axs[i].plot(t, preds[i][:,1], "--", label="Pred I")

    axs[i].plot(t, original[:,2], label="True R")
    axs[i].plot(t, preds[i][:,2], "--", label="Pred R")

    axs[i].set_title(titles[i])
    axs[i].set_xlabel("time")
    axs[i].set_ylabel("state")

    axs[i].legend(loc="center left", bbox_to_anchor=(1, 0.5))  # 👈 legend سمت راست

plt.tight_layout()
plt.savefig("outputs/figures/states.png", dpi=300, bbox_inches="tight")
plt.show()