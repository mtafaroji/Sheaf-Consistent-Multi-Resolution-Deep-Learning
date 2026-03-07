import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Subset

from src.datasets.multi_res_normalized_sir_dataset import MultiResSIRDataset
from src.models.multi_res_forecast_net import ResolutionForecastNet


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# --------------------------------------------------
# hyperparameters
# --------------------------------------------------

K = 32
T_out = 16
stride = 8

norm_stats_path = "outputs/normalization_stats.pt"

# --------------------------------------------------
# datasets
# --------------------------------------------------

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

# --------------------------------------------------
# run split (same as training)
# --------------------------------------------------

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

test_Aj  = Subset(dataset_Aj, test_idx)
test_Aj1 = Subset(dataset_Aj1, test_idx)
test_Aj2 = Subset(dataset_Aj2, test_idx)

loader_Aj  = DataLoader(test_Aj,  batch_size=windows_per_run, shuffle=False)
loader_Aj1 = DataLoader(test_Aj1, batch_size=windows_per_run, shuffle=False)
loader_Aj2 = DataLoader(test_Aj2, batch_size=windows_per_run, shuffle=False)

# --------------------------------------------------
# models
# --------------------------------------------------

model_Aj  = ResolutionForecastNet(K, T_out).to(device)
model_Aj1 = ResolutionForecastNet(K, T_out).to(device)
model_Aj2 = ResolutionForecastNet(K, T_out).to(device)

model_Aj.load_state_dict(torch.load("outputs/models/best_Aj.pt", map_location=device))
model_Aj1.load_state_dict(torch.load("outputs/models/best_Aj1.pt", map_location=device))
model_Aj2.load_state_dict(torch.load("outputs/models/best_Aj2.pt", map_location=device))

model_Aj.eval()
model_Aj1.eval()
model_Aj2.eval()

# --------------------------------------------------
# trajectory reconstruction function
# --------------------------------------------------

def reconstruct(loader, model):

    batch = list(loader)[0]

    x = batch["x_in"].to(device)
    y_true = batch["x_out"].cpu().numpy()

    with torch.no_grad():
        y_pred = model(x).cpu().numpy()

    windows = y_pred.shape[0]

    T_total = stride*(windows-1) + T_out

    pred_traj = np.zeros((T_total,3))
    count = np.zeros((T_total,3))

    for w in range(windows):

        start = w*stride
        end   = start + T_out

        pred_traj[start:end] += y_pred[w]
        count[start:end] += 1

    pred_traj /= count

    true_traj = np.zeros((T_total,3))
    count = np.zeros((T_total,3))

    for w in range(windows):

        start = w*stride
        end   = start + T_out

        true_traj[start:end] += y_true[w]
        count[start:end] += 1

    true_traj /= count

    return true_traj, pred_traj


# --------------------------------------------------
# reconstruct all resolutions
# --------------------------------------------------

true_Aj,  pred_Aj  = reconstruct(loader_Aj,  model_Aj)
true_Aj1, pred_Aj1 = reconstruct(loader_Aj1, model_Aj1)
true_Aj2, pred_Aj2 = reconstruct(loader_Aj2, model_Aj2)


# --------------------------------------------------
# plotting function
# --------------------------------------------------

def plot_traj(true_traj, pred_traj, title):

    t = range(len(true_traj))

    plt.figure(figsize=(12,6))

    plt.plot(t,true_traj[:,0],label="True S")
    plt.plot(t,pred_traj[:,0],"--",label="Pred S")

    plt.plot(t,true_traj[:,1],label="True I")
    plt.plot(t,pred_traj[:,1],"--",label="Pred I")

    plt.plot(t,true_traj[:,2],label="True R")
    plt.plot(t,pred_traj[:,2],"--",label="Pred R")

    plt.title(title)
    plt.xlabel("time")
    plt.ylabel("normalized state")

    plt.legend()

    plt.show()


# --------------------------------------------------
# plots
# --------------------------------------------------

plot_traj(true_Aj,  pred_Aj,  "Trajectory reconstruction (Aj)")
plot_traj(true_Aj1, pred_Aj1, "Trajectory reconstruction (Aj1)")
plot_traj(true_Aj2, pred_Aj2, "Trajectory reconstruction (Aj2)")