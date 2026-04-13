import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Subset
from itertools import islice

from src.datasets.all_nor_opti_dataset import MultiResSIRDataset
from src.models.multi_res_forecast_net import ResolutionForecastNet

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------------------------------
# Hyperparameters
# -------------------------------------------------
K = 8
T_out = 248
stride = 248
max_runs = 200
M = 0   # index of one test run to plot

norm_stats_path = "outputs/normalization_stats.pt"

Aj_file_path  = "outputs/models/best_Aj_248_cons_bidirection_noisy.pt"
Aj1_file_path = "outputs/models/best_Aj1_248_cons_bidirection_noisy.pt"
Aj2_file_path = "outputs/models/best_Aj2_248_cons_bidirection_noisy.pt"

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

# -------------------------------------------------
# Split runs
# -------------------------------------------------
states_df = pd.read_csv("data/raw/states_dn.csv")

run_ids = sorted(states_df["run_id"].unique())
runs = min(len(run_ids), max_runs)

first_run = states_df[states_df["run_id"] == run_ids[0]].sort_values("t")
time_length = len(first_run)

windows_per_run = (time_length - K - T_out) // stride + 1

train_runs = int(0.64 * runs)
val_runs   = int(0.16 * runs)
test_runs  = runs - train_runs - val_runs

def run_to_indices(start_run, num_runs):
    idx = []
    for r in range(start_run, start_run + num_runs):
        start = r * windows_per_run
        end = start + windows_per_run
        idx.extend(range(start, end))
    return idx

test_idx = run_to_indices(train_runs + val_runs, test_runs)
test_subset = Subset(dataset, test_idx)

# batch_size=1 for easy average test MSE over all windows
test_loader = DataLoader(test_subset, batch_size=1, shuffle=False)

# separate loader for plotting one full run
plot_loader = DataLoader(test_subset, batch_size=windows_per_run, shuffle=False)

# -------------------------------------------------
# Load trained models
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
# Compute average Test MSE over all test windows
# -------------------------------------------------
mse_Aj_total = 0.0
mse_Aj1_total = 0.0
mse_Aj2_total = 0.0
n_samples = 0

with torch.no_grad():
    for batch in test_loader:
        p = batch["params"].to(device)

        x_Aj = torch.cat(
            [batch["Aj_in"].to(device), p.unsqueeze(1).repeat(1, K, 1)],
            dim=2
        )
        y_Aj = batch["Aj_out"].to(device)

        x_Aj1 = torch.cat(
            [batch["Aj1_in"].to(device), p.unsqueeze(1).repeat(1, K, 1)],
            dim=2
        )
        y_Aj1 = batch["Aj1_out"].to(device)

        x_Aj2 = torch.cat(
            [batch["Aj2_in"].to(device), p.unsqueeze(1).repeat(1, K, 1)],
            dim=2
        )
        y_Aj2 = batch["Aj2_out"].to(device)

        pred_Aj = model_Aj(x_Aj)
        pred_Aj1 = model_Aj1(x_Aj1)
        pred_Aj2 = model_Aj2(x_Aj2)

        mse_Aj = ((pred_Aj - y_Aj) ** 2).mean().item()
        mse_Aj1 = ((pred_Aj1 - y_Aj1) ** 2).mean().item()
        mse_Aj2 = ((pred_Aj2 - y_Aj2) ** 2).mean().item()

        mse_Aj_total += mse_Aj
        mse_Aj1_total += mse_Aj1
        mse_Aj2_total += mse_Aj2
        n_samples += 1

print("Test MSE Aj :", mse_Aj_total / n_samples)
print("Test MSE Aj1:", mse_Aj1_total / n_samples)
print("Test MSE Aj2:", mse_Aj2_total / n_samples)

# optional: one overall score for easier comparison
overall_test_mse = (mse_Aj_total + mse_Aj1_total + mse_Aj2_total) / (3 * n_samples)
print("Overall Test MSE:", overall_test_mse)

# -------------------------------------------------
# Trajectory reconstruction from overlapping windows
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
        start = w * stride
        end = start + T_out
        pred_traj[start:end] += y_pred[w]
        count[start:end] += 1

    count[count == 0] = 1
    pred_traj = pred_traj / count

    return pred_traj

# -------------------------------------------------
# Reconstruct predictions for one test run
# -------------------------------------------------
pred_Aj  = reconstruct(plot_loader, model_Aj, "Aj")
pred_Aj1 = reconstruct(plot_loader, model_Aj1, "Aj1")
pred_Aj2 = reconstruct(plot_loader, model_Aj2, "Aj2")

# -------------------------------------------------
# Load and normalize original trajectory for one run
# -------------------------------------------------
run_id = run_ids[train_runs + val_runs + M]
run_df = states_df[states_df["run_id"] == run_id].sort_values("t")

original = run_df[["S", "I", "R"]].values
original = original[K:]
original = (original - dataset.state_mean) / dataset.state_std

#T_plot = len(pred_Aj)
#original = original[:T_plot]

T_plot = min(200, len(pred_Aj))

original = original[:T_plot]
pred_Aj  = pred_Aj[:T_plot]
pred_Aj1 = pred_Aj1[:T_plot]
pred_Aj2 = pred_Aj2[:T_plot]

t = range(T_plot)

# -------------------------------------------------
# Plot results
# -------------------------------------------------

"""






fig, axs = plt.subplots(3, 1, figsize=(10, 10))

titles = [
    "Aj prediction vs Original",
    "Aj1 prediction vs Original",
    "Aj2 prediction vs Original"
]

preds = [pred_Aj, pred_Aj1, pred_Aj2]

for i in range(3):
    axs[i].grid()
    axs[i].plot(t, original[:, 0], label="True S")
    axs[i].plot(t, preds[i][:, 0], "--", label="Pred S")

    axs[i].plot(t, original[:, 1], label="True I")
    axs[i].plot(t, preds[i][:, 1], "--", label="Pred I")

    axs[i].plot(t, original[:, 2], label="True R")
    axs[i].plot(t, preds[i][:, 2], "--", label="Pred R")

    axs[i].set_title(titles[i])
    #axs[i].set_xlabel("time")
    axs[i].set_ylabel("state")
    axs[i].legend(loc="center left", bbox_to_anchor=(1, 0.5))

axs[-1].set_xlabel("Time")
plt.tight_layout()
plt.savefig("outputs/figures/states.png", dpi=300, bbox_inches="tight")
plt.show()




"""




titles = [
    "Aj prediction vs Original",
    "Aj1 prediction vs Original",
    "Aj2 prediction vs Original"
]

preds = [pred_Aj, pred_Aj1, pred_Aj2]

for i in range(3):
    plt.figure(figsize=(10, 6))

    plt.grid()

    plt.plot(t, original[:, 0], label="True S")
    plt.plot(t, preds[i][:, 0], "--", label="Pred S")

    plt.plot(t, original[:, 1], label="True I")
    plt.plot(t, preds[i][:, 1], "--", label="Pred I")

    plt.plot(t, original[:, 2], label="True R")
    plt.plot(t, preds[i][:, 2], "--", label="Pred R")

    plt.title(titles[i])
    plt.xlabel("Time")
    plt.ylabel("State")

    plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    plt.tight_layout()

    # هر کدوم جدا ذخیره بشه
    plt.savefig(f"outputs/figures/{titles[i].replace(' ', '_')}.png",
                dpi=300, bbox_inches="tight")

plt.show()



