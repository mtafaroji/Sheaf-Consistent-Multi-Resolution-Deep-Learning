import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import pandas as pd
import matplotlib.pyplot as plt

from src.datasets.multi_res_normalized_sir_dataset import MultiResSIRDataset
from src.models.multi_res_forecast_net import ResolutionForecastNet


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# --------------------------------------------------
# Hyperparameters (باید همان train باشد)
# --------------------------------------------------

K = 32
T_out = 16
stride = 8

norm_stats_path = "outputs/normalization_stats.pt"

# --------------------------------------------------
# Dataset
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
# استخراج run ها
# --------------------------------------------------

states_df = pd.read_csv("data/raw/states.csv")

run_ids = sorted(states_df["run_id"].unique())
runs = len(run_ids)

first_run = states_df[states_df["run_id"] == run_ids[0]]
time_length = len(first_run)

windows_per_run = (time_length - K - T_out) // stride + 1

print("Runs:", runs)
print("Windows per run:", windows_per_run)

# --------------------------------------------------
# همان split که در train استفاده شد
# --------------------------------------------------

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

# --------------------------------------------------
# DataLoaders
# --------------------------------------------------

test_loader_Aj  = DataLoader(test_Aj,  batch_size=windows_per_run, shuffle=False)
test_loader_Aj1 = DataLoader(test_Aj1, batch_size=windows_per_run, shuffle=False)
test_loader_Aj2 = DataLoader(test_Aj2, batch_size=windows_per_run, shuffle=False)

# --------------------------------------------------
# Models
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

criterion = nn.MSELoss()

# --------------------------------------------------
# Evaluate
# --------------------------------------------------

def evaluate(model, loader):

    total_loss = 0

    with torch.no_grad():

        for batch in loader:

            x = batch["x_in"].to(device)
            y = batch["x_out"].to(device)

            pred = model(x)

            loss = criterion(pred, y)

            total_loss += loss.item()

    return total_loss / len(loader)


test_Aj_loss  = evaluate(model_Aj, test_loader_Aj)
test_Aj1_loss = evaluate(model_Aj1, test_loader_Aj1)
test_Aj2_loss = evaluate(model_Aj2, test_loader_Aj2)

print("\n===== TEST RESULTS =====")

print("Aj  Loss :", test_Aj_loss)
print("Aj1 Loss :", test_Aj1_loss)
print("Aj2 Loss :", test_Aj2_loss)

# --------------------------------------------------
# Plot one window from Aj2
# --------------------------------------------------

batch = next(iter(test_loader_Aj2))

x = batch["x_in"].to(device)
y_true = batch["x_out"].to(device)

with torch.no_grad():

    y_pred = model_Aj2(x)

# انتخاب یک پنجره
window = 0

true_traj = y_true[window].cpu().numpy()
pred_traj = y_pred[window].cpu().numpy()

t = range(T_out)

plt.figure(figsize=(10,5))

plt.plot(t, true_traj[:,0], label="True S")
plt.plot(t, pred_traj[:,0], "--", label="Pred S")

plt.plot(t, true_traj[:,1], label="True I")
plt.plot(t, pred_traj[:,1], "--", label="Pred I")

plt.plot(t, true_traj[:,2], label="True R")
plt.plot(t, pred_traj[:,2], "--", label="Pred R")

plt.title("Aj2 prediction vs ground truth")

plt.legend()

plt.xlabel("Future time")

plt.ylabel("Normalized state")

plt.show()