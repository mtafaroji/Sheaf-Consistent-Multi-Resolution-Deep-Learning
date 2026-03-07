import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import pandas as pd

from src.datasets.multi_res_sir_dataset import MultiResSIRDataset
from src.models.multi_res_forecast_net import ResolutionForecastNet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --------------------------------------------------
# Hyperparameters
# --------------------------------------------------

K = 32
T_out = 16
stride = 8
epochs = 2
lr = 1e-3

# --------------------------------------------------
# Dataset (سه نسخه برای سه رزولوشن)
# --------------------------------------------------

dataset_Aj = MultiResSIRDataset(
    "data/raw/params.csv",
    "data/raw/states.csv",
    resolution="Aj",
    K=K, T_out=T_out, stride=stride
)

dataset_Aj1 = MultiResSIRDataset(
    "data/raw/params.csv",
    "data/raw/states.csv",
    resolution="Aj1",
    K=K, T_out=T_out, stride=stride
)

dataset_Aj2 = MultiResSIRDataset(
    "data/raw/params.csv",
    "data/raw/states.csv",
    resolution="Aj2",
    K=K, T_out=T_out, stride=stride
)

# --------------------------------------------------
# استخراج تعداد ران و طول هر ران
# --------------------------------------------------

states_df = pd.read_csv("data/raw/states.csv")
run_ids = states_df["run_id"].unique()
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
# DataLoaders (هر batch = یک run)
# --------------------------------------------------

train_loader_Aj  = DataLoader(train_Aj,  batch_size=windows_per_run, shuffle=False)
train_loader_Aj1 = DataLoader(train_Aj1, batch_size=windows_per_run, shuffle=False)
train_loader_Aj2 = DataLoader(train_Aj2, batch_size=windows_per_run, shuffle=False)

val_loader_Aj  = DataLoader(val_Aj,  batch_size=windows_per_run, shuffle=False)
val_loader_Aj1 = DataLoader(val_Aj1, batch_size=windows_per_run, shuffle=False)
val_loader_Aj2 = DataLoader(val_Aj2, batch_size=windows_per_run, shuffle=False)


# --------------------------------------------------
# Shape check
# --------------------------------------------------

batch = next(iter(train_loader_Aj))

print("x_in shape :", batch["x_in"].shape)
print("x_out shape:", batch["x_out"].shape)

# --------------------------------------------------
# Models
# --------------------------------------------------


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
# Training Loop
# --------------------------------------------------

for epoch in range(epochs):

    model_Aj.train()
    model_Aj1.train()
    model_Aj2.train()

    total_loss_Aj = 0
    total_loss_Aj1 = 0
    total_loss_Aj2 = 0

    for (batch_Aj, batch_Aj1, batch_Aj2) in zip(
        train_loader_Aj, train_loader_Aj1, train_loader_Aj2
    ):

        x_Aj  = batch_Aj["x_in"].to(device)
        y_Aj  = batch_Aj["x_out"].to(device)

        x_Aj1 = batch_Aj1["x_in"].to(device)
        y_Aj1 = batch_Aj1["x_out"].to(device)

        x_Aj2 = batch_Aj2["x_in"].to(device)
        y_Aj2 = batch_Aj2["x_out"].to(device)

        # ---- Aj ----
        opt_Aj.zero_grad()
        pred_Aj = model_Aj(x_Aj)
        loss_Aj = criterion(pred_Aj, y_Aj)
        loss_Aj.backward()
        opt_Aj.step()

        # ---- Aj1 ----
        opt_Aj1.zero_grad()
        pred_Aj1 = model_Aj1(x_Aj1)
        loss_Aj1 = criterion(pred_Aj1, y_Aj1)
        loss_Aj1.backward()
        opt_Aj1.step()

        # ---- Aj2 ----
        opt_Aj2.zero_grad()
        pred_Aj2 = model_Aj2(x_Aj2)
        loss_Aj2 = criterion(pred_Aj2, y_Aj2)
        loss_Aj2.backward()
        opt_Aj2.step()

        total_loss_Aj  += loss_Aj.item()
        total_loss_Aj1 += loss_Aj1.item()
        total_loss_Aj2 += loss_Aj2.item()

    print(f"Epoch {epoch+1}")
    print("Aj  Loss:", total_loss_Aj / len(train_loader_Aj))
    print("Aj1 Loss:", total_loss_Aj1 / len(train_loader_Aj1))
    print("Aj2 Loss:", total_loss_Aj2 / len(train_loader_Aj2))

# --------------------------------------------------
# Test Evaluation
# --------------------------------------------------

model_Aj.eval()
model_Aj1.eval()
model_Aj2.eval()

with torch.no_grad():
    test_loss_Aj = 0
    for batch in DataLoader(test_Aj, batch_size=windows_per_run):
        x = batch["x_in"].to(device)
        y = batch["x_out"].to(device)
        pred = model_Aj(x)
        test_loss_Aj += criterion(pred, y).item()

print("Final Test Loss Aj:", test_loss_Aj)