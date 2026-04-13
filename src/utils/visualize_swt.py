import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.datasets.multi_res_normalized_sir_dataset import MultiResSIRDataset


# ------------------------
# CONFIG
# ------------------------
params_path = "data/raw/params.csv"
states_path = "data/raw/states.csv"

run_to_plot = 1
segment_start = 0
segment_length = 32




K = 32
T_out = 16
stride = 8
epochs = 100
lr = 1e-3

# ------------------------
# Load raw trajectory
# ------------------------
states_df = pd.read_csv(states_path)
states_df = states_df.sort_values(["run_id", "t"])

run_df = states_df[states_df["run_id"] == run_to_plot]
states = run_df[["S", "I", "R"]].to_numpy(dtype=np.float32)

segment = states[segment_start:segment_start+segment_length]


# ------------------------
# Use dataset class only for SWT computation
# ------------------------

norm_path = "outputs/normalization_stats.pt"
""""
dataset = MultiResSIRDataset(
    params_path=params_path,
    states_path=states_path,
    resolution="Aj",  # مهم نیست
    level=3,
    j=2,
    K=32,
    T_out=32
)

"""


dataset = MultiResSIRDataset(
    "data/raw/params.csv",
    "data/raw/states.csv",
    resolution="Aj2",
    K=K,
    T_out=T_out,
    stride=stride,
    norm_stats_path = norm_path
)

print("States shape:", states.shape)
print("Segment start:", segment_start)
print("Segment end:", segment_start + segment_length)
print("Segment shape:", segment.shape)


Aj, Aj1, Aj2 = dataset._compute_resolutions(segment)


# ------------------------
# Plot
# ------------------------
time = np.arange(segment_length)

fig, axes = plt.subplots(3, 1, figsize=(10, 8))

labels = ["S", "I", "R"]

for dim in range(3):

    axes[dim].plot(time, segment[:, dim] , label="Original", linewidth=2)
    axes[dim].plot(time, Aj[:, dim], label="Aj")
    axes[dim].plot(time, Aj1[:, dim], label="Aj+1")
    axes[dim].plot(time, Aj2[:, dim], label="Aj+2")

    axes[dim].set_title(labels[dim])
    axes[dim].legend()

plt.tight_layout()
plt.show()