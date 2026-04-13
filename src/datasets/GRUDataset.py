import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np


class SimpleSIRDataset(Dataset):
    def __init__(self, run_ids, norm_stats_path, K=8, T_out=16, stride=8):

        states_path = "data/raw/states.csv"
        params_path = "data/raw/params.csv"

        states = pd.read_csv(states_path)
        params = pd.read_csv(params_path)

        states = states.sort_values(["run_id", "t"])

        # -----------------------------
        # load normalization stats
        # -----------------------------
        stats = torch.load(norm_stats_path, map_location="cpu")

        self.param_mean = stats["param_mean"]
        self.param_std  = stats["param_std"] + 1e-8

        self.state_mean = stats["state_mean"]
        self.state_std  = stats["state_std"] + 1e-8

        self.samples = []

        for run_id in run_ids:

            traj = states[states["run_id"] == run_id][["S","I","R"]].values.astype(np.float32)

            # ✅ SAME normalization as your main model
            traj = (traj - self.state_mean) / self.state_std

            p = params[params["run_id"] == run_id][
                ["transmissionProbability","meanTimeToRecover","meanImmunityDuration"]
            ].values[0].astype(np.float32)

            # ✅ SAME normalization
            p = (p - self.param_mean) / self.param_std

            T = len(traj)

            for i in range(0, T - K - T_out + 1, stride):

                x = traj[i:i+K]
                y = traj[i+K:i+K+T_out]

                self.samples.append((x, y, p))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):

        x, y, p = self.samples[idx]

        return (
            torch.tensor(x, dtype=torch.float32),
            torch.tensor(y, dtype=torch.float32),
            torch.tensor(p, dtype=torch.float32),
        )