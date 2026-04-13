import numpy as np
import pandas as pd
import pywt
import torch
from torch.utils.data import Dataset

class MultiResSIRDataset(Dataset):

    """
    Efficient SWT multi-resolution dataset

    Each sample contains ALL resolutions:

        Aj_in   : (K,3)
        Aj1_in  : (K,3)
        Aj2_in  : (K,3)

        Aj_out  : (T,3)
        Aj1_out : (T,3)
        Aj2_out : (T,3)

        params  : (3,)
    """

    def __init__(
        self,
        params_path,
        states_path,
        wavelet="db4",
        level=3,
        j=2,
        K=16,
        T_out=32,
        stride=4,
        norm_stats_path=None,
        param_mean=None,
        param_std=None,
        state_mean=None,
        state_std=None,
        max_runs=None
    ):

        self.samples = []
        self.wavelet = wavelet
        self.level = level
        self.j = j
        self.K = K
        self.T_out = T_out
        self.stride = stride
        self.max_runs = max_runs

        # -----------------------------
        # Load normalization stats
        # -----------------------------
        if norm_stats_path is not None:
            stats = torch.load(norm_stats_path, map_location="cpu")

            self.param_mean = np.asarray(stats["param_mean"], dtype=np.float32)
            self.param_std = np.asarray(stats["param_std"], dtype=np.float32)

            self.state_mean = np.asarray(stats["state_mean"], dtype=np.float32)
            self.state_std = np.asarray(stats["state_std"], dtype=np.float32)

        else:
            self.param_mean = np.asarray(param_mean, dtype=np.float32)
            self.param_std = np.asarray(param_std, dtype=np.float32)

            self.state_mean = np.asarray(state_mean, dtype=np.float32)
            self.state_std = np.asarray(state_std, dtype=np.float32)

        self.param_std = self.param_std + 1e-8
        self.state_std = self.state_std + 1e-8

        # -----------------------------
        # Load CSV
        # -----------------------------
        params_df = pd.read_csv(params_path).set_index("run_id")
        states_df = pd.read_csv(states_path).sort_values(["run_id", "t"])

        if self.max_runs is not None:
            selected_runs = states_df["run_id"].unique()[:self.max_runs]
            states_df = states_df[states_df["run_id"].isin(selected_runs)]

        # -----------------------------
        # Loop over runs
        # -----------------------------
        for run_id, run_df in states_df.groupby("run_id"):

            states = run_df[["S", "I", "R"]].to_numpy(dtype=np.float32)

            # ---- normalize states ----
            states_norm = (states - self.state_mean) / self.state_std

            L = len(states_norm)

            # ---- params ----
            row = params_df.loc[run_id]

            params = np.array([
                row["transmissionProbability"],
                row["meanTimeToRecover"],
                row["meanImmunityDuration"]
            ], dtype=np.float32)

            params_norm = (params - self.param_mean) / self.param_std

            # -----------------------------
            # windows
            # -----------------------------
            n_windows = (L - K - T_out) // stride + 1

            for w in range(n_windows):

                start = w * stride
                in_start = start
                in_end = start + K
                out_end = in_end + T_out

                past_window = states_norm[in_start:in_end]
                future_window = states_norm[in_end:out_end]

                Aj_past, Aj1_past, Aj2_past = self._compute_resolutions(past_window)
                Aj_fut, Aj1_fut, Aj2_fut = self._compute_resolutions(future_window)

                self.samples.append({
                    "Aj_in": torch.tensor(Aj_past, dtype=torch.float32),
                    "Aj1_in": torch.tensor(Aj1_past, dtype=torch.float32),
                    "Aj2_in": torch.tensor(Aj2_past, dtype=torch.float32),

                    "Aj_out": torch.tensor(Aj_fut, dtype=torch.float32),
                    "Aj1_out": torch.tensor(Aj1_fut, dtype=torch.float32),
                    "Aj2_out": torch.tensor(Aj2_fut, dtype=torch.float32),

                    "params": torch.tensor(params_norm, dtype=torch.float32)
                })

    # -------------------------------------------------
    # compute resolutions
    # -------------------------------------------------
    def _compute_resolutions(self, window):

        Aj = np.zeros_like(window)
        Aj1 = np.zeros_like(window)
        Aj2 = np.zeros_like(window)

        for dim in range(3):

            signal = window[:, dim]

            comps = pywt.mra(
                signal,
                self.wavelet,
                level=self.level,
                transform="swt"
            )

            A_L = comps[0]

            D_list = [None] * self.level

            for ell in range(1, self.level + 1):
                D_list[ell - 1] = comps[self.level - ell + 1]

            Dj = D_list[self.j - 1]
            Dj1 = D_list[self.j - 2]

            Aj_dim = A_L.copy()

            for idx in range(self.j, self.level):
                Aj_dim += D_list[idx]

            Aj1_dim = Aj_dim + Dj
            Aj2_dim = Aj1_dim + Dj1

            Aj[:, dim] = Aj_dim
            Aj1[:, dim] = Aj1_dim
            Aj2[:, dim] = Aj2_dim

        return Aj, Aj1, Aj2

    # -------------------------------------------------
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]