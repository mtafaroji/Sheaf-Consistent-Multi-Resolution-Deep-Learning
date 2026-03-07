import numpy as np
import pandas as pd
import pywt
import torch
from torch.utils.data import Dataset


class MultiResSIRDataset(Dataset):
    """
    Sliding-window SWT-based dataset using additive MRA reconstruction.

    Each sample:
        x_in  : (K, 6)  -> 3 resolution channels + 3 broadcast params
        x_out : (T, 3)  -> future resolution only
    """

    def __init__(
        self,
        params_path: str,
        states_path: str,
        resolution: str = "Aj",   # "Aj", "Aj1", "Aj2"
        wavelet: str = "db4",
        level: int = 3,
        j: int = 2,
        K: int = 32,
        T_out: int = 16,
        stride: int = 8
    ):

        assert resolution in ["Aj", "Aj1", "Aj2"]
        assert 2 <= j < level

        self.samples = []
        self.wavelet = wavelet
        self.level = level
        self.j = j
        self.K = K
        self.T_out = T_out
        self.stride = stride
        self.resolution = resolution

        params_df = pd.read_csv(params_path)
        states_df = pd.read_csv(states_path)
        states_df = states_df.sort_values(["run_id", "t"])

        for run_id, run_df in states_df.groupby("run_id"):

            run_df = run_df.sort_values("t")
            states = run_df[["S", "I", "R"]].to_numpy(dtype=np.float32)
            L = len(states)

            row = params_df[params_df["run_id"] == run_id].iloc[0]
            params = np.array([
                row["transmissionProbability"],
                row["meanTimeToRecover"],
                row["meanImmunityDuration"]
            ], dtype=np.float32)

            n_windows = (L - K - T_out) // stride + 1

            for w in range(n_windows):

                start = w * stride
                in_start = start
                in_end = start + K
                out_end = in_end + T_out

                past_window = states[in_start:in_end]
                future_window = states[in_end:out_end]

                Aj_past, Aj1_past, Aj2_past = self._compute_resolutions(past_window)
                Aj_future, Aj1_future, Aj2_future = self._compute_resolutions(future_window)

                if resolution == "Aj":
                    res_past = Aj_past
                    res_future = Aj_future
                elif resolution == "Aj1":
                    res_past = Aj1_past
                    res_future = Aj1_future
                else:
                    res_past = Aj2_past
                    res_future = Aj2_future

                # Broadcast params
                params_expanded = np.tile(params, (K, 1))
                x_input = np.concatenate([res_past, params_expanded], axis=1)

                self.samples.append({
                    "x_in": torch.tensor(x_input, dtype=torch.float32),
                    "x_out": torch.tensor(res_future, dtype=torch.float32)
                })

    # -------------------------------------------------
    # SWT Multi-resolution reconstruction using MRA
    # -------------------------------------------------

    def _compute_resolutions(self, window):

        Aj  = np.zeros_like(window)
        Aj1 = np.zeros_like(window)
        Aj2 = np.zeros_like(window)

        for dim in range(3):

            signal = window[:, dim]

            # MRA additive decomposition
            # comps = [A_L, D_L, D_{L-1}, ..., D_1]
            comps = pywt.mra(
                signal,
                self.wavelet,
                level=self.level,
                transform="swt"
            )

            A_L = comps[0]

            # Build D_list = [D1, D2, ..., DL]
            D_list = [None] * self.level
            for ell in range(1, self.level + 1):
                D_list[ell - 1] = comps[self.level - ell + 1]

            # Dj and Dj1
            Dj  = D_list[self.j - 1]
            Dj1 = D_list[self.j - 2]

            # Aj = A_L + sum_{k=j+1..L} D_k
            Aj_dim = A_L.copy()
            for idx in range(self.j, self.level):
                Aj_dim += D_list[idx]

            Aj1_dim = Aj_dim + Dj
            Aj2_dim = Aj1_dim + Dj1

            Aj[:, dim]  = Aj_dim
            Aj1[:, dim] = Aj1_dim
            Aj2[:, dim] = Aj2_dim

        return Aj, Aj1, Aj2

    # -------------------------------------------------

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]