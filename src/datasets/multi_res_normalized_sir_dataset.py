import numpy as np
import pandas as pd
import pywt
import torch
from torch.utils.data import Dataset


class MultiResSIRDataset(Dataset):
    """
    Sliding-window SWT-based dataset using additive MRA reconstruction.

    Each sample:
        x_in  : (K, 6)  -> 3 resolution channels + 3 broadcast params_norm
        x_out : (T, 3)  -> future resolution only (states only)

    Normalization:
        - states: divide by population N (per run)
        - params: z-score using train-only stats (loaded from file or passed)
    """

    def __init__(
        self,
        params_path: str,
        states_path: str,
        resolution: str = "Aj",   # "Aj", "Aj1", "Aj2"
        wavelet: str = "db4",
        level: int = 3,
        j: int = 2,
        K: int = 16,
        T_out: int = 32,
        stride: int = 4,
        norm_stats_path: str | None = None,
        param_mean: np.ndarray | None = None,
        param_std: np.ndarray | None = None,
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

        # ------------------------------
        # Load normalization stats (train-only)
        # ------------------------------
        if norm_stats_path is not None:
            stats = torch.load(norm_stats_path, map_location="cpu")
            self.param_mean = np.asarray(stats["param_mean"], dtype=np.float32)
            self.param_std = np.asarray(stats["param_std"], dtype=np.float32)
        else:
            if param_mean is None or param_std is None:
                raise ValueError("Provide norm_stats_path OR (param_mean and param_std).")
            self.param_mean = np.asarray(param_mean, dtype=np.float32)
            self.param_std = np.asarray(param_std, dtype=np.float32)

        self.param_std = self.param_std + 1e-8  # avoid divide-by-zero

        # ------------------------------
        # Load CSVs
        # ------------------------------
        params_df = pd.read_csv(params_path)
        states_df = pd.read_csv(states_path)
        states_df = states_df.sort_values(["run_id", "t"])

        # ------------------------------
        # Loop over runs (ordered)
        # ------------------------------
        for run_id, run_df in states_df.groupby("run_id"):

            run_df = run_df.sort_values("t")
            states = run_df[["S", "I", "R"]].to_numpy(dtype=np.float32)
            L = len(states)

            # ---- normalize states by population N (per run) ----
            N = float(states[0].sum())
            if N <= 0:
                raise ValueError(f"Non-positive population for run_id={run_id}")
            #N = 1
            states = states / N

            # ---- load and normalize params (train-only stats) ----
            row = params_df[params_df["run_id"] == run_id].iloc[0]
            params = np.array([
                row["transmissionProbability"],
                row["meanTimeToRecover"],
                row["meanImmunityDuration"]
            ], dtype=np.float32)

            params_norm = (params - self.param_mean) / self.param_std

            # ---- windows ----
            n_windows = (L - K - T_out) // stride + 1
            for w in range(n_windows):
                start = w * stride
                in_start = start
                in_end = start + K
                out_end = in_end + T_out

                past_window = states[in_start:in_end]      # (K,3)
                future_window = states[in_end:out_end]     # (T,3)

                Aj_past, Aj1_past, Aj2_past = self._compute_resolutions(past_window)
                Aj_fut, Aj1_fut, Aj2_fut = self._compute_resolutions(future_window)

                if resolution == "Aj":
                    res_past, res_fut = Aj_past, Aj_fut
                elif resolution == "Aj1":
                    res_past, res_fut = Aj1_past, Aj1_fut
                else:
                    res_past, res_fut = Aj2_past, Aj2_fut

                # ---- broadcast normalized params to length K ----
                params_expanded = np.tile(params_norm, (K, 1))      # (K,3)
                x_input = np.concatenate([res_past, params_expanded], axis=1)  # (K,6)

                self.samples.append({
                    "x_in": torch.tensor(x_input, dtype=torch.float32),
                    "x_out": torch.tensor(res_fut, dtype=torch.float32)
                })

    # -------------------------------------------------
    # SWT Multi-resolution reconstruction using MRA
    # -------------------------------------------------
    def _compute_resolutions(self, window: np.ndarray):
        Aj  = np.zeros_like(window)
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
            # comps = [A_L, D_L, D_{L-1}, ..., D_1]
            A_L = comps[0]

            # D_list = [D1, D2, ..., DL]
            D_list = [None] * self.level
            for ell in range(1, self.level + 1):
                D_list[ell - 1] = comps[self.level - ell + 1]

            Dj  = D_list[self.j - 1]   # D_j
            Dj1 = D_list[self.j - 2]   # D_{j-1}

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

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]