import torch
import torch.nn as nn
import torch.nn.functional as F


class ResolutionForecastNet(nn.Module):
    """
    Causal Temporal Conv + MLP Decoder (Residual)

    Input:
        x : (batch, K, 6)
            3 resolution channels + 3 params (broadcast)

    Output:
        future : (batch, T_out, 3)
    """

    def __init__(
        self,
        K,
        T_out,
        hidden_channels=8,
        kernel_size=3,
        stride=1,
        dilation=1,
        mlp_hidden=128,
    ):
        super().__init__()

        self.K = K
        self.T_out = T_out

        in_channels = 6

        padding = (kernel_size - 1) * dilation

        # ---- Conv layers ----
        self.conv1 = nn.Conv1d(
            in_channels, hidden_channels,
            kernel_size=kernel_size,
            stride=stride,
            dilation=dilation,
            padding=padding
        )

        self.conv2 = nn.Conv1d(
            hidden_channels, hidden_channels,
            kernel_size=kernel_size,
            stride=1,
            dilation=dilation,
            padding=padding
        )

        self.conv3 = nn.Conv1d(
            hidden_channels, hidden_channels,
            kernel_size=kernel_size,
            stride=1,
            dilation=dilation,
            padding=padding
        )

        # ---- MLP Decoder ----
        self.mlp = nn.Sequential(
            nn.Linear(hidden_channels, mlp_hidden),
            nn.GELU(),
            nn.Linear(mlp_hidden, mlp_hidden),
            nn.GELU(),
            nn.Linear(mlp_hidden, T_out * 3)
        )

    def forward(self, x):
        """
        x: (B, K, 6)
        """

        B, K, _ = x.shape

        # ---- Conv1D expects (B, C, T)
        x = x.permute(0, 2, 1)

        # ---- Causal Conv ----
        h = F.gelu(self.conv1(x))
        h = h[:, :, :K]

        h = F.gelu(self.conv2(h))
        h = h[:, :, :K]

        h = F.gelu(self.conv3(h))
        h = h[:, :, :K]

        # ---- Global pooling
        h = torch.mean(h, dim=-1)   # (B, hidden_channels)

        # ---- Decode
        delta = self.mlp(h)         # (B, T_out*3)
        delta = delta.view(B, self.T_out, 3)

        # ---- Residual prediction
        last_state = x[:, :3, -1]   # فقط S,I,R
        last_state = last_state.unsqueeze(1)  # (B,1,3)

        future = last_state + delta

        return future