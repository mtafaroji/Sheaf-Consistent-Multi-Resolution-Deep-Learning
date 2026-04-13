import torch
import torch.nn as nn
import torch.nn.functional as F


class ResolutionForecastNet(nn.Module):

    def __init__(
        self,
        K,
        T_out,
        hidden_channels=64,
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

        # ---- MLP ----
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

        # ---- to (B, C, T)
        x = x.permute(0, 2, 1)

        # =========================
        # Block 1 (no skip)
        # =========================
        h = self.conv1(x)
        h = h[:, :, :K]
        h = F.gelu(h)

        # =========================
        # Block 2 (ResNet block)
        # =========================
        res = h

        h = self.conv2(h)
        h = h[:, :, :K]
        h = F.gelu(h)

        h = self.conv3(h)
        h = h[:, :, :K]

        h = h + res
        h = F.gelu(h)

        # ---- Global pooling
        h = torch.mean(h, dim=-1)   # (B, 64)

        # ---- Decode
        delta = self.mlp(h)
        delta = delta.view(B, self.T_out, 3)

        # ---- Residual output
        last_state = x[:, :3, -1].unsqueeze(1)
        future = last_state + delta

        return future