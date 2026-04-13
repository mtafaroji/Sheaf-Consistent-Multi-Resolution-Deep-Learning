import torch
import torch.nn as nn


class TCNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1):
        super().__init__()

        padding = (kernel_size - 1) * dilation

        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=padding,
            dilation=dilation
        )

        self.relu = nn.ReLU()

        self.downsample = None
        if in_channels != out_channels:
            self.downsample = nn.Conv1d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        # x: (B, C, K)
        out = self.conv(x)

        # causal crop
        crop = self.conv.padding[0]
        if crop > 0:
            out = out[:, :, :-crop]

        residual = x if self.downsample is None else self.downsample(x)

        # چون ممکن است طول‌ها دقیقاً یکی نباشند
        if residual.size(-1) > out.size(-1):
            residual = residual[:, :, -out.size(-1):]
        elif out.size(-1) > residual.size(-1):
            out = out[:, :, -residual.size(-1):]

        out = out + residual
        out = self.relu(out)
        return out


class TCNForecast(nn.Module):
    def __init__(self, input_dim=6, hidden_dim=64, T_out=64):
        super().__init__()

        self.T_out = T_out

        self.network = nn.Sequential(
            TCNBlock(input_dim, hidden_dim, kernel_size=3, dilation=1),
            TCNBlock(hidden_dim, hidden_dim, kernel_size=3, dilation=2),
            TCNBlock(hidden_dim, hidden_dim, kernel_size=3, dilation=4),
            TCNBlock(hidden_dim, hidden_dim, kernel_size=3, dilation=8),
        )

        self.fc = nn.Linear(hidden_dim, T_out * 3)

    def forward(self, x, params):
        # x: (B, K, 3)
        # params: (B, 3)

        B, K, _ = x.shape

        # params -> (B, K, 3)
        params_expanded = params.unsqueeze(1).repeat(1, K, 1)

        # (B, K, 6)
        x_cat = torch.cat([x, params_expanded], dim=-1)

        # Conv1d wants (B, C, K)
        x_cat = x_cat.permute(0, 2, 1)

        out = self.network(x_cat)

        # last time feature
        h_last = out[:, :, -1]   # (B, hidden_dim)

        y = self.fc(h_last)      # (B, T_out*3)
        y = y.view(B, self.T_out, 3)

        # residual: add last observed state
        last_state = x[:, -1, :].unsqueeze(1)   # (B,1,3)
        y = last_state + y

        return y