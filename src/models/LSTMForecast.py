import torch
import torch.nn as nn


class LSTMForecast(nn.Module):
    def __init__(self, input_dim=6, hidden_dim=64, num_layers=2, T_out=16):
        super().__init__()

        self.T_out = T_out

        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )

        self.fc = nn.Linear(hidden_dim, T_out * 3)

    def forward(self, x, params):

        B, K, _ = x.shape

        params_expanded = params.unsqueeze(1).repeat(1, K, 1)

        x = torch.cat([x, params_expanded], dim=-1)

        out, _ = self.lstm(x)

        h_last = out[:, -1, :]

        y = self.fc(h_last)

        y = y.view(B, self.T_out, 3)

        # residual
        last_state = x[:, -1, :3].unsqueeze(1)
        y = last_state + y

        return y