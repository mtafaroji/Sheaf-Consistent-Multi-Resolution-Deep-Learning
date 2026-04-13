import torch
import torch.nn as nn
import torch.nn.functional as F
import pywt


class WaveletRestriction(nn.Module):

    def __init__(self, wavelet="db4"):

        super().__init__()

        w = pywt.Wavelet(wavelet)

        hp = torch.tensor(w.dec_hi[::-1], dtype=torch.float32)

        hp1 = hp
        hp2 = self.dilate_filter(hp, 2)

        self.register_buffer("hp1", hp1.view(1,1,-1))
        self.register_buffer("hp2", hp2.view(1,1,-1))

    def dilate_filter(self, h, level):

        step = 2 ** (level - 1)

        h_dilated = []

        for i in range(len(h)):

            h_dilated.append(h[i])

            if i < len(h) - 1:
                h_dilated.extend([0]*(step-1))

        return torch.tensor(h_dilated, dtype=torch.float32)

    def remove_D1(self, signal):

        x = signal.T.unsqueeze(0)

        D = F.conv1d(
            x,
            self.hp1.repeat(3,1,1),
            padding=self.hp1.shape[-1]//2,
            groups=3
        )

        D = D.squeeze(0).T
        
        # fix lenght mismatch due to padding
        D = D[:signal.shape[0]]
        coarse = signal - D

        return coarse

    def remove_D2(self, signal):

        x = signal.T.unsqueeze(0)

        D = F.conv1d(
            x,
            self.hp2.repeat(3,1,1),
            padding=self.hp2.shape[-1]//2,
            groups=3
        )

        D = D.squeeze(0).T

        coarse = signal - D

        return coarse