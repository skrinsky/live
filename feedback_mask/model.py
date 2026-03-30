"""
feedback_mask/model.py — GTCRN adapted for 48 kHz feedback suppression.

Architecture: GTCRN (ShuffleNetV2 + SFE + TRA + 2×DPGRNN, ~24 K params).
Causal, streaming-compatible. Outputs a complex ratio mask applied to the
input STFT, so the model learns "what to keep" rather than "what to remove."

Adapted from: https://github.com/Xiaobin-Rong/gtcrn
Key change vs. original: ERB(65, 64, nfft=512, high_lim=8000, fs=16000)
                       → ERB(45, 84, nfft=960, high_lim=24000, fs=48000)
Total ERB output width: 45+84 = 129 — identical to original, so all downstream
encoder/decoder/DPGRNN dimensions are unchanged. Only the ERB filterbank
Linear layers resize to handle 481 FFT bins instead of 257.
"""

import torch
import torch.nn as nn
import numpy as np
from einops import rearrange

SR     = 48000
N_FFT  = 960
HOP    = 480
N_FREQ = N_FFT // 2 + 1   # 481


class ERB(nn.Module):
    def __init__(self, erb_subband_1, erb_subband_2, nfft=960, high_lim=24000, fs=48000):
        super().__init__()
        erb_filters = self._make_filters(erb_subband_1, erb_subband_2, nfft, high_lim, fs)
        nfreqs = nfft // 2 + 1
        self.erb_subband_1 = erb_subband_1
        self.erb_fc  = nn.Linear(nfreqs - erb_subband_1, erb_subband_2, bias=False)
        self.ierb_fc = nn.Linear(erb_subband_2, nfreqs - erb_subband_1, bias=False)
        self.erb_fc.weight  = nn.Parameter(erb_filters,   requires_grad=False)
        self.ierb_fc.weight = nn.Parameter(erb_filters.T, requires_grad=False)

    @staticmethod
    def _hz2erb(f): return 21.4 * np.log10(0.00437 * f + 1)
    @staticmethod
    def _erb2hz(e): return (10 ** (e / 21.4) - 1) / 0.00437

    def _make_filters(self, erb_subband_1, erb_subband_2, nfft, high_lim, fs):
        low_lim   = erb_subband_1 / nfft * fs
        erb_pts   = np.linspace(self._hz2erb(low_lim), self._hz2erb(high_lim), erb_subband_2)
        bins      = np.round(self._erb2hz(erb_pts) / fs * nfft).astype(np.int32)
        nfreqs    = nfft // 2 + 1
        filters   = np.zeros([erb_subband_2, nfreqs], dtype=np.float32)

        filters[0, bins[0]:bins[1]] = (bins[1] - np.arange(bins[0], bins[1]) + 1e-12) / (bins[1] - bins[0] + 1e-12)
        for i in range(erb_subband_2 - 2):
            filters[i+1, bins[i]:bins[i+1]]   = (np.arange(bins[i], bins[i+1]) - bins[i] + 1e-12) / (bins[i+1] - bins[i] + 1e-12)
            filters[i+1, bins[i+1]:bins[i+2]] = (bins[i+2] - np.arange(bins[i+1], bins[i+2]) + 1e-12) / (bins[i+2] - bins[i+1] + 1e-12)
        filters[-1, bins[-2]:bins[-1]+1] = 1 - filters[-2, bins[-2]:bins[-1]+1]

        filters = filters[:, erb_subband_1:]
        return torch.from_numpy(np.abs(filters))

    def bm(self, x):
        """x: (B,C,T,F) → (B,C,T,erb_subband_1+erb_subband_2)"""
        return torch.cat([x[..., :self.erb_subband_1], self.erb_fc(x[..., self.erb_subband_1:])], dim=-1)

    def bs(self, x):
        """x: (B,C,T,erb_subband_1+erb_subband_2) → (B,C,T,F)"""
        return torch.cat([x[..., :self.erb_subband_1], self.ierb_fc(x[..., self.erb_subband_1:])], dim=-1)


class SFE(nn.Module):
    """Subband Feature Extraction — groups neighbouring frequency bins."""
    def __init__(self, kernel_size=3):
        super().__init__()
        self.ks     = kernel_size
        self.unfold = nn.Unfold(kernel_size=(1, kernel_size), padding=(0, (kernel_size-1)//2))

    def forward(self, x):
        """x: (B,C,T,F) → (B,C*ks,T,F)"""
        return self.unfold(x).reshape(x.shape[0], x.shape[1] * self.ks, x.shape[2], x.shape[3])


class TRA(nn.Module):
    """Temporal Recurrent Attention — per-channel time-varying gate."""
    def __init__(self, channels):
        super().__init__()
        self.gru = nn.GRU(channels, channels * 2, batch_first=True)
        self.fc  = nn.Linear(channels * 2, channels)
        self.act = nn.Sigmoid()

    def forward(self, x):
        """x: (B,C,T,F) → (B,C,T,F)"""
        zt = x.pow(2).mean(-1)                        # (B,C,T)
        at = self.gru(zt.transpose(1, 2))[0]          # (B,T,2C)
        at = self.act(self.fc(at)).transpose(1, 2)     # (B,C,T)
        return x * at[..., None]


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel, stride, padding, groups=1, deconv=False, last=False):
        super().__init__()
        cls       = nn.ConvTranspose2d if deconv else nn.Conv2d
        self.conv = cls(in_ch, out_ch, kernel, stride, padding, groups=groups)
        self.bn   = nn.BatchNorm2d(out_ch)
        self.act  = nn.Tanh() if last else nn.PReLU()

    def forward(self, x): return self.act(self.bn(self.conv(x)))


class GTConvBlock(nn.Module):
    """Group Temporal Conv with causal temporal padding and ShuffleNet channel split."""
    def __init__(self, in_ch, hidden_ch, kernel, stride, padding, dilation, deconv=False):
        super().__init__()
        cls            = nn.ConvTranspose2d if deconv else nn.Conv2d
        self.pad_size  = (kernel[0] - 1) * dilation[0]
        self.sfe       = SFE(kernel_size=3)
        self.pw1       = cls(in_ch // 2 * 3, hidden_ch, 1)
        self.bn1       = nn.BatchNorm2d(hidden_ch)
        self.act1      = nn.PReLU()
        self.dw        = cls(hidden_ch, hidden_ch, kernel, stride=stride,
                             padding=padding, dilation=dilation, groups=hidden_ch)
        self.bn2       = nn.BatchNorm2d(hidden_ch)
        self.act2      = nn.PReLU()
        self.pw2       = cls(hidden_ch, in_ch // 2, 1)
        self.bn3       = nn.BatchNorm2d(in_ch // 2)
        self.tra       = TRA(in_ch // 2)

    def forward(self, x):
        """x: (B,C,T,F)"""
        x1, x2 = x.chunk(2, dim=1)
        x1  = self.sfe(x1)
        h   = self.act1(self.bn1(self.pw1(x1)))
        h   = nn.functional.pad(h, [0, 0, self.pad_size, 0])
        h   = self.act2(self.bn2(self.dw(h)))
        h   = self.bn3(self.pw2(h))
        h   = self.tra(h)
        # channel shuffle
        out = torch.stack([h, x2], dim=1).transpose(1, 2).contiguous()
        return rearrange(out, 'b c g t f -> b (c g) t f')


class GRNN(nn.Module):
    """Grouped GRU — splits input/hidden in half for efficiency."""
    def __init__(self, input_size, hidden_size, bidirectional=False):
        super().__init__()
        self.hidden_size  = hidden_size
        self.bidirectional = bidirectional
        dirs = 2 if bidirectional else 1
        self.rnn1 = nn.GRU(input_size // 2, hidden_size // 2, batch_first=True, bidirectional=bidirectional)
        self.rnn2 = nn.GRU(input_size // 2, hidden_size // 2, batch_first=True, bidirectional=bidirectional)

    def forward(self, x, h=None):
        """x: (B, T, C)"""
        if h is None:
            dirs = 2 if self.bidirectional else 1
            h = x.new_zeros(dirs, x.shape[0], self.hidden_size)
        x1, x2 = x.chunk(2, dim=-1)
        h1, h2 = h.chunk(2, dim=-1)
        y1, h1 = self.rnn1(x1, h1.contiguous())
        y2, h2 = self.rnn2(x2, h2.contiguous())
        return torch.cat([y1, y2], dim=-1), torch.cat([h1, h2], dim=-1)


class DPGRNN(nn.Module):
    """Dual-Path Grouped RNN: intra (frequency, bidirectional) + inter (time, causal)."""
    def __init__(self, channels, width, hidden):
        super().__init__()
        self.width  = width
        self.hidden = hidden
        self.intra_rnn = GRNN(channels, hidden // 2, bidirectional=True)
        self.intra_fc  = nn.Linear(hidden, hidden)
        self.intra_ln  = nn.LayerNorm((width, hidden))
        self.inter_rnn = GRNN(channels, hidden, bidirectional=False)
        self.inter_fc  = nn.Linear(hidden, hidden)
        self.inter_ln  = nn.LayerNorm((width, hidden))

    def forward(self, x, inter_h=None):
        """x: (B,C,T,F), inter_h: optional GRU hidden state from previous call.
        Returns (output, inter_h) so callers can persist state across chunks."""
        # Intra: across frequency bins per frame (bidirectional — no state to carry)
        B, C, T, F = x.shape
        xi = x.permute(0, 2, 3, 1).reshape(B * T, F, C)           # (B*T, F, C)
        xi = self.intra_fc(self.intra_rnn(xi)[0])                   # (B*T, F, C)
        xi = self.intra_ln(xi.reshape(B, T, F, C)) + x.permute(0, 2, 3, 1)  # (B,T,F,C)

        # Inter: across time frames per frequency bin (causal — state carries over)
        xo = xi.permute(0, 2, 1, 3).reshape(B * F, T, C)           # (B*F, T, C)
        xo, inter_h = self.inter_rnn(xo, inter_h)                   # persist hidden state
        xo = self.inter_fc(xo)                                       # (B*F, T, C)
        xo = self.inter_ln(xo.reshape(B, F, T, C).permute(0, 2, 1, 3)) + xi  # (B,T,F,C)

        return xo.permute(0, 3, 1, 2), inter_h   # (B,C,T,F), hidden


class Mask(nn.Module):
    """Complex Ratio Mask: (mask_real, mask_imag) applied to (spec_real, spec_imag)."""
    def forward(self, mask, spec):
        """mask, spec: (B,2,T,F)"""
        r = spec[:, 0] * mask[:, 0] - spec[:, 1] * mask[:, 1]
        i = spec[:, 1] * mask[:, 0] + spec[:, 0] * mask[:, 1]
        return torch.stack([r, i], dim=1)


class FeedbackMaskNet(nn.Module):
    """
    GTCRN adapted for 48 kHz feedback suppression.

    Input:  mic STFT  (B, N_FREQ, T, 2)  — real and imaginary stacked
    Output: enhanced STFT (B, N_FREQ, T, 2)

    ERB compression: 45 linear bins + 84 ERB bands = 129 total.
    Encoder produces (B, 16, T, 33) → 2×DPGRNN → Decoder → mask (B, 2, T, 129)
    → ierb → mask applied to full 481-bin spectrum.
    """
    def __init__(self):
        super().__init__()
        # ERB: keep width=129 so encoder/decoder dims are unchanged from original GTCRN
        self.erb     = ERB(45, 84, nfft=N_FFT, high_lim=24000, fs=SR)
        self.sfe     = SFE(kernel_size=3)

        self.encoder = nn.ModuleList([
            ConvBlock(9,  16, (1,5), stride=(1,2), padding=(0,2)),
            ConvBlock(16, 16, (1,5), stride=(1,2), padding=(0,2), groups=2),
            GTConvBlock(16, 16, (3,3), (1,1), (0,1), (1,1)),
            GTConvBlock(16, 16, (3,3), (1,1), (0,1), (2,1)),
            GTConvBlock(16, 16, (3,3), (1,1), (0,1), (5,1)),
        ])
        self.dpgrnn1 = DPGRNN(16, 33, 16)
        self.dpgrnn2 = DPGRNN(16, 33, 16)
        self.decoder = nn.ModuleList([
            GTConvBlock(16, 16, (3,3), (1,1), (2*5,1), (5,1), deconv=True),
            GTConvBlock(16, 16, (3,3), (1,1), (2*2,1), (2,1), deconv=True),
            GTConvBlock(16, 16, (3,3), (1,1), (2*1,1), (1,1), deconv=True),
            ConvBlock(16, 16, (1,5), stride=(1,2), padding=(0,2), groups=2, deconv=True),
            ConvBlock(16,  2, (1,5), stride=(1,2), padding=(0,2),           deconv=True, last=True),
        ])
        self.mask = Mask()

    def forward(self, spec, h=None):
        """spec: (B, F, T, 2) → (enhanced: (B, F, T, 2), h: GRU state tuple)

        h is a (h1, h2) tuple of inter-GRU hidden states from the two DPGRNNs.
        Pass None to reset (training / first inference block).
        Pass the returned h back in on subsequent calls to maintain temporal context
        across chunk boundaries — used by live.py for streaming inference.
        """
        h1, h2 = (None, None) if h is None else h
        spec_ref = spec  # save for mask application

        r, i  = spec[..., 0].permute(0,2,1), spec[..., 1].permute(0,2,1)   # (B,T,F)
        mag   = (r**2 + i**2 + 1e-12).sqrt()
        feat  = torch.stack([mag, r, i], dim=1)   # (B,3,T,F)

        feat  = self.erb.bm(feat)   # (B,3,T,129)
        feat  = self.sfe(feat)      # (B,9,T,129)

        skips = []
        for layer in self.encoder:
            feat = layer(feat)
            skips.append(feat)

        feat, h1 = self.dpgrnn1(feat, h1)
        feat, h2 = self.dpgrnn2(feat, h2)

        for k, layer in enumerate(self.decoder):
            feat = layer(feat + skips[4 - k])

        mask = self.erb.bs(feat)    # (B,2,T,F)

        enh  = self.mask(mask, spec_ref.permute(0,3,2,1))   # (B,2,T,F)
        return enh.permute(0,3,2,1), (h1, h2)   # (B,F,T,2), hidden state


if __name__ == '__main__':
    model = FeedbackMaskNet().eval()
    n = sum(p.numel() for p in model.parameters())
    print(f'Parameters: {n:,}')
    x = torch.randn(1, N_FREQ, 100, 2)
    y, _ = model(x)
    print(f'Input {tuple(x.shape)} → Output {tuple(y.shape)}')
