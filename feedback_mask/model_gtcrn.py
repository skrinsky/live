"""
feedback_mask/model_gtcrn.py — GTCRN adapted for 48 kHz source-separation feedback suppression.

Architecture: GTCRN (ShuffleNetV2 + SFE + TRA + 2 DPGRNN), adapted from gtcrn/gtcrn.py.
~24 K parameters.

Changes from the original GTCRN (16 kHz / N_FFT=512):
  1. 48 kHz / N_FFT=512 / HOP=256 → 5.33 ms latency (De-Feedback claims 4.9 ms)
     N_FFT=512 at 48 kHz → same 257 bins as original GTCRN at 16 kHz, so the
     entire downstream architecture is byte-for-byte identical.
  2. ERB bank re-parameterised for 48 kHz / N_FFT=512:
       erb_subband_1=65  (linear bins 0-64, covering 0–6094 Hz)
       erb_subband_2=64  (ERB-compress bins 65-256 → 6094–16000 Hz into 64 bins)
       total width = 65+64 = 129  ← identical to original
     The downstream encoder/DPGRNN/decoder is completely unchanged because the
     ERB output width is the same.
  3. No causality changes needed: the inter-RNN (across time) is already
     unidirectional=False; the intra-RNN (across frequency within a single frame)
     being bidirectional does NOT touch future time-steps, so the model is causal.

Why GTCRN over the per-bin GRU (FeedbackMaskNet):
  - Complex Ratio Mask (CRM) recovers both magnitude and phase at each bin,
    enabling proper source separation rather than just amplitude attenuation.
  - Dual-path GRNN gives both cross-frequency (intra) and cross-time (inter)
    context — needed to distinguish "voice formant sustained for 200ms" from
    "ring sustained for seconds."
  - ~24 K vs ~6 K params — enough capacity to learn source separation, not just
    per-bin binary suppression.

HybridLoss48k (defined below):
  - 30× complex MSE (power-compressed 0.7, giving equal weight to quiet and loud
    bins rather than over-weighting loud bins)
  - 70× magnitude MSE (power-compressed 0.3)
  - SI-SNR (scale-invariant, time-domain, sequence-level perceptual quality)
  Combined dense per-bin STFT gradient + sequence-level perceptual gradient.
"""

import torch
import numpy as np
import torch.nn as nn
from einops import rearrange

SR     = 48000
N_FFT  = 512
HOP    = 256
N_FREQ = N_FFT // 2 + 1   # 257 bins — same as original GTCRN at 16 kHz/512FFT

# HOP=256 → 256/48000 = 5.33 ms latency (De-Feedback claims 4.9 ms).
# N_FFT=512 at 48 kHz → 93.75 Hz/bin (coarser than N_FFT=960/50 Hz, but fine
# for source separation — the ERB bank handles cross-bin context anyway).

# ERB compression parameters.  Chosen so ERB output width = 129 = original GTCRN,
# keeping the entire encoder/DPGRNN/decoder architecture unchanged.
_ERB_LIN   = 65       # linear bins (covers 0–6094 Hz at 48kHz/512FFT)
_ERB_COMP  = 64       # ERB bins for 6094–16000 Hz
_ERB_HIGH  = 16000    # Hz ceiling for ERB bank (16–24 kHz not needed for voice/feedback)


# ── Unchanged from gtcrn/gtcrn.py ─────────────────────────────────────────────

class ERB(nn.Module):
    def __init__(self, erb_subband_1, erb_subband_2, nfft=512, high_lim=8000, fs=16000):
        super().__init__()
        erb_filters = self.erb_filter_banks(erb_subband_1, erb_subband_2, nfft, high_lim, fs)
        nfreqs = nfft // 2 + 1
        self.erb_subband_1 = erb_subband_1
        self.erb_fc  = nn.Linear(nfreqs - erb_subband_1, erb_subband_2, bias=False)
        self.ierb_fc = nn.Linear(erb_subband_2, nfreqs - erb_subband_1, bias=False)
        self.erb_fc.weight  = nn.Parameter(erb_filters,   requires_grad=False)
        self.ierb_fc.weight = nn.Parameter(erb_filters.T, requires_grad=False)

    def hz2erb(self, freq_hz):
        return 21.4 * np.log10(0.00437 * freq_hz + 1)

    def erb2hz(self, erb_f):
        return (10 ** (erb_f / 21.4) - 1) / 0.00437

    def erb_filter_banks(self, erb_subband_1, erb_subband_2, nfft=512, high_lim=8000, fs=16000):
        low_lim   = erb_subband_1 / nfft * fs
        erb_low   = self.hz2erb(low_lim)
        erb_high  = self.hz2erb(high_lim)
        erb_points = np.linspace(erb_low, erb_high, erb_subband_2)
        bins = np.round(self.erb2hz(erb_points) / fs * nfft).astype(np.int32)
        erb_filters = np.zeros([erb_subband_2, nfft // 2 + 1], dtype=np.float32)

        erb_filters[0, bins[0]:bins[1]] = (
            (bins[1] - np.arange(bins[0], bins[1]) + 1e-12) / (bins[1] - bins[0] + 1e-12))
        for i in range(erb_subband_2 - 2):
            erb_filters[i+1, bins[i]:bins[i+1]] = (
                (np.arange(bins[i], bins[i+1]) - bins[i] + 1e-12) / (bins[i+1] - bins[i] + 1e-12))
            erb_filters[i+1, bins[i+1]:bins[i+2]] = (
                (bins[i+2] - np.arange(bins[i+1], bins[i+2]) + 1e-12) / (bins[i+2] - bins[i+1] + 1e-12))
        erb_filters[-1, bins[-2]:bins[-1]+1] = 1 - erb_filters[-2, bins[-2]:bins[-1]+1]

        erb_filters = erb_filters[:, erb_subband_1:]
        return torch.from_numpy(np.abs(erb_filters))

    def bm(self, x):
        """x: (B, C, T, F)"""
        x_low  = x[..., :self.erb_subband_1]
        x_high = self.erb_fc(x[..., self.erb_subband_1:])
        return torch.cat([x_low, x_high], dim=-1)

    def bs(self, x_erb):
        """x: (B, C, T, F_erb)"""
        x_erb_low  = x_erb[..., :self.erb_subband_1]
        x_erb_high = self.ierb_fc(x_erb[..., self.erb_subband_1:])
        return torch.cat([x_erb_low, x_erb_high], dim=-1)


class SFE(nn.Module):
    """Subband Feature Extraction"""
    def __init__(self, kernel_size=3, stride=1):
        super().__init__()
        self.kernel_size = kernel_size
        self.unfold = nn.Unfold(
            kernel_size=(1, kernel_size), stride=(1, stride),
            padding=(0, (kernel_size - 1) // 2))

    def forward(self, x):
        """x: (B, C, T, F)"""
        xs = self.unfold(x).reshape(
            x.shape[0], x.shape[1] * self.kernel_size, x.shape[2], x.shape[3])
        return xs


class TRA(nn.Module):
    """Temporal Recurrent Attention"""
    def __init__(self, channels):
        super().__init__()
        self.att_gru = nn.GRU(channels, channels * 2, 1, batch_first=True)
        self.att_fc  = nn.Linear(channels * 2, channels)
        self.att_act = nn.Sigmoid()

    def forward(self, x):
        """x: (B, C, T, F)"""
        zt = torch.mean(x.pow(2), dim=-1)          # (B, C, T)
        at = self.att_gru(zt.transpose(1, 2))[0]
        at = self.att_fc(at).transpose(1, 2)
        at = self.att_act(at)
        return x * at[..., None]                   # (B, C, T, 1) broadcast


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, stride, padding,
                 groups=1, use_deconv=False, is_last=False):
        super().__init__()
        conv = nn.ConvTranspose2d if use_deconv else nn.Conv2d
        self.conv = conv(in_ch, out_ch, kernel_size, stride, padding, groups=groups)
        self.bn   = nn.BatchNorm2d(out_ch)
        self.act  = nn.Tanh() if is_last else nn.PReLU()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class GTConvBlock(nn.Module):
    """Group Temporal Convolution with ShuffleNetV2 channel split"""
    def __init__(self, in_ch, hidden_ch, kernel_size, stride, padding, dilation,
                 use_deconv=False):
        super().__init__()
        self.pad_size = (kernel_size[0] - 1) * dilation[0]
        conv = nn.ConvTranspose2d if use_deconv else nn.Conv2d

        self.sfe         = SFE(kernel_size=3, stride=1)
        self.point_conv1 = conv(in_ch // 2 * 3, hidden_ch, 1)
        self.point_bn1   = nn.BatchNorm2d(hidden_ch)
        self.point_act   = nn.PReLU()
        self.depth_conv  = conv(hidden_ch, hidden_ch, kernel_size,
                                stride=stride, padding=padding,
                                dilation=dilation, groups=hidden_ch)
        self.depth_bn    = nn.BatchNorm2d(hidden_ch)
        self.depth_act   = nn.PReLU()
        self.point_conv2 = conv(hidden_ch, in_ch // 2, 1)
        self.point_bn2   = nn.BatchNorm2d(in_ch // 2)
        self.tra         = TRA(in_ch // 2)

    def shuffle(self, x1, x2):
        x = torch.stack([x1, x2], dim=1)
        x = x.transpose(1, 2).contiguous()
        x = rearrange(x, 'b c g t f -> b (c g) t f')
        return x

    def forward(self, x):
        x1, x2 = torch.chunk(x, 2, dim=1)
        x1 = self.sfe(x1)
        h1 = self.point_act(self.point_bn1(self.point_conv1(x1)))
        h1 = nn.functional.pad(h1, [0, 0, self.pad_size, 0])
        h1 = self.depth_act(self.depth_bn(self.depth_conv(h1)))
        h1 = self.point_bn2(self.point_conv2(h1))
        h1 = self.tra(h1)
        return self.shuffle(h1, x2)


class GRNN(nn.Module):
    """Grouped RNN (splits input/hidden in half for efficiency)"""
    def __init__(self, input_size, hidden_size, num_layers=1,
                 batch_first=True, bidirectional=False):
        super().__init__()
        self.hidden_size  = hidden_size
        self.num_layers   = num_layers
        self.bidirectional = bidirectional
        self.rnn1 = nn.GRU(input_size // 2, hidden_size // 2, num_layers,
                            batch_first=batch_first, bidirectional=bidirectional)
        self.rnn2 = nn.GRU(input_size // 2, hidden_size // 2, num_layers,
                            batch_first=batch_first, bidirectional=bidirectional)

    def forward(self, x, h=None):
        if h is None:
            dirs = 2 if self.bidirectional else 1
            h = torch.zeros(self.num_layers * dirs, x.shape[0],
                            self.hidden_size, device=x.device)
        x1, x2 = torch.chunk(x, 2, dim=-1)
        h1, h2 = torch.chunk(h, 2, dim=-1)
        y1, h1 = self.rnn1(x1, h1.contiguous())
        y2, h2 = self.rnn2(x2, h2.contiguous())
        return torch.cat([y1, y2], dim=-1), torch.cat([h1, h2], dim=-1)


class DPGRNN(nn.Module):
    """Dual-Path Grouped RNN.

    intra: bidirectional across frequency bins within each frame — does NOT
           access future time steps, so the model remains causal.
    inter: unidirectional across time — causal by construction.
    """
    def __init__(self, input_size, width, hidden_size):
        super().__init__()
        self.input_size  = input_size
        self.width       = width
        self.hidden_size = hidden_size

        self.intra_rnn = GRNN(input_size, hidden_size // 2, bidirectional=True)
        self.intra_fc  = nn.Linear(hidden_size, hidden_size)
        self.intra_ln  = nn.LayerNorm((width, hidden_size), eps=1e-8)

        self.inter_rnn = GRNN(input_size, hidden_size, bidirectional=False)
        self.inter_fc  = nn.Linear(hidden_size, hidden_size)
        self.inter_ln  = nn.LayerNorm((width, hidden_size), eps=1e-8)

    def forward(self, x):
        """x: (B, C, T, F)"""
        # Intra: across frequency within each frame
        x = x.permute(0, 2, 3, 1)                                         # (B,T,F,C)
        intra_x = x.reshape(x.shape[0] * x.shape[1], x.shape[2], x.shape[3])  # (B*T,F,C)
        intra_x = self.intra_rnn(intra_x)[0]
        intra_x = self.intra_fc(intra_x)
        intra_x = intra_x.reshape(x.shape[0], -1, self.width, self.hidden_size)
        intra_x = self.intra_ln(intra_x)
        intra_out = x + intra_x

        # Inter: across time for each frequency bin (causal)
        x = intra_out.permute(0, 2, 1, 3)                                 # (B,F,T,C)
        inter_x = x.reshape(x.shape[0] * x.shape[1], x.shape[2], x.shape[3])  # (B*F,T,C)
        inter_x = self.inter_rnn(inter_x)[0]
        inter_x = self.inter_fc(inter_x)
        inter_x = inter_x.reshape(x.shape[0], self.width, -1, self.hidden_size)
        inter_x = inter_x.permute(0, 2, 1, 3)                            # (B,T,F,C)
        inter_x = self.inter_ln(inter_x)
        inter_out = intra_out + inter_x

        return inter_out.permute(0, 3, 1, 2)                              # (B,C,T,F)


class Encoder(nn.Module):
    def __init__(self, C=32):
        super().__init__()
        self.en_convs = nn.ModuleList([
            ConvBlock(3*3, C, (1,5), stride=(1,2), padding=(0,2)),
            ConvBlock(C,   C, (1,5), stride=(1,2), padding=(0,2), groups=2),
            GTConvBlock(C, C, (3,3), stride=(1,1), padding=(0,1), dilation=(1,1)),
            GTConvBlock(C, C, (3,3), stride=(1,1), padding=(0,1), dilation=(2,1)),
            GTConvBlock(C, C, (3,3), stride=(1,1), padding=(0,1), dilation=(5,1)),
        ])

    def forward(self, x):
        en_outs = []
        for layer in self.en_convs:
            x = layer(x)
            en_outs.append(x)
        return x, en_outs


class Decoder(nn.Module):
    def __init__(self, C=32):
        super().__init__()
        self.de_convs = nn.ModuleList([
            GTConvBlock(C, C, (3,3), stride=(1,1), padding=(2*5,1), dilation=(5,1), use_deconv=True),
            GTConvBlock(C, C, (3,3), stride=(1,1), padding=(2*2,1), dilation=(2,1), use_deconv=True),
            GTConvBlock(C, C, (3,3), stride=(1,1), padding=(2*1,1), dilation=(1,1), use_deconv=True),
            ConvBlock(C, C, (1,5), stride=(1,2), padding=(0,2), groups=2, use_deconv=True),
            ConvBlock(C,  2, (1,5), stride=(1,2), padding=(0,2), use_deconv=True, is_last=True),
        ])

    def forward(self, x, en_outs):
        for i, layer in enumerate(self.de_convs):
            x = layer(x + en_outs[len(self.de_convs) - 1 - i])
        return x


class CRM(nn.Module):
    """Complex Ratio Mask — recovers both magnitude and phase."""
    def forward(self, mask, spec):
        """
        mask: (B, 2, T, F)
        spec: (B, 2, T, F)
        """
        s_real = spec[:,0] * mask[:,0] - spec[:,1] * mask[:,1]
        s_imag = spec[:,1] * mask[:,0] + spec[:,0] * mask[:,1]
        return torch.stack([s_real, s_imag], dim=1)   # (B, 2, T, F)


# ── 48 kHz GTCRN ──────────────────────────────────────────────────────────────

class GTCRN48k(nn.Module):
    """
    GTCRN adapted for 48 kHz / N_FFT=512 / HOP=256 (5.33 ms latency).

    Input:  mic STFT  (B, N_FREQ, T, 2)   — real/imag stacked on last dim
    Output: enhanced STFT (B, N_FREQ, T, 2)

    C=32 (default): ~90K params — 4× the original 24K, enough capacity for
    feedback + noise separation on simulated data.
    C=16: original 24K param version (too small for source separation).
    """
    def __init__(self, C=32):
        super().__init__()
        self.erb     = ERB(_ERB_LIN, _ERB_COMP, nfft=N_FFT, high_lim=_ERB_HIGH, fs=SR)
        self.sfe     = SFE(3, 1)
        self.encoder = Encoder(C)
        self.dpgrnn1 = DPGRNN(C, 33, C)
        self.dpgrnn2 = DPGRNN(C, 33, C)
        self.decoder = Decoder(C)
        self.crm     = CRM()

    def forward(self, spec):
        """
        spec: (B, F, T, 2)

        Returns
        -------
        enhanced: (B, F, T, 2)
        """
        # Build 3-channel input: [mag, real, imag]
        spec_real = spec[..., 0].permute(0, 2, 1)   # (B, T, F)
        spec_imag = spec[..., 1].permute(0, 2, 1)
        spec_mag  = torch.sqrt(spec_real**2 + spec_imag**2 + 1e-12)
        feat = torch.stack([spec_mag, spec_real, spec_imag], dim=1)  # (B, 3, T, F)

        feat = self.erb.bm(feat)    # (B, 3, T, 129)
        feat = self.sfe(feat)       # (B, 9, T, 129)

        feat, en_outs = self.encoder(feat)
        feat = self.dpgrnn1(feat)
        feat = self.dpgrnn2(feat)

        m_feat = self.decoder(feat, en_outs)          # (B, 2, T, 129)
        m      = self.erb.bs(m_feat)                  # (B, 2, T, F)

        # Apply CRM: spec is (B,F,T,2), need (B,2,T,F)
        enhanced = self.crm(m, spec.permute(0, 3, 2, 1))   # (B, 2, T, F)
        return enhanced.permute(0, 3, 2, 1)                 # (B, F, T, 2)


# ── Loss ──────────────────────────────────────────────────────────────────────

class HybridLoss48k(nn.Module):
    """
    Hybrid STFT + SI-SNR loss, adapted from gtcrn/loss.py for 48 kHz.

    Components:
      30 × complex compressed-MSE  (power 0.7 — equal weight quiet/loud bins)
      70 × magnitude compressed-MSE (power 0.3)
       1 × SI-SNR  (scale-invariant, time-domain perceptual quality)

    The STFT terms give dense per-bin gradient every frame.
    The SI-SNR gives sequence-level perceptual quality signal.
    Together they are more stable than pure SI-SDR for feedback suppression
    because the per-bin terms penalise spectral distortion at non-ring bins
    even when SI-SNR is already decent.
    """
    def __init__(self):
        super().__init__()
        self._window = None   # lazy-init on first forward (avoids device mismatch)

    def _get_window(self, device):
        if self._window is None or self._window.device != device:
            self._window = torch.hann_window(N_FFT).pow(0.5).to(device)
        return self._window

    def forward(self, pred_stft, true_stft):
        """
        pred_stft, true_stft: (B, F, T, 2)
        """
        device = pred_stft.device
        window = self._get_window(device)

        pred_r = pred_stft[..., 0]   # (B, F, T)
        pred_i = pred_stft[..., 1]
        true_r = true_stft[..., 0]
        true_i = true_stft[..., 1]

        pred_mag = torch.sqrt(pred_r**2 + pred_i**2 + 1e-12)
        true_mag = torch.sqrt(true_r**2 + true_i**2 + 1e-12)

        # Power-compressed complex MSE (0.7 exponent)
        pred_rc = pred_r / (pred_mag ** 0.7)
        pred_ic = pred_i / (pred_mag ** 0.7)
        true_rc = true_r / (true_mag ** 0.7)
        true_ic = true_i / (true_mag ** 0.7)
        real_loss = nn.functional.mse_loss(pred_rc, true_rc)
        imag_loss = nn.functional.mse_loss(pred_ic, true_ic)

        # Power-compressed magnitude MSE (0.3 exponent)
        mag_loss = nn.functional.mse_loss(pred_mag ** 0.3, true_mag ** 0.3)

        # SI-SNR in time domain
        y_pred = torch.istft(
            pred_r + 1j * pred_i, N_FFT, HOP, N_FFT, window=window)
        y_true = torch.istft(
            true_r + 1j * true_i, N_FFT, HOP, N_FFT, window=window)

        # Scale-invariant projection
        dot        = (y_true * y_pred).sum(dim=-1, keepdim=True)
        s_tgt_norm = (y_true * y_true).sum(dim=-1, keepdim=True) + 1e-8
        s_target   = dot / s_tgt_norm * y_true
        sisnr = -torch.log10(
            s_target.norm(dim=-1)**2 /
            ((y_pred - s_target).norm(dim=-1)**2 + 1e-8) + 1e-8
        ).mean()

        return 30 * (real_loss + imag_loss) + 70 * mag_loss + sisnr


if __name__ == '__main__':
    model = GTCRN48k()
    n = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'GTCRN48k: {n:,} parameters')

    spec = torch.randn(1, N_FREQ, 100, 2)
    out  = model(spec)
    print(f'spec {tuple(spec.shape)} → enhanced {tuple(out.shape)}')

    loss_fn = HybridLoss48k()
    tgt  = torch.randn(1, N_FREQ, 100, 2)
    loss = loss_fn(out, tgt)
    print(f'loss: {loss.item():.4f}')
