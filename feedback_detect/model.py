"""
feedback_detect/model.py — Lightweight causal feedback frequency detector.

Architecture:
  1. Log-magnitude STFT input  (no phase needed for detection)
  2. Per-frame 1D conv across frequency bins  (local spectral context)
  3. Per-bin causal GRU over time  (tracks exponential energy growth)
  4. Sigmoid per bin  → feedback probability in [0, 1]

~6 K parameters. Streaming-compatible: hidden state persists across chunks.

The model answers one question per (bin, frame): "is this frequency bin
currently in a Larsen loop?" Everything else — notch placement, notch
depth, hold time — is handled by the signal-processing notch bank.
"""

import torch
import torch.nn as nn

SR      = 48000
N_FFT   = 960
HOP     = 480
N_FREQ  = N_FFT // 2 + 1   # 481 bins, 50 Hz resolution

FREQ_CH    = 16   # channels from frequency conv
GRU_HIDDEN = 32   # hidden size per bin
N_DELTA    = 4    # number of input feature channels: log_mag + 3 delta scales


def prepare_features(mag: 'torch.Tensor') -> 'torch.Tensor':
    """
    Build (B, N_DELTA, N_FREQ, T) feature tensor from raw STFT magnitude.

    Uses causal torch.roll — correct when T > 10 (full training sequences).
    For streaming (T=1 per callback), build the feature tensor from an
    external lm_history buffer instead and pass it directly to forward().

    Features: [log_mag, Δ1 (~10ms), Δ4 (~40ms), Δ10 (~100ms)]
    Exponential feedback growth → all deltas positive for many frames.
    Vocal transients → brief positive delta, then reversal.
    """
    import torch
    lm = torch.log(mag + 1e-8)       # (B, F, T)

    def causal_delta(k):
        d = lm - torch.roll(lm, k, dims=2)
        d[..., :k] = 0.0             # zero-pad where roll wraps around
        return d

    return torch.stack([lm,
                        causal_delta(1),
                        causal_delta(4),
                        causal_delta(10)], dim=1)   # (B, N_DELTA, F, T)


class FeedbackDetector(nn.Module):
    """
    Per-bin causal feedback detector.

    Input:  mic STFT magnitude  (B, N_FREQ, T)
    Output: per-bin feedback probability  (B, N_FREQ, T),  GRU hidden state

    Stage 1 — FreqConv (per frame):
        1D conv across frequency bins captures local spectral spread.
        Feedback tends to excite adjacent bins; vocal notes have a different
        spectral profile. Receptive field ≈ ±200 Hz (kernel=9 at 50 Hz/bin).

    Stage 2 — TemporalGRU (per bin):
        Causal GRU tracks each bin's energy history. Exponentially growing
        energy → hidden state encodes "this bin has been growing N frames."
        Hidden state persists across streaming chunks in live.py.
    """

    def __init__(self, n_freq=N_FREQ, freq_ch=FREQ_CH, gru_hidden=GRU_HIDDEN):
        super().__init__()
        self.n_freq     = n_freq
        self.freq_ch    = freq_ch
        self.gru_hidden = gru_hidden

        # Local frequency context — applied independently to each frame.
        # Input has N_DELTA=4 channels: log_mag, delta-1, delta-4, delta-10.
        # Deltas make exponential growth directly visible as a positive trend
        # rather than forcing the GRU to learn growth from raw magnitudes.
        self.freq_conv = nn.Sequential(
            nn.Conv1d(N_DELTA, freq_ch, kernel_size=9, padding=4),
            nn.PReLU(),
            nn.Conv1d(freq_ch, freq_ch, kernel_size=5, padding=2),
            nn.PReLU(),
        )

        # Causal GRU: input is freq_ch features per (bin, frame)
        # Processes (B*N_FREQ) independent sequences over time
        self.gru = nn.GRU(freq_ch, gru_hidden, num_layers=1, batch_first=True)

        # Per-bin detection head
        self.fc = nn.Linear(gru_hidden, 1)

    def forward(self, features, h=None):
        """
        features : (B, N_DELTA, N_FREQ, T)  — from prepare_features() or streaming buffer
        h        : (1, B*N_FREQ, gru_hidden) or None — GRU hidden state

        For training:
            features = prepare_features(mic_mag)   # uses torch.roll, T > 10
        For streaming (T=1 per callback):
            build features manually from lm_history buffer (see live.py)

        Returns
        -------
        prob  : (B, N_FREQ, T)  — feedback probability per bin per frame
        h_new : (1, B*N_FREQ, gru_hidden)  — updated hidden state
        """
        B, _, F, T = features.shape

        # --- Frequency conv (per frame) ---
        x = features.permute(0, 3, 1, 2).reshape(B * T, N_DELTA, F)  # (B*T, N_DELTA, F)
        x = self.freq_conv(x)                                          # (B*T, freq_ch, F)
        x = x.reshape(B, T, self.freq_ch, F)                          # (B, T, C, F)
        x = x.permute(0, 3, 1, 2)                                     # (B, F, T, C)

        # --- Temporal GRU (per bin) ---
        x = x.reshape(B * F, T, self.freq_ch)              # (B*F, T, C)
        x, h_new = self.gru(x, h)                          # (B*F, T, gru_hidden)

        # --- Detection head ---
        prob = torch.sigmoid(self.fc(x))                   # (B*F, T, 1)
        prob = prob.reshape(B, F, T)                       # (B, F, T)

        return prob, h_new


if __name__ == '__main__':
    model = FeedbackDetector()
    n = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'FeedbackDetector: {n:,} parameters')
    mag  = torch.rand(1, N_FREQ, 100)
    feat = prepare_features(mag)
    prob, h = model(feat)
    print(f'mag {tuple(mag.shape)} → features {tuple(feat.shape)} → prob {tuple(prob.shape)}, h {tuple(h.shape)}')
