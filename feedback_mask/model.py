"""
feedback_mask/model.py — Per-bin GRU feedback suppressor.

Architecture mirrors FeedbackDetector (feedback_detect/model.py), which is
proven to learn ring detection. The only differences:

  1. Input:  raw complex STFT (B, F, T, 2) instead of magnitude
  2. Output: real suppression mask applied to complex STFT → (B, F, T, 2)
  3. FC bias initialized to 3.0  →  sigmoid(3) ≈ 0.95  →  passthrough at init

Why this architecture instead of GTCRN:
  - GTCRN compresses 481 bins to 129 ERB bins — destroys the frequency
    resolution needed to find narrow 50 Hz feedback rings
  - Per-bin GRU gives each frequency bin its own temporal context without
    any cross-bin blurring
  - Real mask (not CRM): preserves input phase exactly, just attenuates
    amplitude at ring bins. Simpler optimisation landscape.
  - ~6 K params vs 24 K for GTCRN, but better suited to this task

Signal flow:
  STFT magnitude → log_mag + temporal deltas → FreqConv (per frame)
  → per-bin GRU → sigmoid mask → enhanced = mask × mic_stft

~6 K parameters. Causal, streaming-compatible.
"""

import torch
import torch.nn as nn

SR     = 48000
N_FFT  = 960
HOP    = 480
N_FREQ = N_FFT // 2 + 1   # 481 bins, 50 Hz resolution

FREQ_CH    = 32   # 16→32: more capacity to distinguish narrow ring spike from broad speech formant
GRU_HIDDEN = 32
N_DELTA    = 6   # log_mag, Δ1, Δ4, Δ10, Δ50, Δ200


def _prepare_features(spec):
    """
    spec: (B, F, T, 2) complex STFT → (B, N_DELTA, F, T) feature tensor.

    Features: [log_mag, Δ1 (~10ms), Δ4 (~40ms), Δ10 (~100ms), Δ50 (~500ms), Δ200 (~2s)]

    Why Δ50 and Δ200:
      Ring builds up over seconds; speech formants are transient (50-500ms).
      At steady-state ring, Δ10 diff (ring vs bg) ≈ 0.011 — barely usable.
      Δ200 diff ≈ 0.134 during buildup — 12× more discriminative.
      Without long deltas the GRU uses only log_mag magnitude as a suppression
      proxy, which partially suppresses high-energy speech formants too.
    """
    mag = (spec[..., 0] ** 2 + spec[..., 1] ** 2 + 1e-12).sqrt()  # (B, F, T)
    lm  = torch.log(mag)

    def causal_delta(k):
        d = lm - torch.roll(lm, k, dims=2)
        d[..., :k] = 0.0
        return d

    return torch.stack([lm,
                        causal_delta(1),
                        causal_delta(4),
                        causal_delta(10),
                        causal_delta(50),
                        causal_delta(200)], dim=1)   # (B, N_DELTA, F, T)


class FeedbackMaskNet(nn.Module):
    """
    Per-bin GRU feedback suppressor.

    Input:  mic STFT  (B, N_FREQ, T, 2)
    Output: enhanced STFT (B, N_FREQ, T, 2), GRU hidden state

    Stage 1 — FreqConv (per frame):
        Same as FeedbackDetector: 1D conv across frequency bins gives local
        spectral context (~±200 Hz receptive field at 50 Hz/bin).

    Stage 2 — TemporalGRU (per bin):
        Causal GRU with persistent hidden state. Tracks whether each bin
        has been growing (ring) vs transient/stable (voice/silence).

    Stage 3 — Mask + apply:
        sigmoid → real mask in (0, 1). Applied to complex STFT.
        FC bias initialised to 3.0 so sigmoid(3) ≈ 0.95 at init (passthrough).
        Non-ring bins see near-zero gradient → stay at passthrough.
        Ring bins see large gradient → mask pushed toward 0 (suppress).
    """

    def __init__(self, n_freq=N_FREQ, freq_ch=FREQ_CH, gru_hidden=GRU_HIDDEN):
        super().__init__()
        self.n_freq     = n_freq
        self.freq_ch    = freq_ch
        self.gru_hidden = gru_hidden

        self.freq_conv = nn.Sequential(
            nn.Conv1d(N_DELTA, freq_ch, kernel_size=9, padding=4),
            nn.PReLU(),
            nn.Conv1d(freq_ch, freq_ch, kernel_size=5, padding=2),
            nn.PReLU(),
        )

        self.gru = nn.GRU(freq_ch, gru_hidden, num_layers=1, batch_first=True)

        self.fc = nn.Linear(gru_hidden, 1)
        # bias=0.0 → sigmoid(0)=0.5 at init = class-balanced BCE equilibrium.
        # At init, ring and nonring gradients are exactly equal (both BCE(0.5,{0,1})≈0.693).
        # This lets the GRU learn to differentiate ring vs non-ring from a neutral starting
        # point. bias=3.0 was wrong: ring gradient (60×) dominated nonring at init, training
        # the GRU to suppress everything before passthrough could be established.
        nn.init.constant_(self.fc.bias, 0.0)

    def forward(self, spec, h=None):
        """
        spec:  (B, F, T, 2)               — complex STFT (real/imag stacked)
        h:     (1, B*F, gru_hidden)|None  — GRU state; None resets to zero

        For training on full clips: pass h=None each clip.
        For streaming inference: pass h back in on every chunk.

        Returns
        -------
        enhanced : (B, F, T, 2)
        h_new    : (1, B*F, gru_hidden)
        """
        B, F, T, _ = spec.shape

        feat = _prepare_features(spec)                               # (B, N_DELTA, F, T)

        # FreqConv: per-frame local spectral context
        x = feat.permute(0, 3, 1, 2).reshape(B * T, N_DELTA, F)    # (B*T, N_DELTA, F)
        x = self.freq_conv(x)                                        # (B*T, freq_ch, F)
        x = x.reshape(B, T, self.freq_ch, F).permute(0, 3, 1, 2)   # (B, F, T, freq_ch)

        # Per-bin GRU
        x = x.reshape(B * F, T, self.freq_ch)                       # (B*F, T, freq_ch)
        x, h_new = self.gru(x, h)                                    # (B*F, T, gru_hidden)

        # Mask
        mask = torch.sigmoid(self.fc(x))                             # (B*F, T, 1)
        mask = mask.reshape(B, F, T, 1)                              # (B, F, T, 1)

        enhanced = spec * mask                                       # (B, F, T, 2)
        return enhanced, mask.squeeze(-1), h_new                     # mask: (B, F, T)


if __name__ == '__main__':
    model = FeedbackMaskNet()
    n = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'FeedbackMaskNet: {n:,} parameters')
    spec = torch.randn(1, N_FREQ, 100, 2)
    out, h = model(spec)
    print(f'spec {tuple(spec.shape)} → enhanced {tuple(out.shape)}, h {tuple(h.shape)}')
