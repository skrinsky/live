"""
train/model.py — Frequency-Domain Kalman Filter Network (FDKFNet)

Kalman filter per frequency bin + small GRU for covariance estimation.
Causal, streaming-compatible. ~500K parameters.

Based on NeuralKalmanAHS (Zhang et al. 2023, arXiv 2309.16049).
"""

import torch
import torch.nn as nn
import torchaudio

SR       = 48000
N_FFT    = 960
HOP      = 480
WIN_LEN  = 960
N_FREQS  = N_FFT // 2 + 1   # 481
N_BANDS  = 64                # ERB-compressed bands for GRU input
HIDDEN   = 128
N_LAYERS = 2


class FDKFNet(nn.Module):
    def __init__(self, n_freqs=N_FREQS, n_bands=N_BANDS, hidden=HIDDEN, n_layers=N_LAYERS):
        super().__init__()
        self.n_freqs = n_freqs
        self.n_bands = n_bands

        # Mel filterbank: compresses 481 FFT bins → 64 bands for GRU
        # (ERB and mel are equivalent for this purpose)
        fb = torchaudio.functional.melscale_fbanks(
            n_freqs=n_freqs, n_mels=n_bands,
            f_min=80.0, f_max=24000.0, sample_rate=SR, norm=None
        )  # (n_freqs, n_bands)
        self.register_buffer('mel_fb', fb)

        # GRU: estimates log(Q_k) and log(R_k) per ERB band per frame
        # Input: log power of mic, ref, and innovation per band (3 × n_bands)
        self.gru  = nn.GRU(n_bands * 3, hidden, num_layers=n_layers, batch_first=True)
        # Output: log(Q) and log(R) per band (interpolated back to all bins)
        self.proj = nn.Linear(hidden, n_bands * 2)

    def _to_erb(self, power):
        """power: (B, F) real → (B, n_bands) log-compressed"""
        return torch.log1p(power @ self.mel_fb)   # (B, n_bands)

    def init_state(self, batch_size=1, device='cpu'):
        F = self.n_freqs
        H = torch.zeros(batch_size, F, dtype=torch.cfloat,  device=device)
        P = torch.ones( batch_size, F, dtype=torch.float32, device=device) * 0.1
        h = torch.zeros(self.gru.num_layers, batch_size, self.gru.hidden_size, device=device)
        return H, P, h

    def forward_frame(self, mic_f, ref_f, H_prev, P_prev, gru_h):
        """
        One STFT frame.
        mic_f, ref_f : (B, F) complex
        H_prev       : (B, F) complex  — feedback path estimate
        P_prev       : (B, F) float    — error covariance
        Returns: speech_f, H_new, P_new, gru_h
        """
        ref_power  = ref_f.abs().pow(2)                       # (B, F) real
        innovation = mic_f - H_prev * ref_f                   # (B, F) complex

        # GRU features (ERB-compressed log power)
        feat = torch.cat([
            self._to_erb(mic_f.abs().pow(2)),
            self._to_erb(ref_power),
            self._to_erb(innovation.abs().pow(2)),
        ], dim=-1).unsqueeze(1)                               # (B, 1, 3*n_bands)

        gru_out, gru_h = self.gru(feat, gru_h)               # (B, 1, hidden)
        cov = self.proj(gru_out[:, 0, :])                     # (B, 2*n_bands)
        log_Q = cov[:, :self.n_bands]
        log_R = cov[:, self.n_bands:]

        # Interpolate covariances back to all frequency bins
        Q_k = torch.exp(log_Q @ self.mel_fb.T) + 1e-8        # (B, F)
        R_k = torch.exp(log_R @ self.mel_fb.T) + 1e-8        # (B, F)

        # VAD gate — freeze H and P updates when no feedback is present.
        # When mic energy >> ref energy, the signal is clean speech with no active
        # feedback loop. Updating H(z) in this state causes the filter to adapt to
        # speech harmonics, which corrupts the feedback path estimate. When feedback
        # risk returns, H(z) is wrong and takes time to re-converge — causing audible
        # distortion on exactly the sustained notes most vulnerable to feedback.
        #
        # Gate logic: if mic_power / (ref_power + eps) > VAD_RATIO, freeze.
        # VAD_RATIO=20.0 (13dB): feedback present only when ref is within 13dB of mic.
        # When ref is 13dB or more below mic, speaker output is low relative to mic —
        # no meaningful feedback path is active. Use .detach() so gate doesn't block grad.
        VAD_RATIO   = 20.0
        mic_power   = mic_f.abs().pow(2)                      # (B, F)
        vad_gate    = (mic_power / (ref_power + 1e-8) < VAD_RATIO).float().detach()
        # vad_gate = 1.0 → feedback present, update H/P normally
        # vad_gate = 0.0 → clean speech, hold H/P at previous values

        # Kalman update (gated)
        P_pred = P_prev + Q_k
        S_k    = ref_power * P_pred + R_k                     # innovation covariance
        K_k    = P_pred * ref_f.conj() / S_k                  # Kalman gain (complex)
        H_new  = H_prev + vad_gate * K_k * innovation         # freeze if no feedback
        # (K_k * ref_f).real is provably real: K_k = P_pred*conj(ref_f)/S_k, so
        # K_k*ref_f = P_pred*|ref_f|^2/S_k — a ratio of real scalars.
        # FRAGILE: .real is correct ONLY for this specific K_k formula. If you change
        # the Kalman gain expression, re-derive this identity before keeping .real.
        P_new  = (1.0 - vad_gate * (K_k * ref_f).real) * P_pred
        # Clamp both ends: min prevents divide-by-zero; max prevents unbounded growth
        # during silence (when ref ≈ 0, K_k ≈ 0, and P_new ≈ P_prev + Q_k each frame).
        # Without the max, P inflates over multi-second silences and K_k spikes when
        # the reference returns — causing transient divergence artifacts.
        P_new  = P_new.clamp(min=1e-8, max=100.0)

        speech_f = mic_f - H_new * ref_f                      # clean speech estimate
        return speech_f, H_new, P_new, gru_h

    def forward(self, mic_frames, ref_frames, H_0=None, P_0=None, gru_h=None):
        """
        mic_frames, ref_frames: (B, T, F) complex
        Returns: speech_frames (B, T, F) complex, final state tuple
        """
        B, T, F = mic_frames.shape
        if H_0 is None:
            H_0, P_0, gru_h = self.init_state(B, mic_frames.device)
        H_k, P_k = H_0, P_0
        out = []
        for t in range(T):
            s, H_k, P_k, gru_h = self.forward_frame(
                mic_frames[:, t, :], ref_frames[:, t, :], H_k, P_k, gru_h
            )
            out.append(s)
        return torch.stack(out, dim=1), (H_k, P_k, gru_h)
