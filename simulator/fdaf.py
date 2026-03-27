"""
simulator/fdaf.py — Frequency-Domain NLMS Adaptive Filter (overlap-save method)

Used for Stage 1 acoustic feedback cancellation. Given the reference signal
(what the PA is playing) and the mic signal, estimates the feedback path H(z)
and subtracts it from the mic in real time.

Per-block cost: 4 FFTs of size 2048 — < 1ms on Pi 5. No sample loops.
Filter length 1024 taps = 21.3ms at 48kHz — covers speaker-to-mic acoustic
delays for any stage geometry up to ~7m.

Converges online in ~0.5–2 seconds depending on signal level and mu.
No training required.
"""

import numpy as np


class FreqDomainNLMS:
    """Overlap-save frequency-domain NLMS adaptive filter.

    Args:
        filter_len:  FIR tap count. Must span the max acoustic path delay.
                     At 48kHz: speaker 3m away = 421 taps minimum.
                     Default 1024 (21.3ms) covers any realistic stage geometry.
        block_size:  Samples per callback — must match inference HOP (480 at 48kHz).
        mu:          NLMS step size. 0.02 = moderate (stable, converges in ~1–2s).
                     Increase to 0.05–0.1 for faster convergence if signal is strong.
        eps:         Power floor — prevents divide-by-zero during silence.
    """

    def __init__(self, filter_len: int = 1024, block_size: int = 480,
                 mu: float = 0.02, eps: float = 1e-6):
        self.L   = filter_len
        self.B   = block_size
        # FFT size: smallest power of 2 >= L + B - 1
        self.N   = 1 << (filter_len + block_size - 1).bit_length()
        self.mu  = mu
        self.eps = eps
        # Frequency-domain filter weights (complex)
        self.W       = np.zeros(self.N // 2 + 1, dtype=np.complex128)
        # Overlap-save reference input buffer (N samples)
        self.ref_buf = np.zeros(self.N, dtype=np.float64)

    def process(self, mic_block: np.ndarray, ref_block: np.ndarray) -> np.ndarray:
        """Process one block. Returns the residual (feedback-cancelled signal).

        mic_block: (B,) float32 — raw microphone signal
        ref_block: (B,) float32 — reference = what the speaker is currently playing
        Returns:   (B,) float32 — mic with estimated feedback subtracted
        """
        # Overlap-save: shift buffer left, append new reference block
        self.ref_buf = np.roll(self.ref_buf, -self.B)
        self.ref_buf[-self.B:] = ref_block.astype(np.float64)

        # Frequency-domain reference
        X = np.fft.rfft(self.ref_buf)                         # (N//2+1,) complex

        # Estimated feedback via linear convolution (overlap-save valid output = last B samples)
        y = np.fft.irfft(self.W * X)[self.N - self.B:].astype(np.float32)

        # Residual after cancellation
        e = mic_block - y

        # Constrained NLMS update — enforce causal FIR of length L to prevent circular aliasing
        e_padded               = np.zeros(self.N, dtype=np.float64)
        e_padded[self.N - self.B:] = e.astype(np.float64)
        E                      = np.fft.rfft(e_padded)

        grad           = np.fft.irfft(E * np.conj(X))    # cross-correlation gradient (N,)
        grad[self.L:]  = 0.0                              # zero non-causal taps (constraint)
        G              = np.fft.rfft(grad)

        self.W += (self.mu / (np.mean(np.abs(X) ** 2) + self.eps)) * G

        return e

    def reset(self):
        """Reset filter state — call when acoustic environment changes significantly."""
        self.W[:]       = 0.0
        self.ref_buf[:] = 0.0
