"""dfts_day5.py - DFT-s-OFDM link, Day 5 step 1: AWGN + per-subcarrier LMMSE (perfect CSI).

Sanity check: flat channel h = [1] at Es/N0 = 14 dB must match Gray 16-QAM AWGN theory.
Reuses the Day 4 transmitter and receiver blocks.
"""
import numpy as np
from scipy.signal import lfilter
from scipy.special import erfc

from params import N_FFT, M, DATA_SYMS
from dfts_day4 import (BINS, qam16_mod, qam16_demod_hard, transform_precode,
                       map_grid, ofdm_mod, ofdm_demod)

BITS_PER_SLOT = 4 * M * len(DATA_SYMS)    # 4 x 600 x 12 = 28800


def add_awgn(x, snr_db, rng):
    """Complex AWGN with time-domain variance 1/(N_FFT*SNR) -> 1/SNR per subcarrier after the FFT."""
    sigma2_t = 1 / (N_FFT * 10 ** (snr_db / 10))
    n = np.sqrt(sigma2_t / 2) * (rng.standard_normal(x.shape) + 1j * rng.standard_normal(x.shape))
    return x + n


def lmmse_equalize(R, H, sigma2):
    """Per-subcarrier LMMSE, unbiased for DFT-s-OFDM.
    R: (M, n_sym) received data REs; H: (M,) or (M, n_sym); sigma2: noise variance per subcarrier.
    Returns the unbiased equalized REs and the post-despreading noise variance per OFDM symbol."""
    H = H if H.ndim == 2 else H[:, None]          # (M,) -> (M, 1), broadcasts over symbols
    W = np.conj(H) / (np.abs(H) ** 2 + sigma2)    # elementwise, like MATLAB ./
    mu = np.mean((W * H).real, axis=0)            # one bias per OFDM symbol (mean over subcarriers)
    sigma2_eff = 1 / mu - 1                       # noise variance after despreading (for LLRs later)
    return W * R / mu, sigma2_eff


def run_slot(snr_db, h, rng):
    """One slot: random bits -> TX -> channel h + AWGN -> LMMSE -> despread -> hard demap.
    Returns (bit errors, symbol errors)."""
    bits = rng.integers(0, 2, BITS_PER_SLOT, dtype=np.uint8)
    qam = qam16_mod(bits)
    wave = ofdm_mod(map_grid(transform_precode(qam)))

    y = add_awgn(lfilter(h, 1, wave), snr_db, rng)
    R = ofdm_demod(y)[:M, DATA_SYMS]                     # (600, 12) data REs
    H = np.fft.fft(h, N_FFT)[BINS[:M]]                   # perfect CSI on the data subcarriers

    Z, _ = lmmse_equalize(R, H, 10 ** (-snr_db / 10))
    a = (np.fft.ifft(Z, axis=0) * np.sqrt(M)).ravel(order="F")   # IDFT despreading
    bits_hat = qam16_demod_hard(a)

    bit_err = np.count_nonzero(bits_hat != bits)
    sym_err = np.count_nonzero(np.any((bits_hat != bits).reshape(-1, 4), axis=1))
    return bit_err, sym_err


def theory_16qam(esn0_db):
    """Exact Gray 16-QAM BER and SER over AWGN."""
    Q = lambda x: 0.5 * erfc(x / np.sqrt(2))
    d = np.sqrt(10 ** (esn0_db / 10) / 5)        # half min. distance / noise std per dimension
    ber = (3 * Q(d) + 2 * Q(3 * d) - Q(5 * d)) / 4
    ser = 1 - (1 - 1.5 * Q(d)) ** 2
    return ber, ser


if __name__ == "__main__":
    snr_db, n_slots, seed = 14.0, 10, 1
    ber_th, ser_th = theory_16qam(snr_db)

    for name, h in [("flat h=[1]", np.array([1.0])),
                    ("3-tap", np.array([0.8, 0.5, 0.3]) / np.linalg.norm([0.8, 0.5, 0.3]))]:
        rng = np.random.default_rng(seed)
        be = se = 0
        for _ in range(n_slots):
            b, s = run_slot(snr_db, h, rng)
            be, se = be + b, se + s
        ber, ser = be / (n_slots * BITS_PER_SLOT), se / (n_slots * BITS_PER_SLOT // 4)
        print(f"{name:11s} Es/N0 {snr_db} dB: BER {ber:.3e} ({be} errors), ratio {ber / ber_th:.3f};"
              f"  SER {ser:.3e}, ratio {ser / ser_th:.3f}")
    print(f"theory (AWGN): BER {ber_th:.3e}, SER {ser_th:.3e}")