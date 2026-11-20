"""dfts_day5.py - DFT-s-OFDM link, Day 5 step 1: AWGN + per-subcarrier LMMSE (perfect CSI).

Step 1: flat channel h = [1] at Es/N0 = 14 dB matches Gray 16-QAM AWGN theory.
Step 2: BER vs Es/N0 (>= 100 bit errors per point), flat and 3-tap channels.
Reuses the Day 4 transmitter and receiver blocks.
"""
import matplotlib
matplotlib.use("Agg")                     # file output only (plt.show() hangs in your terminal)
import matplotlib.pyplot as plt
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


def sinr_eff_db(snr_db, h):
    """Post-despreading SINR of unbiased LMMSE: mu/(1-mu), mu = mean |H|^2/(|H|^2 + sigma2)."""
    H2 = np.abs(np.fft.fft(h, N_FFT)[BINS[:M]]) ** 2
    s2 = 10 ** (-np.atleast_1d(snr_db)[:, None] / 10)
    mu = np.mean(H2 / (H2 + s2), axis=1)
    return 10 * np.log10(mu / (1 - mu))


def ber_point(snr_db, h, rng, min_err=500, max_slots=5000):
    """Run slots until min_err bit errors (or max_slots). Returns (BER, bit errors, bits)."""
    errors = slots = 0
    while errors < min_err and slots < max_slots:
        b, _ = run_slot(snr_db, h, rng)
        errors += b
        slots += 1
    n_bits = slots * BITS_PER_SLOT
    return errors / n_bits, errors, n_bits


if __name__ == "__main__":
    snr_list = np.arange(0, 19, 2)                       # Es/N0 in dB
    h3 = np.array([0.8, 0.5, 0.3]) / np.linalg.norm([0.8, 0.5, 0.3])
    channels = {"flat": np.array([1.0]), "3tap": h3}

    results = {}
    for c, (name, h) in enumerate(channels.items()):
        ss = np.random.SeedSequence([1, c])              # master seed 1, one branch per channel
        rngs = [np.random.default_rng(s) for s in ss.spawn(len(snr_list))]   # independent stream per point
        ber = np.zeros(len(snr_list))
        print(f"--- {name} ---")
        for i, (snr_db, rng) in enumerate(zip(snr_list, rngs)):
            ber[i], err, n_bits = ber_point(snr_db, h, rng)
            ber_th, _ = theory_16qam(sinr_eff_db(snr_db, h)[0])   # 16-QAM theory at the effective SINR
            print(f"Es/N0 {snr_db:4.1f} dB  BER {ber[i]:.3e}  errors {err:5d}  bits {n_bits:8d}"
                  f"  ratio to theory at SINR_eff {ber[i] / ber_th:.3f}")
        results[name] = ber

    np.savez("ber_python.npz", snr=snr_list, ber_flat=results["flat"], ber_3tap=results["3tap"])

    snr_fine = np.linspace(0, 18, 181)
    plt.semilogy(snr_fine, theory_16qam(snr_fine)[0], "k-", label="16-QAM AWGN theory")
    plt.semilogy(snr_fine, theory_16qam(sinr_eff_db(snr_fine, h3))[0], "k--",
                 label="16-QAM theory at 3-tap SINR_eff (Gaussian approx.)")
    plt.semilogy(snr_list, results["flat"], "o", label="Python, flat h=[1], LMMSE")
    plt.semilogy(snr_list, results["3tap"], "s", mfc="none", label="Python, 3-tap, LMMSE")
    plt.xlabel("Es/N0 (dB)")
    plt.ylabel("BER")
    plt.title("Uncoded DFT-s-OFDM, 16-QAM, 50 PRB, perfect CSI")
    plt.grid(True, which="both", alpha=0.3)
    plt.legend()
    plt.savefig("ber_day5.png", dpi=150, bbox_inches="tight")
    print("saved ber_python.npz and ber_day5.png")