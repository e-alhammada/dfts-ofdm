"""dfts_day4.py - DFT-s-OFDM link, Day 4 (uncoded, no DM-RS, no scrambling).

Transmitter: 38.211 16-QAM -> transform precoding (DFT) -> grid -> IFFT + CP
Channel:     ideal, then a fixed 3-tap channel (no noise)
Receiver:    remove CP -> FFT -> extract subcarriers -> one-tap ZF -> IDFT -> hard demap
Every transmitter signal is compared with the MATLAB reference ref_day4.mat.
"""
from pathlib import Path

import numpy as np
from scipy.io import loadmat
from scipy.signal import lfilter

from params import N_FFT, N_SC, M, N_SYM, CP_LEN, DATA_SYMS

# Scale of the IFFT. If the "wave" check fails and the printed ratio is about 32,
# set OFDM_SCALE = np.sqrt(N_FFT): I am not sure which scaling nrOFDMModulate uses.
OFDM_SCALE = 1.0

# Grid subcarrier k (0..623) -> FFT bin. The grid is centred: k = 312 sits at DC,
# lower half goes to the negative-frequency bins at the top of the FFT (like nrOFDMModulate).
BINS = (np.arange(N_SC) - N_SC // 2) % N_FFT


# ---------------------------------------------------------------- modulation
def qam16_mod(bits):
    """TS 38.211 §5.1.3 16-QAM, unit average power. bits: 1-D 0/1 array, length multiple of 4."""
    b = bits.reshape(-1, 4).astype(float)  # one row per symbol: b0 b1 b2 b3; float avoids uint8 wrap-around (1 - 2*1 = 255!)
    re = (1 - 2 * b[:, 0]) * (2 - (1 - 2 * b[:, 2]))
    im = (1 - 2 * b[:, 1]) * (2 - (1 - 2 * b[:, 3]))
    return (re + 1j * im) / np.sqrt(10)


def qam16_demod_hard(s):
    """Hard decision, inverse of qam16_mod. Returns a 1-D uint8 bit array."""
    s = s * np.sqrt(10)                    # back to the {±1, ±3} grid
    b = np.empty((s.size, 4), dtype=np.uint8)
    b[:, 0] = s.real < 0                   # sign of I
    b[:, 1] = s.imag < 0                   # sign of Q
    b[:, 2] = np.abs(s.real) > 2           # outer ring in I
    b[:, 3] = np.abs(s.imag) > 2           # outer ring in Q
    return b.ravel()                       # row-major: b0 b1 b2 b3 b0 b1 ...


# ---------------------------------------------------------------- transmitter
def transform_precode(qam):
    """M-point unitary DFT on each data symbol. Returns M x 12."""
    a = qam.reshape(M, -1, order="F")      # column-major, same as MATLAB reshape(qam, M, [])
    return np.fft.fft(a, axis=0) / np.sqrt(M)   # axis=0: down the columns, like MATLAB fft


def map_grid(d):
    """Put the precoded symbols on PRBs 0..49 of the data symbols. Returns N_SC x N_SYM."""
    grid = np.zeros((N_SC, N_SYM), dtype=complex)
    grid[:M, DATA_SYMS] = d                # DM-RS symbols 2 and 11 stay empty
    return grid


def ofdm_mod(grid):
    """IFFT per symbol, then add the (variable-length) cyclic prefix. Returns a 1-D waveform."""
    X = np.zeros((N_FFT, N_SYM), dtype=complex)
    X[BINS, :] = grid                      # centred mapping onto the FFT bins
    x = np.fft.ifft(X, axis=0) * OFDM_SCALE
    symbols = [np.concatenate([x[-CP_LEN[l]:, l], x[:, l]]) for l in range(N_SYM)]
    return np.concatenate(symbols)         # 15360 samples = 1 slot


# ---------------------------------------------------------------- receiver
def ofdm_demod(y):
    """Remove CP, FFT per symbol, back to the N_SC x N_SYM grid."""
    Y = np.zeros((N_FFT, N_SYM), dtype=complex)
    start = 0
    for l in range(N_SYM):                 # 14 iterations only: a loop is fine here
        start += CP_LEN[l]                 # skip the CP
        Y[:, l] = np.fft.fft(y[start:start + N_FFT]) / OFDM_SCALE
        start += N_FFT
    return Y[BINS, :]


def extract_and_despread(grid_rx, H=None):
    """Take the data REs, optional one-tap ZF (noiseless check only), IDFT despreading."""
    r = grid_rx[:M, DATA_SYMS]
    if H is not None:
        r = r / H[:, None]                 # broadcast H (M,) over the 12 symbols; LMMSE comes on Day 5
    a = np.fft.ifft(r, axis=0) * np.sqrt(M)
    return a.ravel(order="F")


# ---------------------------------------------------------------- checks
def check(name, py, ref, tol=1e-10):
    """Relative error between a Python array and its MATLAB reference."""
    py, ref = np.asarray(py), np.asarray(ref)
    if py.size != ref.size:
        print(f"{name:5s} FAIL: size {py.shape} vs MATLAB {ref.shape}")
        return
    py, ref = py.ravel(order="F"), ref.ravel(order="F")
    err = np.linalg.norm(py - ref) / np.linalg.norm(ref)
    print(f"{name:5s} rel. error {err:.1e}  {'OK' if err < tol else 'FAIL'}")


if __name__ == "__main__":
    ref = loadmat(Path(__file__).parent / "ref_day4.mat")
    bits = ref["bits"].ravel().astype(np.uint8)          # (28800, 1) double -> (28800,) uint8

    # transmitter
    qam = qam16_mod(bits)
    d = transform_precode(qam)
    grid = map_grid(d)
    wave = ofdm_mod(grid)

    print("--- Python vs MATLAB ---")
    for name, py in [("qam", qam), ("d", d), ("grid", grid), ("wave", wave)]:
        check(name, py, ref[name])
    w_ref = ref["wave"].ravel()
    ratio = np.vdot(wave, w_ref) / np.vdot(wave, wave)   # least-squares scale MATLAB/Python
    print(f"wave scale MATLAB/Python: {ratio.real:.4f} {ratio.imag:+.4f}j")

    print("--- noiseless loopback ---")
    bits_hat = qam16_demod_hard(extract_and_despread(ofdm_demod(wave)))
    print("ideal channel: bit errors =", np.count_nonzero(bits_hat != bits))

    h = np.array([0.8, 0.5, 0.3])
    h = h / np.linalg.norm(h)                            # unit energy; delay spread 2 samples < CP
    y = lfilter(h, 1, wave)                              # same as MATLAB filter(h, 1, wave)
    H = np.fft.fft(h, N_FFT)[BINS[:M]]                   # channel frequency response on the data subcarriers
    bits_hat = qam16_demod_hard(extract_and_despread(ofdm_demod(y), H))
    print("3-tap channel: bit errors =", np.count_nonzero(bits_hat != bits))