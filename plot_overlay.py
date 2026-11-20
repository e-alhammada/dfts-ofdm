"""plot_overlay.py - Python vs MATLAB BER curves (Day 5 proof, Gate A).
Needs ber_python.npz (from dfts_day5.py) and ber_matlab.mat (from ref_day5.m) in this folder."""
import matplotlib
matplotlib.use("Agg")                     # file output only
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat

from dfts_day5 import theory_16qam, sinr_eff_db

py = np.load("ber_python.npz")
ml = loadmat("ber_matlab.mat")
snr = py["snr"]
h3 = np.array([0.8, 0.5, 0.3]) / np.linalg.norm([0.8, 0.5, 0.3])

print("Es/N0   flat: Py/MATLAB   3-tap: Py/MATLAB")
for i, s in enumerate(snr):
    print(f"{s:5.1f}   {py['ber_flat'][i] / ml['ber_flat'].ravel()[i]:14.3f}"
          f"   {py['ber_3tap'][i] / ml['ber_3tap'].ravel()[i]:15.3f}")

snr_fine = np.linspace(0, 18, 181)
plt.semilogy(snr_fine, theory_16qam(snr_fine)[0], "k-", label="16-QAM AWGN theory")
plt.semilogy(snr_fine, theory_16qam(sinr_eff_db(snr_fine, h3))[0], "k--", label="theory at 3-tap SINR_eff")
plt.semilogy(snr, py["ber_flat"], "o", color="C0", label="Python, flat")
plt.semilogy(snr, ml["ber_flat"].ravel(), "x", color="C0", ms=9, label="MATLAB, flat")
plt.semilogy(snr, py["ber_3tap"], "s", mfc="none", color="C1", label="Python, 3-tap")
plt.semilogy(snr, ml["ber_3tap"].ravel(), "+", color="C1", ms=11, label="MATLAB, 3-tap")
plt.xlabel("Es/N0 (dB)")
plt.ylabel("BER")
plt.title("Uncoded DFT-s-OFDM, 16-QAM, 50 PRB, LMMSE, perfect CSI")
plt.grid(True, which="both", alpha=0.3)
plt.legend()
plt.savefig("ber_overlay.png", dpi=150, bbox_inches="tight")
print("saved ber_overlay.png")