# params.py — matches the MATLAB carrier above
N_FFT     = 1024
N_SC      = 624                     # 52 RBs × 12 subcarriers in the grid
M         = 600                     # 50 PRBs allocated = DFT size
N_SYM     = 14
CP_LEN    = [80] + [72]*6 + [80] + [72]*6   # list, 0-based: CP_LEN[0] = 80, CP_LEN[7] = 80
DMRS_SYMS = [2, 11]
DATA_SYMS = [l for l in range(N_SYM) if l not in DMRS_SYMS]  # 12 data symbols
QAM_BITS  = 4                       # 16-QAM