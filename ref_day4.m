% ref_day4.m — MATLAB reference for the Python port (no coding, no scrambling, no DM-RS)
carrier  = nrCarrierConfig('NSizeGrid',52,'SubcarrierSpacing',15);  % 10 MHz, 15 kHz
rng(1);                                                              % fixed seed
M        = 600;                                % 50 PRBs × 12 (52 not allowed with DFT-s)
dataSyms = setdiff(0:13, [2 11]);              % 12 data symbols, 0-based
bits     = randi([0 1], M*4*numel(dataSyms), 1);
qam      = nrSymbolModulate(bits, '16QAM');    % 38.211 Gray 16-QAM, unit power
d        = fft(reshape(qam, M, [])) / sqrt(M); % transform precoding, one column per symbol
grid     = nrResourceGrid(carrier);            % 624 × 14 zeros
grid(1:M, dataSyms+1) = d;                     % PRBs 0..49, data symbols only
wave     = nrOFDMModulate(carrier, grid, 'Windowing', 0);  % plain IFFT + CP
save('ref_day4.mat', 'bits', 'qam', 'd', 'grid', 'wave', '-v7');  % -v7 so SciPy can read it
size(wave)                                     % expect 15360 × 1