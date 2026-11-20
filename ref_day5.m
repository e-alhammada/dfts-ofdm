% ref_day5.m - MATLAB reference for Day 5: AWGN + unbiased LMMSE (perfect CSI), BER vs Es/N0.
% Same chain and settings as dfts_day5.py: 50 PRB, 16-QAM, flat and 3-tap channels,
% Es/N0 = 0:2:18 dB, at least 500 bit errors per point (cap 5000 slots).
% MATLAB uses its own random generator, so the curves agree statistically, not bit for bit.
carrier  = nrCarrierConfig('NSizeGrid',52,'SubcarrierSpacing',15);
Nfft     = 1024;
M        = 600;
dataSyms = setdiff(0:13, [2 11]);
nBits    = 4*M*numel(dataSyms);                      % 28800 bits per slot
snrList  =0:2:18;
minErr   = 500;  maxSlots = 5000;

bins = mod((0:623) - 312, Nfft) + 1;                 % grid row -> FFT bin (centred), 1-based
h3   = [0.8 0.5 0.3] / norm([0.8 0.5 0.3]);
chans = {1, h3};  names = {'flat', '3tap'};

% --- noiseless check: Python receiver uses plain fft, FFT window at the end of the CP (fraction 1)
g  = nrResourceGrid(carrier);  g(1:M, dataSyms+1) = 1;
gr = nrOFDMDemodulate(carrier, nrOFDMModulate(carrier, g, 'Windowing', 0), 'CyclicPrefixFraction', 1);
fprintf('noiseless OFDM loopback rel. error: %.1e (expect ~1e-15)\n', norm(gr(:)-g(:))/norm(g(:)));

ber = zeros(numel(chans), numel(snrList));
for c = 1:numel(chans)
    h = chans{c};
    Hf = fft(h, Nfft);  H = Hf(bins(1:M)).';         % M x 1, perfect CSI on the data subcarriers
    fprintf('--- %s ---\n', names{c});
    for i = 1:numel(snrList)
        s = RandStream('mrg32k3a', 'Seed', 1);       % one master seed ...
        s.Substream = (c-1)*numel(snrList) + i;      % ... independent substream per (channel, SNR)
        snr = 10^(snrList(i)/10);  s2t = 1/(Nfft*snr);
        err = 0;  slots = 0;
        while err < minErr && slots < maxSlots
            bits = randi(s, [0 1], nBits, 1);
            d    = fft(reshape(nrSymbolModulate(bits, '16QAM'), M, [])) / sqrt(M);
            grid = nrResourceGrid(carrier);  grid(1:M, dataSyms+1) = d;
            wave = nrOFDMModulate(carrier, grid, 'Windowing', 0);
            y    = filter(h, 1, wave);
            y    = y + sqrt(s2t/2)*complex(randn(s, size(y)), randn(s, size(y)));
            R    = nrOFDMDemodulate(carrier, y, 'CyclicPrefixFraction', 1);
            R    = R(1:M, dataSyms+1);
            W    = conj(H) ./ (abs(H).^2 + 1/snr);        % LMMSE, M x 1 (implicit expansion)
            mu   = mean(real(W .* H), 1);                  % bias (scalar here, same for all symbols)
            Z    = W .* R ./ mu;                           % unbiased
            a    = ifft(Z) * sqrt(M);                      % IDFT despreading
            bh   = nrSymbolDemodulate(a(:), '16QAM', 'DecisionType', 'Hard');
            err  = err + sum(bh ~= bits);
            slots = slots + 1;
        end
        ber(c, i) = err / (slots*nBits);
        fprintf('Es/N0 %4.1f dB  BER %.3e  errors %5d  bits %8d\n', snrList(i), ber(c,i), err, slots*nBits);
    end
end

snr = snrList;  ber_flat = ber(1,:);  ber_3tap = ber(2,:);
save('ber_matlab.mat', 'snr', 'ber_flat', 'ber_3tap', '-v7');   % -v7 so SciPy can read it