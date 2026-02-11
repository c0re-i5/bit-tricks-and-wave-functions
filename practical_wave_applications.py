"""
=============================================================================
  PRACTICAL WAVE APPLICATIONS — Tools That Work
  Six working implementations built from wave principles
=============================================================================

  Everything here WORKS. These aren't toy demos — they're stripped-down
  versions of techniques used in production systems. Each one traces
  directly back to the Fourier/wavelet principles from the previous
  scripts and connects to the bit-level world of repo 1.
"""

import math
import struct
import time
import numpy as np

SEPARATOR = "=" * 72
SUBSEP = "─" * 72

# ── Shared helpers ───────────────────────────────────────────────────────────

def ascii_plot(values, width=64, height=13, label=""):
    """Simple ASCII waveform plotter."""
    n = len(values)
    if n == 0:
        return
    min_val = min(values)
    max_val = max(values)
    if max_val == min_val:
        max_val = min_val + 1.0

    indices = [int(i * (n - 1) / (width - 1)) for i in range(width)]
    sampled = [values[idx] for idx in indices]
    grid = [[" " for _ in range(width)] for _ in range(height)]

    if min_val < 0 < max_val:
        zr = int((max_val - 0) / (max_val - min_val) * (height - 1))
        zr = max(0, min(height - 1, zr))
        for c in range(width):
            grid[zr][c] = "·"

    for c in range(width):
        row = int((max_val - sampled[c]) / (max_val - min_val) * (height - 1))
        row = max(0, min(height - 1, row))
        grid[row][c] = "█"

    for c in range(1, width):
        r_prev = int((max_val - sampled[c-1]) / (max_val - min_val) * (height - 1))
        r_curr = int((max_val - sampled[c]) / (max_val - min_val) * (height - 1))
        r_prev, r_curr = max(0, min(height-1, r_prev)), max(0, min(height-1, r_curr))
        for r in range(min(r_prev, r_curr), max(r_prev, r_curr) + 1):
            if grid[r][c] == " ":
                grid[r][c] = "│"

    for r in range(height):
        if r == 0:
            al = f"{max_val:>8.2f}"
        elif r == height - 1:
            al = f"{min_val:>8.2f}"
        elif r == height // 2:
            al = f"{(max_val + min_val) / 2:>8.2f}"
        else:
            al = "        "
        print(f"    {al} ┤{''.join(grid[r])}")
    print(f"    {'':>8} └{'─' * width}")
    if label:
        print(f"    {'':>8}  {label}")


def ascii_spectrum(frequencies, magnitudes, width=60, height=13, max_freq=None):
    """Render a frequency spectrum as ASCII bars."""
    if max_freq:
        mask = [i for i, f in enumerate(frequencies) if f <= max_freq]
        frequencies = [frequencies[i] for i in mask]
        magnitudes = [magnitudes[i] for i in mask]
    if not magnitudes:
        return
    max_mag = max(magnitudes) or 1
    n = len(magnitudes)
    step = max(1, n // width)
    display_mags = []
    for i in range(0, min(n, width * step), step):
        display_mags.append(max(magnitudes[i:i + step]))
    w = len(display_mags)
    for row in range(height):
        threshold = max_mag * (height - row) / height
        if row == 0:
            al = f"{max_mag:>8.1f}"
        elif row == height - 1:
            al = f"{'0':>8}"
        else:
            al = "        "
        line = "".join("█" if m >= threshold else " " for m in display_mags)
        print(f"    {al} ┤{line}")
    print(f"    {'':>8} └{'─' * w}")


# =============================================================================
#  APPLICATION 1: JPEG-STYLE COMPRESSION (1D DCT)
#  The technique behind every photo on the internet
# =============================================================================

def app1_jpeg_compression():
    print(f"\n{SEPARATOR}")
    print("  APPLICATION 1: JPEG-STYLE COMPRESSION")
    print("  DCT + quantization — the core of image compression")
    print(SEPARATOR)

    print("""
    JPEG doesn't use the FFT directly — it uses the DCT
    (Discrete Cosine Transform), which is the REAL-VALUED cousin.

    The DCT is like the FFT but:
      • Uses only cosines (no sines), so output is real, not complex
      • Has better "energy compaction" — more energy in fewer coefficients
      • Doesn't assume periodicity (avoids boundary artifacts)

    This is why JPEG, MP3, AAC, H.264, and H.265 all use DCT.

    The compression pipeline:
      1. Split data into blocks of 8 samples
      2. DCT each block → frequency coefficients
      3. Quantize (round to fewer levels) → lossy step
      4. Only store the nonzero quantized coefficients
    """)

    def dct_1d(block):
        """
        Type-II DCT (the standard "DCT" used in JPEG).

        X[k] = Σ x[n] · cos(π·k·(2n+1) / (2N))

        Implemented from scratch to show the math.
        """
        N = len(block)
        result = []
        for k in range(N):
            total = 0.0
            for n in range(N):
                total += block[n] * math.cos(math.pi * k * (2 * n + 1) / (2 * N))
            # Normalization
            if k == 0:
                total *= math.sqrt(1 / N)
            else:
                total *= math.sqrt(2 / N)
            result.append(total)
        return result

    def idct_1d(coeffs):
        """Inverse DCT (Type III)."""
        N = len(coeffs)
        result = []
        for n in range(N):
            total = 0.0
            for k in range(N):
                c_k = coeffs[k]
                if k == 0:
                    c_k *= math.sqrt(1 / N)
                else:
                    c_k *= math.sqrt(2 / N)
                total += c_k * math.cos(math.pi * k * (2 * n + 1) / (2 * N))
            result.append(total)
        return result

    # Create a test "image row" (1D signal simulating pixel intensity)
    np.random.seed(42)
    N = 64
    t = np.linspace(0, 1, N)
    signal = (128 + 50 * np.sin(2 * np.pi * 3 * t) +
              30 * np.sin(2 * np.pi * 7 * t) +
              10 * np.sin(2 * np.pi * 15 * t) +
              5 * np.random.randn(N))
    signal = np.clip(signal, 0, 255)

    print("    Original 'image row' (64 pixel intensities):")
    ascii_plot(signal.tolist(), height=9, width=60)
    print()

    # Process in 8-sample blocks (like real JPEG)
    block_size = 8
    quality_levels = [100, 50, 25, 10, 5]

    print(f"    {'Quality':>8} {'Nonzero':>10} {'RMSE':>8} "
          f"{'PSNR (dB)':>10} {'Compression':>13}")
    print(f"    {'─'*8} {'─'*10} {'─'*8} {'─'*10} {'─'*13}")

    best_compressed = None

    for quality in quality_levels:
        # Quantization step size (higher = more loss)
        quant_step = max(1, (101 - quality) // 5)

        reconstructed = []
        total_nonzero = 0
        total_coeffs = 0

        for start in range(0, N, block_size):
            block = signal[start:start + block_size].tolist()
            if len(block) < block_size:
                block.extend([0] * (block_size - len(block)))

            # DCT
            coeffs = dct_1d(block)

            # Quantize
            quantized = [round(c / quant_step) for c in coeffs]
            nonzero = sum(1 for q in quantized if q != 0)
            total_nonzero += nonzero
            total_coeffs += len(quantized)

            # Dequantize
            dequantized = [q * quant_step for q in quantized]

            # Inverse DCT
            recon_block = idct_1d(dequantized)
            reconstructed.extend(recon_block[:min(block_size, N - start)])

        reconstructed = np.array(reconstructed[:N])
        rmse = np.sqrt(np.mean((signal[:N] - reconstructed) ** 2))
        if rmse > 0:
            psnr = 20 * np.log10(255 / rmse)
        else:
            psnr = float('inf')
        compression = total_coeffs / max(1, total_nonzero)

        print(f"    {quality:>7}% {total_nonzero:>10} {rmse:>8.2f} "
              f"{psnr:>10.1f} {compression:>12.1f}×")

        if quality == 25:
            best_compressed = reconstructed

    print()
    if best_compressed is not None:
        print("    Compressed at quality=25 (4× compression):")
        ascii_plot(best_compressed.tolist(), height=9, width=60)

    print("""
    ► The DCT is the Fourier Transform's practical cousin
    ► JPEG quality slider literally controls the quantization step
    ► Real JPEG also applies Huffman coding (repo 1!) after DCT
    ► Connection: DCT (this repo) + Huffman (repo 1) = JPEG
      Waves and bits working together in every photo you've ever seen
    """)


# =============================================================================
#  APPLICATION 2: AUDIO SYNTHESIZER
#  Building sounds from pure math
# =============================================================================

def app2_audio_synth():
    print(f"\n{SEPARATOR}")
    print("  APPLICATION 2: AUDIO SYNTHESIZER")
    print("  Building sounds from pure mathematics")
    print(SEPARATOR)

    print("""
    Since Fourier proved that any sound = sum of sine waves,
    we can BUILD any sound by adding sines at the right frequencies.

    Two classic synthesis techniques:
      1. ADDITIVE: directly sum harmonics (Fourier in reverse)
      2. FM (Frequency Modulation): modulate one wave's frequency
         with another, creating complex timbres cheaply
    """)

    N = 2048
    sr = 8000
    t = np.arange(N) / sr

    # --- Additive synthesis: build instrument timbres ---
    instruments = {
        "Pure tone": [(1.0,)],
        "Organ":     [(1.0,), (0.8,), (0.6,), (0.4,), (0.3,), (0.2,)],
        "Clarinet":  [(1.0,), (0,), (0.75,), (0,), (0.5,), (0,), (0.14,)],
        "Sawtooth":  [(1/k,) for k in range(1, 16)],
        "Square":    [(1/k,) if k % 2 == 1 else (0,) for k in range(1, 16)],
    }

    freq = 220  # A3

    print("  ── Additive Synthesis: different timbres from harmonics ──\n")

    for name, harmonics in instruments.items():
        signal = np.zeros(N)
        for k, (amp,) in enumerate(harmonics, 1):
            signal += amp * np.sin(2 * np.pi * freq * k * t)
        # Normalize
        signal = signal / (np.max(np.abs(signal)) + 1e-10)

        # Count harmonics
        n_harm = sum(1 for (a,) in harmonics if a > 0)
        print(f"    {name} ({n_harm} harmonics):")
        ascii_plot(signal[:512].tolist(), height=7, width=60)
        print()

    # --- FM Synthesis ---
    print("  ── FM Synthesis: complex sounds from just 2 oscillators ──\n")

    print("""
    FM synthesis: f(t) = sin(2π·f_c·t + β·sin(2π·f_m·t))

    f_c = carrier frequency (the pitch you hear)
    f_m = modulation frequency (controls timbre)
    β   = modulation index (controls richness/brightness)
    """)

    carrier = 220
    modulator = 220  # 1:1 ratio → harmonic spectrum

    betas = [0, 1, 3, 5, 10]
    for beta in betas:
        signal = np.sin(2 * np.pi * carrier * t +
                        beta * np.sin(2 * np.pi * modulator * t))
        # Spectrum
        X = np.abs(np.fft.fft(signal))[:N//2] / N
        freqs = np.fft.fftfreq(N, 1/sr)[:N//2]

        n_peaks = np.sum(X > 0.02)
        print(f"    β = {beta:>2}  ({n_peaks} significant harmonics):")
        ascii_plot(signal[:512].tolist(), height=5, width=60)
        print()

    print("""
    As β increases, the FM signal develops more and more harmonics.
    With just TWO oscillators, FM synthesis creates rich, complex
    timbres — this powered the Yamaha DX7, one of the best-selling
    synthesizers ever.

    ► The SPECTRUM of FM synthesis is given by Bessel functions —
      the same math that describes vibrating drums and radio antennas.
    ► FM radio uses the exact same principle: carry audio information
      by modulating the FREQUENCY of a carrier wave.
    """)


# =============================================================================
#  APPLICATION 3: SPECTRAL NOISE FILTER
#  Recover signals buried in noise using the FFT
# =============================================================================

def app3_noise_filter():
    print(f"\n{SEPARATOR}")
    print("  APPLICATION 3: SPECTRAL NOISE FILTER")
    print("  FFT-based denoising that actually works")
    print(SEPARATOR)

    print("""
    The simplest FFT application: identify signal frequencies,
    zero out everything else, inverse FFT.

    Steps:
    1. FFT the noisy signal → frequency domain
    2. Identify signal peaks (above noise floor)
    3. Zero out non-peak frequencies (spectral gating)
    4. Inverse FFT → cleaned signal
    """)

    np.random.seed(42)
    N = 1024
    fs = 1024
    t = np.arange(N) / fs

    # Clean signal: 3 distinct frequencies
    f1, f2, f3 = 40, 85, 150
    clean = (1.0 * np.sin(2 * np.pi * f1 * t) +
             0.7 * np.sin(2 * np.pi * f2 * t) +
             0.4 * np.sin(2 * np.pi * f3 * t))

    # Add heavy noise
    noise = np.random.randn(N) * 1.5
    noisy = clean + noise

    snr_before = 10 * np.log10(np.mean(clean**2) / np.mean(noise**2))

    print(f"    Clean signal: {f1} Hz + {f2} Hz + {f3} Hz")
    ascii_plot(clean[:256].tolist(), height=7, width=60)
    print()
    print(f"    Noisy signal (SNR = {snr_before:.1f} dB):")
    ascii_plot(noisy[:256].tolist(), height=7, width=60)
    print()

    # FFT
    X = np.fft.fft(noisy)
    freqs = np.fft.fftfreq(N, 1/fs)
    magnitudes = np.abs(X) / N

    # Find peaks: anything > 4× the median magnitude
    median_mag = np.median(magnitudes[:N//2])
    threshold = 4 * median_mag

    # Create spectral mask
    mask = np.zeros(N)
    for i in range(N):
        if magnitudes[i] > threshold:
            # Keep this frequency and neighbors (smooth transition)
            for j in range(max(0, i-2), min(N, i+3)):
                mask[j] = 1.0

    # Apply mask and inverse FFT
    X_filtered = X * mask
    filtered = np.real(np.fft.ifft(X_filtered))

    snr_after = 10 * np.log10(
        np.mean(clean**2) / (np.mean((clean - filtered)**2) + 1e-30))

    print(f"    Filtered signal (SNR = {snr_after:.1f} dB):")
    ascii_plot(filtered[:256].tolist(), height=7, width=60)
    print()

    # Show the spectrum
    pos = freqs >= 0
    pos_freqs = freqs[pos].tolist()
    pos_mags_noisy = (magnitudes[pos] * 2).tolist()
    pos_mags_filtered = (np.abs(X_filtered[pos]) / N * 2).tolist()

    print("    Frequency spectrum — noisy:")
    ascii_spectrum(pos_freqs, pos_mags_noisy, width=60, height=9, max_freq=250)
    print()
    print("    Frequency spectrum — filtered (only signal peaks kept):")
    ascii_spectrum(pos_freqs, pos_mags_filtered, width=60, height=9, max_freq=250)

    # Detected frequencies
    detected = [(f, m) for f, m in zip(freqs[:N//2], magnitudes[:N//2])
                if m > threshold]
    detected.sort(key=lambda x: -x[1])

    print(f"\n    Detected signal components:")
    for f, m in detected[:6]:
        print(f"      {f:>6.0f} Hz  (magnitude {m:.3f})")

    print(f"""
    SNR improvement: {snr_before:.1f} dB → {snr_after:.1f} dB (+{snr_after - snr_before:.1f} dB)

    ► This is spectral gating — used in audio noise reduction
      software (like noise removal in Audacity).
    ► The same principle (filter in frequency domain, convert back)
      is used in noise-canceling headphones, radio receivers,
      and medical signal processing.
    """)


# =============================================================================
#  APPLICATION 4: TEMPLATE MATCHING VIA FFT CORRELATION
#  Find a pattern in a signal using fast cross-correlation
# =============================================================================

def app4_template_matching():
    print(f"\n{SEPARATOR}")
    print("  APPLICATION 4: FFT-BASED TEMPLATE MATCHING")
    print("  Finding patterns fast via cross-correlation")
    print(SEPARATOR)

    print("""
    Cross-correlation measures how similar a small template is to
    every position in a longer signal. Naively it's O(N·M).
    Via the convolution theorem, it's O(N log N).

    Applications:
      • Radar: match reflected pulse to find target distance
      • Audio: Shazam-style song matching
      • DNA: find gene sequences in a genome
      • Image: find a template in a larger image
    """)

    np.random.seed(42)
    N = 8192
    M = 128  # template length

    # Create a "haystack" signal with a hidden template
    haystack = np.random.randn(N) * 0.3  # background noise

    # Template: a distinctive pattern
    t_template = np.linspace(0, 1, M)
    template = (np.sin(2 * np.pi * 3 * t_template) *
                np.exp(-3 * (t_template - 0.5)**2))  # windowed sine burst

    # Hide template at a specific position
    hidden_pos = 5432
    haystack[hidden_pos:hidden_pos + M] += template * 2

    print(f"    Haystack: {N:,} samples of noise with hidden template at position {hidden_pos}")
    print(f"    Template: {M} samples (windowed sine burst)")
    print()

    print("    Template:")
    ascii_plot(template.tolist(), height=7, width=60)
    print()

    # Method 1: Naive correlation (O(N·M))
    start = time.perf_counter()
    naive_corr = np.correlate(haystack, template, mode='full')
    t_naive = time.perf_counter() - start

    # Method 2: FFT correlation (O(N log N))
    start = time.perf_counter()
    # Pad template to match haystack length
    padded_template = np.zeros(N)
    padded_template[:M] = template[::-1]  # time-reversed for correlation

    H = np.fft.fft(haystack)
    T = np.fft.fft(padded_template)
    fft_corr = np.real(np.fft.ifft(H * T))
    t_fft = time.perf_counter() - start

    # Find the peak
    peak_pos_naive = np.argmax(np.abs(naive_corr)) - M + 1
    peak_pos_fft = np.argmax(np.abs(fft_corr))

    # Adjust FFT peak position (circular correlation offset)
    if peak_pos_fft > N // 2:
        peak_pos_fft = peak_pos_fft - N

    speedup = t_naive / t_fft if t_fft > 0 else float('inf')

    print(f"    Naive O(N·M): {t_naive*1000:.1f}ms  → found at position {peak_pos_naive}")
    print(f"    FFT O(NlogN): {t_fft*1000:.1f}ms  → found at position {peak_pos_fft}")
    print(f"    Speedup:      {speedup:.0f}×")
    print(f"    True position: {hidden_pos}")
    print()

    # Show the correlation peak
    region_start = max(0, hidden_pos - 200)
    region_end = min(N, hidden_pos + 200)
    corr_region = np.abs(fft_corr[region_start:region_end])

    print("    FFT correlation near the hidden template:")
    ascii_plot(corr_region.tolist(), height=9, width=60,
               label="Sharp peak = template location")

    print(f"""
    ► The FFT correlation found the template in {t_fft*1000:.1f}ms
      vs {t_naive*1000:.1f}ms for naive search — a {speedup:.0f}× speedup.
    ► For longer signals (millions of samples), the speedup
      grows to 1000× or more.
    ► This is the CONVOLUTION THEOREM in action:
      correlation in time = multiplication in frequency.
    """)


# =============================================================================
#  APPLICATION 5: SPECTROGRAM
#  Time-frequency visualization as ASCII art
# =============================================================================

def app5_spectrogram():
    print(f"\n{SEPARATOR}")
    print("  APPLICATION 5: SPECTROGRAM")
    print("  Visualizing frequencies over time")
    print(SEPARATOR)

    print("""
    A spectrogram is a 2D image where:
      • X-axis = time
      • Y-axis = frequency
      • Brightness = power at that time-frequency point

    It's an STFT magnitude, displayed as an image.
    This is how music visualizers, speech analysis, and
    birdsong identification work.
    """)

    # Create an interesting test signal
    N = 4096
    sr = 1024
    t = np.arange(N) / sr

    # Multi-component signal:
    # - Piano-like: note at 100 Hz for first quarter, then 200 Hz
    # - Plus a chirp sweeping from 50 to 400 Hz
    signal = np.zeros(N)

    # Note 1: 100 Hz, first half
    mask1 = (t < 2.0).astype(float)
    signal += 1.0 * np.sin(2 * np.pi * 100 * t) * mask1

    # Note 2: 200 Hz, second half
    mask2 = (t >= 2.0).astype(float)
    signal += 0.8 * np.sin(2 * np.pi * 200 * t) * mask2

    # Chirp: sweeps 50 → 400 Hz
    chirp_freq = 50 + 350 * t / t[-1]
    chirp_phase = 2 * np.pi * np.cumsum(chirp_freq) / sr
    signal += 0.5 * np.sin(chirp_phase)

    # Add some noise
    np.random.seed(42)
    signal += 0.2 * np.random.randn(N)

    print("    Signal: 100 Hz note → 200 Hz note + chirp (50→400 Hz)")
    print()

    # Compute spectrogram via STFT
    win_size = 256
    hop = 64
    n_windows = (N - win_size) // hop + 1
    n_freq = win_size // 2
    hann = 0.5 * (1 - np.cos(2 * np.pi * np.arange(win_size) / win_size))

    spec = np.zeros((n_freq, n_windows))
    for w in range(n_windows):
        start = w * hop
        segment = signal[start:start + win_size] * hann
        X = np.fft.fft(segment)[:n_freq]
        spec[:, w] = np.abs(X) ** 2  # power spectrum

    # Convert to dB
    spec_db = 10 * np.log10(spec + 1e-10)
    spec_db = np.clip(spec_db, spec_db.max() - 40, spec_db.max())

    # Render as ASCII heatmap
    display_height = 24
    display_width = 64
    chars = " ░▒▓█"

    # Resample to display size
    y_step = max(1, n_freq // display_height)
    x_step = max(1, n_windows // display_width)

    display = np.zeros((display_height, display_width))
    for r in range(display_height):
        for c in range(display_width):
            r0 = r * y_step
            c0 = c * x_step
            block = spec_db[r0:min(r0+y_step, n_freq),
                            c0:min(c0+x_step, n_windows)]
            if block.size > 0:
                display[r, c] = np.mean(block)
            else:
                display[r, c] = 0.0

    # Replace any NaN with 0
    display = np.nan_to_num(display, nan=0.0)

    # Normalize
    dmin, dmax = display.min(), display.max()
    if dmax > dmin:
        display = (display - dmin) / (dmax - dmin)

    # Print (flip vertically: high freq at top)
    display = display[::-1]

    max_freq = sr // 2
    print("    Spectrogram (time → frequency ↑ brightness = power):\n")
    for r in range(display_height):
        if r == 0:
            yl = f"{max_freq:>5} Hz"
        elif r == display_height - 1:
            yl = f"{'0':>5} Hz"
        elif r == display_height // 2:
            yl = f"{max_freq//2:>5} Hz"
        else:
            yl = "       "
        line = ""
        for c in range(display_width):
            idx = int(display[r, c] * (len(chars) - 1))
            idx = max(0, min(len(chars) - 1, idx))
            line += chars[idx]
        print(f"    {yl} ┤{line}")
    print(f"    {'':>7} └{'─' * display_width}")
    print(f"    {'':>7}  t=0s{'':>{display_width//2-5}}t=2s{'':>{display_width//2-5}}t=4s")

    print("""
    You can see:
      • 100 Hz horizontal line (first half) → 200 Hz (second half)
      • Diagonal line: the chirp sweeping from low to high frequency
      • Background speckle: noise

    This is exactly what speech recognition systems see when you
    talk to Siri/Alexa — your voice becomes a spectrogram, and
    a neural network reads the patterns.

    ► Connection to repo 1: The spectrogram is to audio what hex
      dumps are to binary files — a VISUAL REPRESENTATION that
      reveals hidden structure in the data.
    """)


# =============================================================================
#  APPLICATION 6: DIGITAL FILTERS
#  FIR low-pass, high-pass, and band-pass filters
# =============================================================================

def app6_filters():
    print(f"\n{SEPARATOR}")
    print("  APPLICATION 6: DIGITAL FILTERS")
    print("  Sculpting signals in the frequency domain")
    print(SEPARATOR)

    print("""
    A digital filter removes or enhances specific frequencies.
    The three basic types:

      • LOW-PASS:   Keep low frequencies, remove high (smoothing)
      • HIGH-PASS:  Keep high frequencies, remove low (edge detection)
      • BAND-PASS:  Keep a specific range, remove everything else

    We'll build FIR (Finite Impulse Response) filters — the simplest
    kind, created by windowing an ideal frequency response.
    """)

    def design_fir_lowpass(cutoff_hz, sample_rate, num_taps=51):
        """
        Design a low-pass FIR filter using the windowed-sinc method.

        The ideal low-pass filter in the frequency domain is a rectangle.
        Its impulse response (inverse Fourier transform) is a sinc function.
        Multiply by a Hamming window to get a practical finite filter.
        """
        fc = cutoff_hz / sample_rate  # normalized frequency
        n = np.arange(num_taps)
        mid = (num_taps - 1) / 2

        # Sinc function (ideal low-pass impulse response)
        h = np.where(
            np.abs(n - mid) < 1e-10,
            2 * fc,
            np.sin(2 * np.pi * fc * (n - mid)) / (np.pi * (n - mid))
        )

        # Apply Hamming window
        window = 0.54 - 0.46 * np.cos(2 * np.pi * n / (num_taps - 1))
        h = h * window

        # Normalize
        h = h / np.sum(h)
        return h

    def apply_filter(signal, h):
        """Apply FIR filter via convolution."""
        return np.convolve(signal, h, mode='same')

    # Create test signal
    np.random.seed(42)
    N = 1024
    sr = 1024
    t = np.arange(N) / sr

    low_freq = np.sin(2 * np.pi * 10 * t)      # 10 Hz
    mid_freq = np.sin(2 * np.pi * 80 * t)       # 80 Hz
    high_freq = np.sin(2 * np.pi * 200 * t)     # 200 Hz
    signal = low_freq + mid_freq + high_freq

    print("    Test signal: 10 Hz + 80 Hz + 200 Hz\n")
    ascii_plot(signal[:256].tolist(), height=7, width=60, label="Original mix")
    print()

    # Low-pass filter: keep only < 50 Hz
    h_lp = design_fir_lowpass(50, sr, 101)
    filtered_lp = apply_filter(signal, h_lp)

    print("    LOW-PASS filter (cutoff = 50 Hz):")
    ascii_plot(filtered_lp[:256].tolist(), height=7, width=60,
               label="Only 10 Hz remains")
    print()

    # High-pass filter: spectral inversion of low-pass
    h_hp = -design_fir_lowpass(50, sr, 101)
    h_hp[(len(h_hp) - 1) // 2] += 1  # spectral inversion trick
    filtered_hp = apply_filter(signal, h_hp)

    print("    HIGH-PASS filter (cutoff = 50 Hz):")
    ascii_plot(filtered_hp[:256].tolist(), height=7, width=60,
               label="80 Hz + 200 Hz remain")
    print()

    # Band-pass filter: combine low-pass + high-pass
    h_bp_low = design_fir_lowpass(120, sr, 101)   # upper cutoff
    h_bp_high = -design_fir_lowpass(50, sr, 101)   # lower cutoff
    h_bp_high[(len(h_bp_high) - 1) // 2] += 1
    h_bp = np.convolve(h_bp_low, h_bp_high, mode='same')
    h_bp = h_bp / np.max(np.abs(h_bp)) * np.max(np.abs(h_bp_low))
    filtered_bp = apply_filter(signal, h_bp)

    print("    BAND-PASS filter (50–120 Hz):")
    ascii_plot(filtered_bp[:256].tolist(), height=7, width=60,
               label="Only 80 Hz remains")
    print()

    # Show frequency responses
    print("  ── Filter frequency responses ──\n")

    freqs_resp = np.fft.fftfreq(1024, 1/sr)[:512]
    for name, h in [("Low-pass", h_lp), ("High-pass", h_hp)]:
        H = np.abs(np.fft.fft(h, 1024))[:512]
        print(f"    {name} response:")
        ascii_spectrum(freqs_resp.tolist(), H.tolist(),
                       width=60, height=7, max_freq=300)
        print()

    print("""
    ► Filters are CONVOLUTION — they use the convolution theorem!
    ► The windowed-sinc method designs filters in frequency domain
      (rectangle) then transforms to time domain (sinc × window).
    ► Low-pass → smoothing, anti-aliasing
    ► High-pass → edge detection, DC removal
    ► Band-pass → radio tuning, voice isolation, EQ

    ► Connection to repo 1: The anti-aliasing filter in the
      ADC pipeline (signals_and_sampling.py) is exactly the
      low-pass filter we just built.
    """)


# =============================================================================
# SUMMARY
# =============================================================================

def summary():
    print(f"\n{SEPARATOR}")
    print("  SUMMARY: YOUR WAVE PROCESSING TOOLKIT")
    print(SEPARATOR)
    print("""
    ┌────────────────────────────┬──────────────────────────────────┐
    │ APPLICATION                │ KEY TECHNIQUE                    │
    ├────────────────────────────┼──────────────────────────────────┤
    │ 1. JPEG Compression        │ DCT + quantization               │
    │    Photos & video          │ Fourier's real-valued cousin     │
    ├────────────────────────────┼──────────────────────────────────┤
    │ 2. Audio Synthesizer       │ Additive + FM synthesis          │
    │    Sound from math         │ Fourier in reverse               │
    ├────────────────────────────┼──────────────────────────────────┤
    │ 3. Noise Filter            │ Spectral gating via FFT          │
    │    Signal recovery         │ Zero non-signal frequencies      │
    ├────────────────────────────┼──────────────────────────────────┤
    │ 4. Template Matching       │ FFT cross-correlation            │
    │    Fast pattern search     │ Convolution theorem              │
    ├────────────────────────────┼──────────────────────────────────┤
    │ 5. Spectrogram             │ STFT (windowed FFT)              │
    │    Time-freq visualization │ Speech, music, radar analysis    │
    ├────────────────────────────┼──────────────────────────────────┤
    │ 6. Digital Filters         │ Windowed-sinc FIR design         │
    │    Frequency sculpting     │ Convolution in time domain       │
    └────────────────────────────┴──────────────────────────────────┘

    Every one of these traces back to Euler's formula:

        e^(iθ) = cos(θ) + i·sin(θ)

    Fourier showed that all signals are sums of these spinning
    exponentials. The FFT makes decomposition fast. Everything
    above is a consequence.

    ┌──────────────────────────────────────────────────────────┐
    │  Repo 1 built tools from BIT representation:             │
    │    fast math, hash tables, PRNGs, Bloom filters          │
    │                                                          │
    │  This chapter built tools from WAVE representation:      │
    │    compression, synthesis, filtering, correlation         │
    │                                                          │
    │  They COMBINE in real systems:                           │
    │    JPEG = DCT (waves) + Huffman (bits)                   │
    │    WiFi = OFDM (waves) + LDPC (bits) + CRC (bits)       │
    │    MRI  = FFT (waves) + floating point (bits)            │
    └──────────────────────────────────────────────────────────┘

    NEXT: What happens when the waves aren't just signals
    but the fabric of computation itself?
    → quantum_wave_functions.py
    """)


# =============================================================================
# RUN EVERYTHING
# =============================================================================

if __name__ == "__main__":
    print("\n" + "█" * 72)
    print("█  PRACTICAL WAVE APPLICATIONS                                       █")
    print("█  Real tools built from Fourier and wavelet principles              █")
    print("█" * 72)

    app1_jpeg_compression()
    app2_audio_synth()
    app3_noise_filter()
    app4_template_matching()
    app5_spectrogram()
    app6_filters()
    summary()
