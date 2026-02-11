"""
=============================================================================
  WAVELETS AND UNCERTAINTY — Having It Both Ways
  Fourier tells you WHAT frequencies. Wavelets tell you WHEN.
=============================================================================

  The Fourier Transform has a blindspot: it tells you WHAT
  frequencies are in a signal, but not WHEN they occur.

  A music recording has a C note in the first bar and an E note
  in the second bar. The FFT sees both frequencies but can't tell
  you which came first. For that, we need WAVELETS.

  This script explores the uncertainty principle — the fundamental
  limit on simultaneous time-frequency resolution — and shows how
  wavelets provide the best compromise.
"""

import math
import numpy as np

SEPARATOR = "=" * 72
SUBSEP = "─" * 72

# ── Helper: ASCII plot ───────────────────────────────────────────────────────

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
        r_prev, r_curr = (max(0, min(height-1, r_prev)),
                          max(0, min(height-1, r_curr)))
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


def ascii_heatmap(matrix, width=64, height=20, label="",
                  x_labels=None, y_labels=None):
    """
    Render a 2D matrix as an ASCII heatmap.
    Uses density characters: ' ', '░', '▒', '▓', '█'
    """
    rows, cols = matrix.shape
    if rows == 0 or cols == 0:
        return

    # Resample to fit display
    y_step = max(1, rows // height)
    x_step = max(1, cols // width)

    display = np.zeros((min(rows, height), min(cols, width)))
    for r in range(display.shape[0]):
        for c in range(display.shape[1]):
            r0 = r * y_step
            c0 = c * x_step
            display[r, c] = np.mean(matrix[r0:r0+y_step, c0:c0+x_step])

    # Normalize to [0, 1]
    dmin, dmax = display.min(), display.max()
    if dmax > dmin:
        display = (display - dmin) / (dmax - dmin)

    chars = " ░▒▓█"

    for r in range(display.shape[0]):
        if y_labels and r == 0:
            yl = f"{y_labels[1]:>8}"
        elif y_labels and r == display.shape[0] - 1:
            yl = f"{y_labels[0]:>8}"
        else:
            yl = "        "
        line = ""
        for c in range(display.shape[1]):
            idx = int(display[r, c] * (len(chars) - 1))
            idx = max(0, min(len(chars) - 1, idx))
            line += chars[idx]
        print(f"    {yl} ┤{line}")

    print(f"    {'':>8} └{'─' * display.shape[1]}")
    if x_labels:
        print(f"    {'':>8}  {x_labels[0]:<{display.shape[1]//2}}"
              f"{x_labels[1]:>{display.shape[1]//2}}")
    if label:
        print(f"    {'':>8}  {label}")


# =============================================================================
# PART 1: THE UNCERTAINTY PRINCIPLE
# =============================================================================

def part1_uncertainty():
    print(f"\n{SEPARATOR}")
    print("  PART 1: THE UNCERTAINTY PRINCIPLE")
    print("  You cannot know both time and frequency perfectly")
    print(SEPARATOR)

    print("""
    In quantum mechanics, Heisenberg's uncertainty principle says:

        Δx · Δp ≥ ℏ/2

    You can't know a particle's position and momentum simultaneously
    with arbitrary precision. Measure one precisely → the other
    becomes uncertain.

    Signal processing has its OWN uncertainty principle:

    ┌──────────────────────────────────────────────────────────┐
    │  GABOR UNCERTAINTY PRINCIPLE (1946)                       │
    │                                                          │
    │      Δt · Δf ≥ 1/(4π)                                   │
    │                                                          │
    │  You cannot localize a signal in BOTH time AND frequency │
    │  simultaneously. Better time resolution = worse frequency│
    │  resolution, and vice versa.                             │
    └──────────────────────────────────────────────────────────┘

    This isn't a limitation of our tools — it's a mathematical
    FACT about signals. Let's see why:
    """)

    N = 1024
    t = np.linspace(0, 1, N)

    # Case 1: Pure tone — perfect frequency, no time localization
    pure_tone = np.sin(2 * np.pi * 10 * t)
    X1 = np.abs(np.fft.fft(pure_tone))[:N//2]

    print("    CASE 1: Pure sine wave (10 Hz)")
    print("    → Perfect frequency knowledge, ZERO time localization")
    print()
    print("    Time domain:")
    ascii_plot(pure_tone.tolist(), height=7, width=60)
    print("    Frequency domain:")
    freqs = np.fft.fftfreq(N, 1/N)[:N//2]
    ascii_plot(X1[:50].tolist(), height=7, width=60, label="Sharp peak at 10 Hz")
    print()

    # Case 2: Impulse — perfect time, no frequency localization
    impulse = np.zeros(N)
    impulse[N // 2] = 1.0
    X2 = np.abs(np.fft.fft(impulse))[:N//2]

    print("    CASE 2: Single impulse (click)")
    print("    → Perfect time knowledge, ALL frequencies present")
    print()
    print("    Time domain:")
    ascii_plot(impulse.tolist(), height=7, width=60)
    print("    Frequency domain:")
    ascii_plot(X2[:100].tolist(), height=7, width=60, label="FLAT — all frequencies!")
    print()

    # Case 3: Short burst — some of each
    burst = np.zeros(N)
    center = N // 2
    width_samples = 50
    burst[center - width_samples : center + width_samples] = \
        np.sin(2 * np.pi * 10 * t[:2*width_samples])
    X3 = np.abs(np.fft.fft(burst))[:N//2]

    print("    CASE 3: Short burst of 10 Hz (finite duration)")
    print("    → Some time knowledge, some frequency knowledge")
    print()
    print("    Time domain:")
    ascii_plot(burst.tolist(), height=7, width=60)
    print("    Frequency domain:")
    ascii_plot(X3[:50].tolist(), height=7, width=60,
               label="Broadened peak — frequency is SPREAD")

    print("""
    The pattern is clear:

      Narrow in time  → wide in frequency  (impulse → flat spectrum)
      Narrow in freq  → wide in time       (pure tone → infinite)
      In between      → in between         (burst → broadened peak)

    You CAN'T have both narrow. This is fundamental. It's the same
    math as Heisenberg's principle, just applied to signals.

    ► This is why the standard FFT gives you an ALL-or-NOTHING
      choice: perfect frequency resolution, zero time resolution.
      To get time information, we need something different.
    """)


# =============================================================================
# PART 2: SHORT-TIME FOURIER TRANSFORM (STFT)
# =============================================================================

def part2_stft():
    print(f"\n{SEPARATOR}")
    print("  PART 2: THE SHORT-TIME FOURIER TRANSFORM (STFT)")
    print("  The windowed compromise")
    print(SEPARATOR)

    print("""
    The simplest fix: chop the signal into short windows and
    FFT each window separately. This is the STFT.

        STFT(t, f) = FFT of [signal × window centered at time t]

    Each window gives you the frequencies present at THAT time.
    Slide the window across the signal → time-frequency map.
    """)

    # Create a chirp signal: frequency increases over time
    N = 2048
    t = np.linspace(0, 2, N)
    # Chirp: linearly increasing frequency from 5 to 50 Hz
    freq_t = 5 + 22.5 * t  # instantaneous frequency
    phase = 2 * np.pi * np.cumsum(freq_t) / N * 2
    chirp = np.sin(phase)

    print("    Signal: a CHIRP (frequency sweeps from 5 to 50 Hz)")
    ascii_plot(chirp[:512].tolist(), height=9, width=60,
               label="First quarter — low frequency")
    ascii_plot(chirp[1536:].tolist(), height=9, width=60,
               label="Last quarter — high frequency")
    print()

    # Compute STFT
    window_size = 128
    hop = 32
    n_windows = (N - window_size) // hop + 1
    n_freq_bins = window_size // 2

    stft_mag = np.zeros((n_freq_bins, n_windows))
    hann = 0.5 * (1 - np.cos(2 * np.pi * np.arange(window_size) / window_size))

    for w in range(n_windows):
        start = w * hop
        segment = chirp[start:start + window_size] * hann
        X = np.fft.fft(segment)[:n_freq_bins]
        stft_mag[:, w] = np.abs(X)

    # Show as heatmap (flip vertically so low freq at bottom)
    stft_display = stft_mag[::-1]  # flip so high freq at top

    print("    STFT spectrogram (time → , frequency ↑):")
    print("    Brightness = energy at that time-frequency point\n")
    ascii_heatmap(stft_display, width=60, height=20,
                  x_labels=["t=0", "t=2s"],
                  y_labels=["0 Hz", f"{N//2} Hz"])
    print()

    print("""
    The diagonal line shows the chirp: frequency increases over time!
    The FFT alone would show a FLAT spectrum (all frequencies present).
    The STFT reveals the TIME structure.

    But the STFT has a fundamental limitation — the uncertainty
    principle strikes again:

      • SHORT window → good time resolution, poor frequency resolution
      • LONG window  → good frequency resolution, poor time resolution

    The window size is FIXED. You pick it and live with the tradeoff.

    Wavelets solve this by using VARIABLE-SIZE windows.
    """)


# =============================================================================
# PART 3: THE HAAR WAVELET
# =============================================================================

def part3_haar():
    print(f"\n{SEPARATOR}")
    print("  PART 3: THE HAAR WAVELET — THE SIMPLEST WAVELET")
    print("  Addition and subtraction at different scales")
    print(SEPARATOR)

    print("""
    The Haar wavelet (1909) is the simplest wavelet:

        ψ(t) = { +1  for 0 ≤ t < 0.5
               { -1  for 0.5 ≤ t < 1
               {  0  otherwise

    It's literally: take the DIFFERENCE between the left half
    and right half of a signal segment.
    """)

    # Show the Haar wavelet
    N = 100
    haar = np.zeros(N)
    haar[:N//2] = 1
    haar[N//2:] = -1
    ascii_plot(haar.tolist(), height=9, width=40, label="Haar wavelet ψ(t)")
    print()

    print("""
    The Haar wavelet transform works like this:

    Given a signal [a, b, c, d], compute:

      Averages (approximation):  [(a+b)/2, (c+d)/2]
      Differences (detail):      [(a-b)/2, (c-d)/2]

    That's it. Average = low-frequency content. Difference = detail.
    """)

    def haar_transform_1level(signal):
        """
        One level of Haar wavelet transform.

        Produces two arrays of half the length:
          approx: pairwise averages (low-frequency content)
          detail: pairwise differences (high-frequency content)

        This is the SAME operation as the Haar wavelet filter bank.
        """
        n = len(signal)
        approx = [(signal[i] + signal[i + 1]) / 2 for i in range(0, n, 2)]
        detail = [(signal[i] - signal[i + 1]) / 2 for i in range(0, n, 2)]
        return approx, detail

    def haar_inverse_1level(approx, detail):
        """Reconstruct signal from approximation and detail."""
        signal = []
        for a, d in zip(approx, detail):
            signal.append(a + d)  # original even sample
            signal.append(a - d)  # original odd sample
        return signal

    # Demonstrate on a simple signal
    test = [4, 6, 10, 14, 12, 8, 6, 2]
    approx, detail = haar_transform_1level(test)

    print(f"    Input signal:    {test}")
    print(f"    Approximation:   {approx}    (averages)")
    print(f"    Detail:          {detail}    (differences)")
    print()

    # Verify perfect reconstruction
    reconstructed = haar_inverse_1level(approx, detail)
    print(f"    Reconstructed:   {reconstructed}")
    print(f"    Match: {'✓ Perfect' if reconstructed == test else '✗ Error!'}")

    print("""
    ┌──────────────────────────────────────────────────────────┐
    │  CONNECTION TO REPO 1                                    │
    │                                                          │
    │  The Haar transform is literally:                        │
    │    Average = (a + b) >> 1     (addition + right shift)   │
    │    Detail  = (a - b) >> 1     (subtraction + right shift)│
    │                                                          │
    │  These are BIT-LEVEL OPERATIONS from bit_tricks_demo.py! │
    │  The simplest wavelet is built from the simplest bit     │
    │  manipulation. Discrete meets continuous.                │
    └──────────────────────────────────────────────────────────┘
    """)

    # Apply to a more interesting signal
    print("  ── Haar transform of a step signal ──\n")
    N = 64
    t = np.arange(N)
    step_signal = np.zeros(N)
    step_signal[:N//4] = 1.0
    step_signal[N//4:N//2] = 3.0
    step_signal[N//2:3*N//4] = 2.0
    step_signal[3*N//4:] = 0.5

    print("    Original signal:")
    ascii_plot(step_signal.tolist(), height=9, width=60)
    print()

    approx, detail = haar_transform_1level(step_signal.tolist())
    print("    Approximation (smooth trend):")
    ascii_plot(approx, height=7, width=60)
    print("    Detail (where changes happen):")
    ascii_plot(detail, height=7, width=60)

    print("""
    Notice: the detail coefficients are zero EXCEPT at the
    transitions! The Haar wavelet DETECTS EDGES — it tells
    you exactly WHERE the signal changes.

    ► This is why wavelets are used for edge detection in
      image processing and for detecting transients (clicks,
      drum hits) in audio.
    """)


# =============================================================================
# PART 4: MULTIRESOLUTION ANALYSIS
# =============================================================================

def part4_multiresolution():
    print(f"\n{SEPARATOR}")
    print("  PART 4: MULTIRESOLUTION ANALYSIS")
    print("  Decompose a signal at multiple scales simultaneously")
    print(SEPARATOR)

    print("""
    The Haar transform gives us ONE level of approximation/detail.
    But we can RECURSIVELY decompose the approximation:

    Level 0:  [original signal]                  (N samples)
    Level 1:  [approximation₁] + [detail₁]      (N/2 each)
    Level 2:  [approximation₂] + [detail₂] + [detail₁]
    Level 3:  [approximation₃] + [detail₃] + [detail₂] + [detail₁]
    ...

    Each level captures detail at a different SCALE:
      • detail₁:  finest details (highest frequencies)
      • detail₂:  medium details
      • detail₃:  coarse details
      • approx:   overall trend (lowest frequencies)

    This is MULTIRESOLUTION ANALYSIS (MRA).
    """)

    def haar_full_decompose(signal, max_levels=None):
        """
        Full Haar wavelet decomposition.

        Returns: (final_approx, [detail_at_each_level])
        Levels go from coarsest to finest.
        """
        current = list(signal)
        details = []
        levels = 0
        max_l = max_levels or int(math.log2(len(signal)))

        while len(current) > 1 and levels < max_l:
            approx = []
            detail = []
            for i in range(0, len(current), 2):
                if i + 1 < len(current):
                    approx.append((current[i] + current[i + 1]) / 2)
                    detail.append((current[i] - current[i + 1]) / 2)
                else:
                    approx.append(current[i])
            details.append(detail)
            current = approx
            levels += 1

        return current, details

    def haar_full_reconstruct(approx, details):
        """Reconstruct from full wavelet decomposition."""
        current = list(approx)
        for detail in reversed(details):
            reconstructed = []
            for a, d in zip(current, detail):
                reconstructed.append(a + d)
                reconstructed.append(a - d)
            current = reconstructed
        return current

    # Create a test signal: smooth trend + medium oscillation + fine noise
    np.random.seed(42)
    N = 256
    t = np.linspace(0, 1, N)

    smooth = 2 * np.sin(2 * np.pi * 2 * t)         # 2 Hz (smooth trend)
    medium = 0.8 * np.sin(2 * np.pi * 15 * t)       # 15 Hz (medium detail)
    fine = 0.3 * np.sin(2 * np.pi * 60 * t)         # 60 Hz (fine detail)
    signal = smooth + medium + fine

    print("    Test signal: 2 Hz + 15 Hz + 60 Hz\n")
    ascii_plot(signal.tolist(), height=9, width=60,
               label="Composite signal")
    print()

    # Decompose
    approx, details = haar_full_decompose(signal.tolist(), max_levels=5)

    print(f"    Decomposition: {len(details)} levels\n")
    print(f"    Approximation (coarsest trend, {len(approx)} coefficients):")
    ascii_plot(approx, height=7, width=60)
    print()

    for level, detail in enumerate(details):
        freq_band = f"~{2**(level)} - {2**(level+1)} Hz range"
        print(f"    Detail level {level + 1} ({len(detail)} coefficients, {freq_band}):")
        ascii_plot(detail, height=5, width=60)
        print()

    # Verify perfect reconstruction
    reconstructed = haar_full_reconstruct(approx, details)
    error = max(abs(signal[i] - reconstructed[i]) for i in range(N))
    print(f"    Reconstruction error: {error:.2e}")
    print(f"    Perfect reconstruction: {'✓' if error < 1e-10 else '✗'}")

    print("""
    ┌──────────────────────────────────────────────────────────┐
    │  CONNECTION TO REPO 1: MORTON CODES                      │
    │                                                          │
    │  In repo 1, Morton codes gave us SPATIAL hierarchy:      │
    │  bit interleaving organized 2D points at multiple        │
    │  resolutions for fast nearest-neighbor search.           │
    │                                                          │
    │  Wavelets give us FREQUENCY hierarchy: decomposition at  │
    │  multiple scales for time-frequency analysis.            │
    │                                                          │
    │  Both are MULTIRESOLUTION DATA STRUCTURES built from     │
    │  the same principle: divide, organize, conquer.          │
    └──────────────────────────────────────────────────────────┘
    """)


# =============================================================================
# PART 5: WAVELET DENOISING
# =============================================================================

def part5_denoising():
    print(f"\n{SEPARATOR}")
    print("  PART 5: WAVELET DENOISING")
    print("  Cleaning signals by thresholding wavelet coefficients")
    print(SEPARATOR)

    print("""
    The killer application of wavelets: DENOISING.

    The idea is beautifully simple:
    1. Decompose the signal into wavelet coefficients
    2. Signal energy concentrates in a FEW large coefficients
    3. Noise energy spreads across MANY small coefficients
    4. Zero out the small coefficients (thresholding)
    5. Reconstruct → noise is gone, signal is preserved

    This works because signals have STRUCTURE (concentrated energy)
    and noise doesn't (spread energy).
    """)

    def haar_decompose(signal, max_levels=None):
        current = list(signal)
        details = []
        max_l = max_levels or int(math.log2(len(signal)))
        while len(current) > 1 and len(details) < max_l:
            approx, detail = [], []
            for i in range(0, len(current), 2):
                if i + 1 < len(current):
                    approx.append((current[i] + current[i + 1]) / 2)
                    detail.append((current[i] - current[i + 1]) / 2)
                else:
                    approx.append(current[i])
            details.append(detail)
            current = approx
        return current, details

    def haar_reconstruct(approx, details):
        current = list(approx)
        for detail in reversed(details):
            reconstructed = []
            for a, d in zip(current, detail):
                reconstructed.append(a + d)
                reconstructed.append(a - d)
            current = reconstructed
        return current

    def soft_threshold(coefficients, threshold):
        """
        Soft thresholding: coefficients smaller than threshold → 0.
        Larger coefficients are shrunk toward zero by 'threshold'.
        This avoids introducing discontinuities.
        """
        return [max(0, abs(c) - threshold) * (1 if c >= 0 else -1)
                for c in coefficients]

    # Create noisy signal
    np.random.seed(42)
    N = 512
    t = np.linspace(0, 1, N)
    clean = (np.sin(2 * np.pi * 5 * t) +
             0.5 * np.sin(2 * np.pi * 15 * t) +
             0.3 * np.sin(2 * np.pi * 25 * t))
    noise = np.random.randn(N) * 0.8
    noisy = clean + noise

    print("    Clean signal:")
    ascii_plot(clean.tolist(), height=9, width=60)
    print()
    print("    Noisy signal (SNR ≈ 4 dB):")
    ascii_plot(noisy.tolist(), height=9, width=60)
    print()

    # Wavelet denoise
    approx, details = haar_decompose(noisy.tolist(), max_levels=6)

    # Estimate noise level (median absolute deviation of finest detail)
    finest_detail = details[0]
    mad = sorted(abs(d) for d in finest_detail)[len(finest_detail) // 2]
    sigma_est = mad / 0.6745  # standard estimator
    threshold = sigma_est * math.sqrt(2 * math.log(N))

    print(f"    Estimated noise σ: {sigma_est:.3f}")
    print(f"    Threshold:         {threshold:.3f}")
    print()

    # Count coefficients before and after
    total_coeffs = sum(len(d) for d in details)
    nonzero_before = sum(1 for d in details for c in d if abs(c) > 1e-10)

    # Apply threshold to detail coefficients
    details_denoised = [soft_threshold(d, threshold) for d in details]

    nonzero_after = sum(1 for d in details_denoised for c in d if abs(c) > 1e-10)

    # Reconstruct
    denoised = haar_reconstruct(approx, details_denoised)

    print("    Denoised signal:")
    ascii_plot(denoised[:N], height=9, width=60)
    print()

    # Measure quality
    mse_noisy = np.mean((noisy - clean) ** 2)
    mse_denoised = np.mean((np.array(denoised[:N]) - clean) ** 2)
    snr_noisy = 10 * np.log10(np.mean(clean**2) / mse_noisy)
    snr_denoised = 10 * np.log10(np.mean(clean**2) / mse_denoised)

    print(f"    Results:")
    print(f"      Noisy SNR:       {snr_noisy:.1f} dB")
    print(f"      Denoised SNR:    {snr_denoised:.1f} dB")
    print(f"      Improvement:     {snr_denoised - snr_noisy:.1f} dB")
    print(f"      Coefficients:    {nonzero_before} → {nonzero_after} "
          f"({nonzero_after/nonzero_before*100:.0f}% kept)")

    print("""
    The wavelet denoiser improved SNR significantly by
    throwing away the small (noise-dominated) coefficients
    and keeping the large (signal-dominated) ones.

    ► Connection to repo 1: This is the same principle as
      Huffman coding — frequent/important data gets more
      representation, rare/noise data gets less. Both exploit
      the NON-UNIFORM distribution of information.
    """)


# =============================================================================
# PART 6: WAVELET COMPRESSION
# =============================================================================

def part6_compression():
    print(f"\n{SEPARATOR}")
    print("  PART 6: WAVELET COMPRESSION")
    print("  Keep the big coefficients, discard the rest")
    print(SEPARATOR)

    print("""
    Denoising and compression are two sides of the same coin:

      Denoising:    discard coefficients that are likely noise
      Compression:  discard coefficients that contribute least

    In wavelet compression:
    1. Transform the signal
    2. Sort coefficients by magnitude
    3. Keep only the largest K coefficients
    4. Reconstruct from K coefficients instead of N
    5. Compression ratio = N / K
    """)

    def haar_decompose(signal, max_levels=None):
        current = list(signal)
        details = []
        max_l = max_levels or int(math.log2(len(signal)))
        while len(current) > 1 and len(details) < max_l:
            approx, detail = [], []
            for i in range(0, len(current), 2):
                if i + 1 < len(current):
                    approx.append((current[i] + current[i + 1]) / 2)
                    detail.append((current[i] - current[i + 1]) / 2)
                else:
                    approx.append(current[i])
            details.append(detail)
            current = approx
        return current, details

    def haar_reconstruct(approx, details):
        current = list(approx)
        for detail in reversed(details):
            reconstructed = []
            for a, d in zip(current, detail):
                reconstructed.append(a + d)
                reconstructed.append(a - d)
            current = reconstructed
        return current

    # Test signal
    np.random.seed(42)
    N = 512
    t = np.linspace(0, 1, N)
    signal = (2.0 * np.sin(2 * np.pi * 3 * t) +
              1.0 * np.sin(2 * np.pi * 10 * t) +
              0.5 * np.sin(2 * np.pi * 25 * t) +
              0.2 * np.sin(2 * np.pi * 50 * t))

    print("    Original signal (4 frequency components):")
    ascii_plot(signal.tolist(), height=9, width=60)
    print()

    # Decompose
    approx, details = haar_decompose(signal.tolist(), max_levels=7)

    # Collect all coefficients
    all_coeffs = list(approx)
    for d in details:
        all_coeffs.extend(d)

    total = len(all_coeffs)

    # Compress at different ratios
    print(f"    {'Keep %':>8} {'Coefficients':>14} {'RMSE':>10} "
          f"{'SNR (dB)':>10} {'Quality':>10}")
    print(f"    {'─'*8} {'─'*14} {'─'*10} {'─'*10} {'─'*10}")

    keep_fractions = [1.0, 0.5, 0.25, 0.1, 0.05, 0.02]

    for frac in keep_fractions:
        k = max(1, int(total * frac))

        # Find threshold: keep k largest coefficients
        sorted_mags = sorted(abs(c) for c in all_coeffs)
        if k < total:
            thresh = sorted_mags[total - k]
        else:
            thresh = 0

        # Apply threshold
        approx_c = [c if abs(c) >= thresh else 0 for c in approx]
        details_c = [[c if abs(c) >= thresh else 0 for c in d] for d in details]

        # Reconstruct
        reconstructed = haar_reconstruct(approx_c, details_c)

        rmse = np.sqrt(np.mean((signal - np.array(reconstructed[:N])) ** 2))
        if rmse > 0:
            snr = 10 * np.log10(np.mean(signal**2) / rmse**2)
        else:
            snr = float('inf')

        quality = ("Perfect" if rmse < 1e-10 else
                   "Excellent" if snr > 40 else
                   "Good" if snr > 25 else
                   "Fair" if snr > 15 else
                   "Poor")

        print(f"    {frac*100:>7.0f}% {k:>14} {rmse:>10.4f} "
              f"{snr:>10.1f} {quality:>10}")

    # Show the 10% compression visually
    k = max(1, int(total * 0.10))
    sorted_mags = sorted(abs(c) for c in all_coeffs)
    thresh = sorted_mags[total - k]
    approx_c = [c if abs(c) >= thresh else 0 for c in approx]
    details_c = [[c if abs(c) >= thresh else 0 for c in d] for d in details]
    reconstructed = haar_reconstruct(approx_c, details_c)

    print(f"\n    Reconstructed from only 10% of coefficients:")
    ascii_plot(reconstructed[:N], height=9, width=60)

    print("""
    With just 10% of coefficients, we get excellent reconstruction!
    This is the principle behind JPEG 2000, which uses wavelets
    (specifically the CDF 9/7 wavelet) for compression.

    ┌──────────────────────────────────────────────────────────┐
    │  WHY WAVELETS COMPRESS WELL                              │
    │                                                          │
    │  Most natural signals have energy concentrated at a few  │
    │  scales. Wavelets capture this sparsity:                 │
    │                                                          │
    │  • Smooth regions → large approx coefficients            │
    │  • Edges/transients → large detail coefficients          │
    │  • Everything else → small (discardable) coefficients    │
    │                                                          │
    │  Fourier can't do this as well because a sharp edge      │
    │  requires MANY Fourier coefficients (Gibbs phenomenon),  │
    │  but only a FEW wavelet coefficients.                    │
    └──────────────────────────────────────────────────────────┘
    """)


# =============================================================================
# PART 7: THE PHILOSOPHY
# =============================================================================

def part7_philosophy():
    print(f"\n{SEPARATOR}")
    print("  CONCLUSION: FOURIER vs. WAVELETS")
    print(SEPARATOR)

    print("""
    ┌──────────────────────────────────────────────────────────┐
    │  FOURIER                    │  WAVELETS                   │
    │  ───────                    │  ────────                   │
    │  Fixed basis: sin/cos       │  Variable basis: ψ at scales│
    │  Global: all time at once   │  Local: time AND frequency  │
    │  Great for stationary       │  Great for transient signals│
    │  signals (constant freq)    │  (changing, real-world)     │
    │  Perfect for: radio, audio  │  Perfect for: compression,  │
    │  analysis, spectral ID      │  denoising, edge detection  │
    │                              │                             │
    │  Uncertainty: all frequency, │  Uncertainty: adapts window │
    │  no time                     │  size to frequency          │
    └──────────────────────────────────────────────────────────┘

    The wavelet's secret: it uses SHORT windows for HIGH frequencies
    (good time resolution where it matters) and LONG windows for
    LOW frequencies (good frequency resolution where it matters).

    This ADAPTIVE resolution is what makes wavelets fundamentally
    different from — and often better than — the STFT.

    ┌──────────────────────────────────────────────────────────┐
    │  THE REPRESENTATION HIERARCHY                            │
    │                                                          │
    │  Repo 1:   Bits → bit tricks → hidden structure          │
    │  Script 1: Continuous → sample → discrete                │
    │  Script 2: Time → Fourier → frequency                    │
    │  Script 3: Time+Freq → wavelet → multiresolution         │
    │                                                          │
    │  Each level reveals DIFFERENT structure in the same data. │
    │  Each level enables DIFFERENT applications.              │
    │  Same principle: representation is a tool.               │
    └──────────────────────────────────────────────────────────┘

    NEXT: Now let's build PRACTICAL TOOLS from these ideas.
    → practical_wave_applications.py
    """)


# =============================================================================
# RUN EVERYTHING
# =============================================================================

if __name__ == "__main__":
    print("\n" + "█" * 72)
    print("█  WAVELETS AND UNCERTAINTY                                          █")
    print("█  When Fourier isn't enough: time meets frequency                   █")
    print("█" * 72)

    part1_uncertainty()
    part2_stft()
    part3_haar()
    part4_multiresolution()
    part5_denoising()
    part6_compression()
    part7_philosophy()
