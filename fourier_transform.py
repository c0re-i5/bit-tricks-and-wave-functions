"""
=============================================================================
  THE FOURIER TRANSFORM — Seeing Frequencies
  The most important algorithm most programmers never learned
=============================================================================

  In 1807, Joseph Fourier claimed something outrageous to the
  French Academy of Sciences:

    "Any function whatsoever can be expressed as a sum of
     sines and cosines."

  Lagrange objected. Laplace was skeptical. It took over a century
  to fully formalize. But Fourier was essentially right, and his
  insight became one of the most powerful tools in all of science
  and engineering.

  This script builds the Fourier Transform from first principles:
  Euler's formula → DFT → FFT → practical spectral analysis.

  The FFT is to signal processing what Carmack's hack is to
  game engines: it reads the same data differently and finds
  hidden structure that makes the impossible tractable.
"""

import math
import cmath
import time
import numpy as np

SEPARATOR = "=" * 72
SUBSEP = "─" * 72

# ── Helper: ASCII bar chart for spectra ──────────────────────────────────────

def ascii_spectrum(frequencies, magnitudes, width=60, height=15, label="",
                   max_freq=None):
    """
    Render a frequency spectrum as an ASCII bar chart.
    Frequencies on x-axis, magnitude on y-axis.
    """
    if max_freq:
        mask = [i for i, f in enumerate(frequencies) if f <= max_freq]
        frequencies = [frequencies[i] for i in mask]
        magnitudes = [magnitudes[i] for i in mask]

    if not magnitudes:
        return

    max_mag = max(magnitudes) if magnitudes else 1
    if max_mag == 0:
        max_mag = 1

    # Resample to fit width
    n = len(magnitudes)
    step = max(1, n // width)
    display_mags = []
    display_freqs = []
    for i in range(0, min(n, width * step), step):
        chunk = magnitudes[i:i + step]
        display_mags.append(max(chunk))
        display_freqs.append(frequencies[i])

    w = len(display_mags)

    # Build grid
    for row in range(height):
        threshold = max_mag * (height - row) / height
        if row == 0:
            axis_label = f"{max_mag:>8.1f}"
        elif row == height - 1:
            axis_label = f"{'0':>8}"
        elif row == height // 2:
            axis_label = f"{max_mag/2:>8.1f}"
        else:
            axis_label = "        "
        line = ""
        for c in range(w):
            if display_mags[c] >= threshold:
                line += "█"
            else:
                line += " "
        print(f"    {axis_label} ┤{line}")

    print(f"    {'':>8} └{'─' * w}")

    # Frequency axis labels
    if display_freqs:
        first_f = display_freqs[0]
        mid_f = display_freqs[w // 2] if w > 1 else first_f
        last_f = display_freqs[-1]
        label_line = f"{first_f:.0f} Hz"
        pad = w // 2 - len(label_line)
        label_line += " " * max(1, pad) + f"{mid_f:.0f} Hz"
        pad2 = w - len(label_line) - len(f"{last_f:.0f} Hz")
        label_line += " " * max(1, pad2) + f"{last_f:.0f} Hz"
        print(f"    {'':>8}  {label_line}")

    if label:
        print(f"    {'':>8}  {label}")


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

    # Zero line
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
        r_prev = max(0, min(height - 1, r_prev))
        r_curr = max(0, min(height - 1, r_curr))
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


# =============================================================================
# PART 1: EULER'S FORMULA — THE ROSETTA STONE
# =============================================================================

def part1_euler():
    print(f"\n{SEPARATOR}")
    print("  PART 1: EULER'S FORMULA — THE ROSETTA STONE")
    print("  Where exponentials become waves")
    print(SEPARATOR)

    print("""
    Before we can understand the Fourier Transform, we need the
    single most important formula in this entire repo:

    ┌──────────────────────────────────────────────────────────┐
    │  EULER'S FORMULA                                         │
    │                                                          │
    │      e^(iθ) = cos(θ) + i·sin(θ)                         │
    │                                                          │
    │  An exponential IS a wave. A wave IS an exponential.     │
    │  They're the same thing, written in different notation.  │
    └──────────────────────────────────────────────────────────┘

    Think of e^(iθ) as an arrow spinning around the origin in the
    complex plane. As θ increases, the arrow rotates:
    """)

    # Visualize the unit circle with complex exponential
    print("    The unit circle: e^(iθ) for θ = 0 to 2π\n")

    n_points = 24
    circle_size = 9  # radius in characters (odd for center)
    grid = [[" " for _ in range(2 * circle_size + 3)]
            for _ in range(circle_size + 1)]
    cx, cy = circle_size + 1, circle_size // 2

    for k in range(n_points):
        theta = 2 * math.pi * k / n_points
        x = round(math.cos(theta) * (circle_size - 1)) + cx
        y = round(-math.sin(theta) * (circle_size // 2 - 1)) + cy
        x = max(0, min(len(grid[0]) - 1, x))
        y = max(0, min(len(grid) - 1, y))
        grid[y][x] = "●"

    # Mark axes
    for c in range(len(grid[0])):
        if grid[cy][c] == " ":
            grid[cy][c] = "─"
    for r in range(len(grid)):
        if grid[r][cx] == " ":
            grid[r][cx] = "│"
    grid[cy][cx] = "┼"

    # Labels
    for r in range(len(grid)):
        print(f"    {'':>4}{''.join(grid[r])}")
    print(f"    {'':>4}{'':>{cx-1}}Re")
    print(f"    {'':>4} Im ↑")

    print("""
    Each point ● is e^(iθ) for one value of θ.
    Real part = cos(θ), Imaginary part = sin(θ).

    Now watch: as θ increases uniformly with time (θ = ωt),
    the REAL PART traces out a cosine wave, and the
    IMAGINARY PART traces out a sine wave:
    """)

    t = np.linspace(0, 2, 500)
    freq = 2  # Hz
    z = np.exp(1j * 2 * np.pi * freq * t)

    print("    Real part of e^(i·2π·2·t):  cos(2π·2·t)")
    ascii_plot(z.real.tolist(), height=9, width=60, label="")
    print("    Imaginary part:             sin(2π·2·t)")
    ascii_plot(z.imag.tolist(), height=9, width=60, label="")

    print("""
    ► e^(iθ) packages a cosine AND a sine together into one object.
    ► This is why complex numbers appear everywhere in signal
      processing: they're the natural language for waves.
    ► Connection to repo 1: just as Carmack saw that IEEE 754
      bits are "secretly" logarithms, Euler saw that exponentials
      are "secretly" waves.
    """)


# =============================================================================
# PART 2: THE DFT — BY HAND
# =============================================================================

def part2_dft():
    print(f"\n{SEPARATOR}")
    print("  PART 2: THE DFT — ANALYZING FREQUENCIES BY HAND")
    print("  What's really happening inside a Fourier Transform")
    print(SEPARATOR)

    print("""
    The Discrete Fourier Transform (DFT) takes N time-domain
    samples and produces N frequency-domain coefficients:

    ┌──────────────────────────────────────────────────────────┐
    │  DFT FORMULA                                             │
    │                                                          │
    │       N-1                                                │
    │  X[k] = Σ  x[n] · e^(-i·2π·k·n/N)                      │
    │       n=0                                                │
    │                                                          │
    │  x[n] = time sample at position n                        │
    │  X[k] = frequency component at frequency k               │
    │  N    = total number of samples                          │
    └──────────────────────────────────────────────────────────┘

    What is this doing? For each frequency k, it:
    1. Creates a "test wave" at that frequency: e^(-i·2π·k·n/N)
    2. Multiplies the signal by the test wave (correlation)
    3. Sums the result

    If the signal CONTAINS frequency k, the multiplication
    reinforces and produces a large sum. If not, the products
    cancel out to nearly zero.

    Let's implement this from scratch:
    """)

    def dft_naive(x):
        """
        Compute DFT using the direct formula. O(N²).
        Returns complex frequency-domain coefficients.
        """
        N = len(x)
        X = []
        for k in range(N):
            total = 0 + 0j
            for n in range(N):
                angle = -2 * math.pi * k * n / N
                total += x[n] * cmath.exp(1j * angle)
            X.append(total)
        return X

    # Test signal: 5 Hz + 12 Hz
    N = 64
    fs = 64  # sample rate = N for simplicity
    t = np.arange(N) / fs
    signal = np.sin(2 * np.pi * 5 * t) + 0.6 * np.sin(2 * np.pi * 12 * t)

    print("    Test signal: sin(2π·5·t) + 0.6·sin(2π·12·t)")
    ascii_plot(signal.tolist(), height=9, width=60, label="Two frequencies: 5 Hz and 12 Hz")
    print()

    # Compute DFT
    X = dft_naive(signal.tolist())
    magnitudes = [abs(x) / N for x in X]
    frequencies = [k * fs / N for k in range(N)]

    print("    DFT magnitude spectrum (first half — positive frequencies):")
    ascii_spectrum(frequencies[:N//2], magnitudes[:N//2],
                   width=60, height=11, max_freq=fs/2)
    print()

    # Show the peaks
    print("    Detected peaks:")
    peak_threshold = 0.1
    for k in range(N // 2):
        if magnitudes[k] > peak_threshold:
            print(f"      Frequency: {frequencies[k]:>6.1f} Hz    "
                  f"Magnitude: {magnitudes[k]:.3f}")

    print("""
    The DFT correctly identified both 5 Hz and 12 Hz!
    The 12 Hz peak has magnitude ~0.3 (half of 0.6, because the
    energy splits between positive and negative frequencies).

    But there's a problem: this naïve DFT is O(N²).
    For N = 1,000,000 that's a TRILLION operations.
    Audio at 44.1 kHz generates that many samples in 22 seconds.

    We need to go faster.
    """)


# =============================================================================
# PART 3: THE FFT — THE "WHAT THE FUCK?" OF SIGNAL PROCESSING
# =============================================================================

def part3_fft():
    print(f"\n{SEPARATOR}")
    print("  PART 3: THE FFT — THE COOLEY-TUKEY ALGORITHM")
    print("  O(n²) → O(n log n): one of the top 10 algorithms ever")
    print(SEPARATOR)

    print("""
    In 1965, James Cooley and John Tukey published an algorithm that
    reduced the DFT from O(N²) to O(N log N). It changed the world.

    The trick: DIVIDE AND CONQUER on the DFT sum.

    Split the signal into EVEN-indexed and ODD-indexed samples:

        X[k] = Σ x[2m]·W^(2mk) + W^k · Σ x[2m+1]·W^((2m+1)k)
             = DFT_even[k]     + W^k · DFT_odd[k]

    where W = e^(-i·2π/N) is the "twiddle factor" — a root of unity.

    Each half-DFT has size N/2. Apply the same split recursively
    until you reach size-1 DFTs (which are trivial: X[0] = x[0]).

    Cost: T(N) = 2·T(N/2) + O(N) → T(N) = O(N log N)
    """)

    print("""    ┌──────────────────────────────────────────────────────────┐
    │  THE BUTTERFLY OPERATION                                 │
    │                                                          │
    │  At each stage, pairs of values are combined:            │
    │                                                          │
    │     a ──────┬──── a + W^k · b                            │
    │             ╳                                             │
    │     b ──────┘──── a - W^k · b                            │
    │                                                          │
    │  This "butterfly" pattern gives the FFT its structure.   │
    │  The twiddle factor W^k ROTATES the complex number b     │
    │  before adding/subtracting — it's Euler's formula at work│
    └──────────────────────────────────────────────────────────┘
    """)

    def fft_cooley_tukey(x):
        """
        Recursive Cooley-Tukey FFT. Requires len(x) to be a power of 2.

        This is the actual algorithm — no numpy, no libraries.
        Compare with the naïve DFT to see the speedup.
        """
        N = len(x)
        if N <= 1:
            return list(x)

        # Split into even and odd
        even = fft_cooley_tukey(x[0::2])
        odd = fft_cooley_tukey(x[1::2])

        # Combine with twiddle factors
        T = [cmath.exp(-2j * cmath.pi * k / N) * odd[k] for k in range(N // 2)]
        return [even[k] + T[k] for k in range(N // 2)] + \
               [even[k] - T[k] for k in range(N // 2)]

    # Verify correctness against naive DFT
    N = 64
    fs = 64
    t = np.arange(N) / fs
    signal = (np.sin(2 * np.pi * 5 * t) +
              0.6 * np.sin(2 * np.pi * 12 * t)).tolist()

    # Our FFT
    X_fft = fft_cooley_tukey(signal)
    mags_fft = [abs(x) / N for x in X_fft]

    # numpy FFT for comparison
    X_numpy = np.fft.fft(signal)
    mags_numpy = np.abs(X_numpy) / N

    # Check agreement
    max_diff = max(abs(mags_fft[i] - mags_numpy[i]) for i in range(N))
    print(f"    Verification: our FFT vs numpy FFT")
    print(f"    Maximum difference: {max_diff:.2e}")
    print(f"    Match: {'✓ Perfect' if max_diff < 1e-10 else '✗ Error!'}")
    print()

    # Benchmark: O(N²) vs O(N log N)
    print("  ── Speed comparison: DFT O(N²) vs FFT O(N log N) ──\n")

    def dft_naive_list(x):
        N = len(x)
        X = []
        for k in range(N):
            total = 0 + 0j
            for n in range(N):
                total += x[n] * cmath.exp(-2j * cmath.pi * k * n / N)
            X.append(total)
        return X

    sizes = [64, 128, 256, 512, 1024]
    print(f"    {'N':>6} {'DFT O(N²)':>12} {'FFT O(NlogN)':>14} {'numpy FFT':>12} {'Speedup':>10}")
    print(f"    {'─'*6} {'─'*12} {'─'*14} {'─'*12} {'─'*10}")

    for N in sizes:
        test_signal = [math.sin(2 * math.pi * 5 * i / N) for i in range(N)]
        test_array = np.array(test_signal)

        # DFT timing
        if N <= 512:
            start = time.perf_counter()
            dft_naive_list(test_signal)
            t_dft = time.perf_counter() - start
        else:
            t_dft = float('inf')

        # Our FFT timing
        start = time.perf_counter()
        fft_cooley_tukey(test_signal)
        t_fft = time.perf_counter() - start

        # numpy timing
        start = time.perf_counter()
        for _ in range(100):
            np.fft.fft(test_array)
        t_np = (time.perf_counter() - start) / 100

        dft_str = f"{t_dft*1000:.1f}ms" if t_dft < 1 else "—"
        speedup = t_dft / t_fft if t_dft < float('inf') else 0

        print(f"    {N:>6} {dft_str:>12} {t_fft*1000:>11.1f}ms "
              f"{t_np*1000:>11.3f}ms"
              f" {speedup:>9.1f}×" if speedup > 0 else
              f"    {N:>6} {dft_str:>12} {t_fft*1000:>11.1f}ms "
              f"{t_np*1000:>11.3f}ms {'—':>10}")

    print("""
    The FFT is dramatically faster for large N.
    numpy's FFT is even faster because it's written in C and uses
    vectorized operations — but it's the SAME algorithm.

    ┌──────────────────────────────────────────────────────────┐
    │  CONNECTION TO REPO 1                                    │
    │                                                          │
    │  The FFT uses BIT-REVERSAL PERMUTATION to reorder the    │
    │  input before the butterfly stages. To bit-reverse an    │
    │  index: reverse its binary representation.               │
    │                                                          │
    │  Index 3 in 8-point FFT: 011 → 110 = 6                  │
    │  Index 5: 101 → 101 = 5 (palindrome!)                   │
    │                                                          │
    │  This is the SAME bit manipulation from repo 1's         │
    │  bit_tricks_demo.py — bit reversal, now serving the      │
    │  Fourier Transform.                                      │
    └──────────────────────────────────────────────────────────┘
    """)

    # Demonstrate bit reversal
    print("  ── Bit-reversal permutation for 8-point FFT ──\n")
    print(f"    {'Index':>6} {'Binary':>8} {'Reversed':>10} {'New Index':>11}")
    print(f"    {'─'*6} {'─'*8} {'─'*10} {'─'*11}")
    for i in range(8):
        binary = f"{i:03b}"
        reversed_bin = binary[::-1]
        new_idx = int(reversed_bin, 2)
        swap = "← swap" if new_idx > i else ("  (self)" if new_idx == i else "")
        print(f"    {i:>6} {binary:>8} {reversed_bin:>10} {new_idx:>11} {swap}")


# =============================================================================
# PART 4: SPECTRAL ANALYSIS — FINDING HIDDEN FREQUENCIES
# =============================================================================

def part4_spectral_analysis():
    print(f"\n{SEPARATOR}")
    print("  PART 4: SPECTRAL ANALYSIS")
    print("  Finding hidden frequencies in noisy signals")
    print(SEPARATOR)

    print("""
    The real power of the FFT: finding frequencies BURIED in noise.

    Imagine a signal with a weak periodic component hidden under
    heavy random noise. Your eyes can't see it. The FFT can.
    """)

    np.random.seed(42)
    N = 1024
    fs = 1024  # 1024 Hz sample rate
    t = np.arange(N) / fs

    # Hidden signal: weak 50 Hz + very weak 120 Hz
    clean = 0.7 * np.sin(2 * np.pi * 50 * t) + \
            0.3 * np.sin(2 * np.pi * 120 * t)

    # Bury it in noise
    noise = 2.0 * np.random.randn(N)
    noisy = clean + noise

    print("    Clean signal (50 Hz + 120 Hz):")
    ascii_plot(clean[:256].tolist(), height=9, width=60)
    print()
    print("    Same signal buried in noise (SNR ≈ -6 dB, noise is LOUDER):")
    ascii_plot(noisy[:256].tolist(), height=9, width=60)
    print()

    # FFT analysis
    X = np.fft.fft(noisy)
    freqs = np.fft.fftfreq(N, 1/fs)

    # Only positive frequencies
    pos_mask = freqs >= 0
    pos_freqs = freqs[pos_mask]
    pos_mags = np.abs(X[pos_mask]) / N

    print("    FFT magnitude spectrum of the NOISY signal:")
    ascii_spectrum(pos_freqs.tolist(), pos_mags.tolist(),
                   width=60, height=13, max_freq=200)
    print()

    # Find peaks
    threshold = 0.1
    peaks = [(pos_freqs[i], pos_mags[i]) for i in range(len(pos_freqs))
             if pos_mags[i] > threshold]
    peaks.sort(key=lambda x: -x[1])

    print("    Detected peaks (above noise floor):")
    for freq, mag in peaks[:5]:
        print(f"      {freq:>6.0f} Hz  magnitude = {mag:.3f}")

    print("""
    Even though the noise was LOUDER than the signal (negative SNR),
    the FFT extracted both hidden frequencies perfectly.

    This is why the FFT is used in:
      • Music apps (Shazam: identify songs from noisy recordings)
      • Medical imaging (MRI: reconstruct images from RF signals)
      • Seismology (find earthquake frequencies in ground vibrations)
      • Radar (find target frequencies in reflected signals)
      • Radio (tune to one station among hundreds of overlapping signals)
    """)


# =============================================================================
# PART 5: THE CONVOLUTION THEOREM
# =============================================================================

def part5_convolution():
    print(f"\n{SEPARATOR}")
    print("  PART 5: THE CONVOLUTION THEOREM")
    print("  Multiplication in frequency = convolution in time")
    print(SEPARATOR)

    print("""
    CONVOLUTION is one of the most fundamental operations in signal
    processing. It's how we apply filters, blur images, find
    correlations, and multiply polynomials.

    The definition:

        (f * g)[n] = Σ f[k] · g[n - k]

    Slide one signal over another, multiply point by point, sum.
    Direct computation is O(N²).

    ┌──────────────────────────────────────────────────────────┐
    │  THE CONVOLUTION THEOREM                                 │
    │                                                          │
    │  Convolution in time domain equals MULTIPLICATION        │
    │  in frequency domain:                                    │
    │                                                          │
    │      f * g  =  IFFT( FFT(f) · FFT(g) )                  │
    │                                                          │
    │  Cost:  O(N log N)  instead of  O(N²)                    │
    └──────────────────────────────────────────────────────────┘

    This is HUGE. Instead of O(N²) sliding/multiplying/summing,
    do three FFTs (each O(N log N)) and one pointwise multiply (O(N)).
    """)

    # Demonstrate: convolution = filtering
    np.random.seed(42)
    N = 512
    t = np.arange(N) / N

    # Signal with noise
    signal = np.sin(2 * np.pi * 5 * t) + 0.5 * np.random.randn(N)

    # Smoothing kernel (moving average)
    kernel_size = 15
    kernel = np.ones(kernel_size) / kernel_size

    # Direct convolution
    start = time.perf_counter()
    direct_result = np.convolve(signal, kernel, mode='same')
    t_direct = time.perf_counter() - start

    # FFT-based convolution
    start = time.perf_counter()
    padded_signal = np.zeros(N + kernel_size - 1)
    padded_kernel = np.zeros(N + kernel_size - 1)
    padded_signal[:N] = signal
    padded_kernel[:kernel_size] = kernel

    fft_result = np.real(np.fft.ifft(
        np.fft.fft(padded_signal) * np.fft.fft(padded_kernel)
    ))
    # Trim to match 'same' mode
    offset = (kernel_size - 1) // 2
    fft_result = fft_result[offset:offset + N]
    t_fft = time.perf_counter() - start

    print("    Noisy signal:")
    ascii_plot(signal[:256].tolist(), height=9, width=60)
    print()
    print("    After smoothing filter (convolution with moving average):")
    ascii_plot(direct_result[:256].tolist(), height=9, width=60)
    print()

    error = np.max(np.abs(direct_result - fft_result))
    print(f"    Direct convolution:   {t_direct*1000:.3f}ms")
    print(f"    FFT convolution:      {t_fft*1000:.3f}ms")
    print(f"    Max difference:       {error:.2e}")

    print("""
    For small kernels, direct convolution wins due to FFT overhead.
    But for large kernels or long signals, FFT convolution dominates.

    Applications of the convolution theorem:

      • Audio reverb:     convolve sound with room impulse response
      • Image blur:       convolve image with Gaussian kernel
      • Edge detection:   convolve with Sobel/Laplacian kernel
      • Polynomial mult:  coefficients are "signals," multiply via FFT
      • Correlation:      convolve with time-reversed template
    """)


# =============================================================================
# PART 6: WINDOWING AND SPECTRAL LEAKAGE
# =============================================================================

def part6_windowing():
    print(f"\n{SEPARATOR}")
    print("  PART 6: WINDOWING AND SPECTRAL LEAKAGE")
    print("  Why real-world FFT isn't as clean as the theory")
    print(SEPARATOR)

    print("""
    In theory, the DFT assumes the signal repeats FOREVER.
    In practice, we have a FINITE window of data. If the signal
    doesn't align perfectly with the window boundary, the DFT
    sees artificial discontinuities at the edges.

    This causes SPECTRAL LEAKAGE: energy from one frequency
    "leaks" into neighboring frequencies, smearing the spectrum.
    """)

    N = 256
    fs = 256
    t = np.arange(N) / fs

    # A pure 10 Hz tone — but 10 doesn't divide evenly into 256/256
    freq = 10.5  # deliberately non-integer to cause leakage
    signal = np.sin(2 * np.pi * freq * t)

    # No window (rectangular)
    X_rect = np.fft.fft(signal)
    freqs = np.fft.fftfreq(N, 1/fs)
    pos = freqs >= 0
    mags_rect = np.abs(X_rect[pos]) / N

    print(f"    Signal: {freq} Hz sine (non-integer frequency)")
    print()
    print("    Spectrum with rectangular window (no windowing):")
    ascii_spectrum(freqs[pos].tolist(), mags_rect.tolist(),
                   width=60, height=11, max_freq=30)
    print()

    # Hann window
    window = 0.5 * (1 - np.cos(2 * np.pi * np.arange(N) / N))
    X_hann = np.fft.fft(signal * window)
    mags_hann = np.abs(X_hann[pos]) / N * 2  # compensate for window gain

    print("    Spectrum with Hann window:")
    ascii_spectrum(freqs[pos].tolist(), mags_hann.tolist(),
                   width=60, height=11, max_freq=30)

    print("""
    The Hann window dramatically reduces spectral leakage.
    The peak is sharper and the "skirt" around it is much lower.

    Common windows and their properties:

      ┌──────────────┬───────────────┬────────────────────────┐
      │ Window       │ Main lobe     │ Side lobe attenuation  │
      ├──────────────┼───────────────┼────────────────────────┤
      │ Rectangular  │ Narrow        │ -13 dB (poor)          │
      │ Hann         │ Medium        │ -31 dB                 │
      │ Hamming      │ Medium        │ -42 dB                 │
      │ Blackman     │ Wide          │ -58 dB (excellent)     │
      │ Kaiser       │ Adjustable    │ Adjustable             │
      └──────────────┴───────────────┴────────────────────────┘

    The tradeoff: narrower main lobe (better frequency resolution)
    vs. lower side lobes (less leakage). You can't have both —
    this is ANOTHER manifestation of the uncertainty principle!
    """)


# =============================================================================
# PART 7: PARSEVAL'S THEOREM — ENERGY CONSERVATION
# =============================================================================

def part7_parseval():
    print(f"\n{SEPARATOR}")
    print("  PART 7: PARSEVAL'S THEOREM — ENERGY CONSERVATION")
    print("  The universe's bookkeeping between time and frequency")
    print(SEPARATOR)

    print("""
    ┌──────────────────────────────────────────────────────────┐
    │  PARSEVAL'S THEOREM                                      │
    │                                                          │
    │      Σ |x[n]|²  =  (1/N) · Σ |X[k]|²                   │
    │                                                          │
    │  Total energy in time domain = total energy in frequency.│
    │  The Fourier Transform conserves energy perfectly.        │
    └──────────────────────────────────────────────────────────┘

    This isn't just mathematical convenience — it's a CONSERVATION
    LAW. No energy is created or destroyed by the transform.
    """)

    np.random.seed(42)
    N = 1024

    signals = [
        ("Pure sine", np.sin(2 * np.pi * 10 * np.arange(N) / N)),
        ("Two sines", np.sin(2 * np.pi * 10 * np.arange(N) / N) +
                      0.5 * np.sin(2 * np.pi * 30 * np.arange(N) / N)),
        ("White noise", np.random.randn(N)),
        ("Chirp", np.sin(2 * np.pi * np.cumsum(np.linspace(1, 50, N) / N))),
    ]

    print(f"    {'Signal':>15} {'Time energy':>14} {'Freq energy':>14} {'Ratio':>8}")
    print(f"    {'─'*15} {'─'*14} {'─'*14} {'─'*8}")

    for name, sig in signals:
        time_energy = np.sum(np.abs(sig) ** 2)
        X = np.fft.fft(sig)
        freq_energy = np.sum(np.abs(X) ** 2) / N
        ratio = freq_energy / time_energy

        print(f"    {name:>15} {time_energy:>14.4f} {freq_energy:>14.4f} {ratio:>8.6f}")

    print("""
    Every ratio is 1.000000 — perfect energy conservation.

    Why this matters:

    1. COMPRESSION: If you zero out small frequency components,
       you know EXACTLY how much signal energy you're discarding.
       This is how JPEG/MP3 decide what to throw away.

    2. FILTERING: When you remove a frequency band, Parseval's
       tells you exactly how much power you removed.

    3. QUANTUM MECHANICS: Unitarity (energy conservation) of
       quantum gates is the quantum version of Parseval's theorem.
       The QFT is unitary precisely BECAUSE the classical FFT
       conserves energy.

    ► Connection to repo 1: CRC checksums detect errors by
      checking invariants. Parseval's theorem is the spectral
      checksum — if energy isn't conserved, something went wrong.
    """)


# =============================================================================
# PART 8: THE PHILOSOPHY
# =============================================================================

def part8_philosophy():
    print(f"\n{SEPARATOR}")
    print("  CONCLUSION: THE DUAL VIEW")
    print(SEPARATOR)

    print("""
    The Fourier Transform reveals a fundamental duality:

    ┌──────────────────────────────────────────────────────────┐
    │  Every signal has TWO representations:                   │
    │                                                          │
    │  TIME DOMAIN                │  FREQUENCY DOMAIN          │
    │  ───────────                │  ────────────────          │
    │  "What happened when?"      │  "What frequencies exist?" │
    │  Direct measurement         │  FFT analysis              │
    │  Convolution: O(N²)         │  Multiplication: O(N)      │
    │  Local changes easy         │  Global patterns easy      │
    │                                                          │
    │  Connected by: the FOURIER TRANSFORM                     │
    │  Made practical by: the FFT (Cooley-Tukey, 1965)         │
    └──────────────────────────────────────────────────────────┘

    This is the continuous-world counterpart of repo 1's core
    insight. There, the same data (a float's bits) could be read
    as a number OR as a logarithmic encoding. Here, the same
    data (a signal's samples) can be read as amplitudes over time
    OR as magnitudes at different frequencies.

    SAME PRINCIPLE:
      Representation has structure. Read it differently, gain power.

    ┌──────────────────────────────────────────────────────────┐
    │  Carmack (1999): Reinterpret float bits as integer       │
    │                  → fast inverse square root               │
    │                                                          │
    │  Cooley-Tukey (1965): Reinterpret time samples as        │
    │                       frequency components via FFT        │
    │                       → fast spectral analysis            │
    │                                                          │
    │  Same trick. Different domain. Same power.               │
    └──────────────────────────────────────────────────────────┘

    NEXT: The Fourier Transform tells you WHAT frequencies exist
    but not WHEN they happen. For that, we need wavelets.
    → wavelets_and_uncertainty.py
    """)


# =============================================================================
# RUN EVERYTHING
# =============================================================================

if __name__ == "__main__":
    print("\n" + "█" * 72)
    print("█  THE FOURIER TRANSFORM                                             █")
    print("█  Seeing frequencies: from Euler's formula to the FFT               █")
    print("█" * 72)

    part1_euler()
    part2_dft()
    part3_fft()
    part4_spectral_analysis()
    part5_convolution()
    part6_windowing()
    part7_parseval()
    part8_philosophy()
