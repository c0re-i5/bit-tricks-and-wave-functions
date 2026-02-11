"""
=============================================================================
  SIGNALS AND SAMPLING — The Analog Bridge
  Where the continuous world becomes discrete, and why that works
=============================================================================

  Before Fourier, before wavelets, before quantum mechanics, there's
  a more fundamental question: how do you turn a continuous wave into
  discrete numbers that a computer can process?

  The answer — the sampling theorem — is one of the most important
  results in information theory. It says you can PERFECTLY reconstruct
  a continuous signal from discrete samples, as long as you sample
  fast enough. Not approximately. PERFECTLY.

  This script explores that bridge between continuous and discrete,
  connecting to the bit-level world from repo 1.
"""

import math
import numpy as np

SEPARATOR = "=" * 72
SUBSEP = "─" * 72

# ── Helper: ASCII waveform plotter ───────────────────────────────────────────

def ascii_plot(values, width=64, height=17, label="", show_zero=True):
    """
    Render a signal as ASCII art.

    Uses braille-like characters for a clean waveform display.
    The approach: for each column, map the signal value to a row,
    then place a marker character.
    """
    n = len(values)
    if n == 0:
        return

    min_val = min(values)
    max_val = max(values)

    # Avoid degenerate case
    if max_val == min_val:
        max_val = min_val + 1.0

    # Resample signal to fit width
    indices = [int(i * (n - 1) / (width - 1)) for i in range(width)]
    sampled = [values[idx] for idx in indices]

    # Build grid
    grid = [[" " for _ in range(width)] for _ in range(height)]

    # Plot zero line if requested
    if show_zero and min_val < 0 < max_val:
        zero_row = int((max_val - 0) / (max_val - min_val) * (height - 1))
        zero_row = max(0, min(height - 1, zero_row))
        for c in range(width):
            grid[zero_row][c] = "·"

    # Plot signal
    for c in range(width):
        val = sampled[c]
        row = int((max_val - val) / (max_val - min_val) * (height - 1))
        row = max(0, min(height - 1, row))
        grid[row][c] = "█"

    # Connect vertically between consecutive points for continuity
    for c in range(1, width):
        val_prev = sampled[c - 1]
        val_curr = sampled[c]
        row_prev = int((max_val - val_prev) / (max_val - min_val) * (height - 1))
        row_curr = int((max_val - val_curr) / (max_val - min_val) * (height - 1))
        row_prev = max(0, min(height - 1, row_prev))
        row_curr = max(0, min(height - 1, row_curr))

        r_lo = min(row_prev, row_curr)
        r_hi = max(row_prev, row_curr)
        for r in range(r_lo, r_hi + 1):
            if grid[r][c] == " ":
                grid[r][c] = "│"

    # Render
    for r in range(height):
        if r == 0:
            axis_label = f"{max_val:>8.2f}"
        elif r == height - 1:
            axis_label = f"{min_val:>8.2f}"
        elif r == height // 2:
            mid = (max_val + min_val) / 2
            axis_label = f"{mid:>8.2f}"
        else:
            axis_label = "        "
        row_str = "".join(grid[r])
        print(f"    {axis_label} ┤{row_str}")
    print(f"    {'':>8} └{'─' * width}")
    if label:
        print(f"    {'':>8}  {label}")


def ascii_stem(sample_positions, sample_values, total_length, width=64, height=13):
    """
    Render discrete samples as a stem plot (vertical lines from zero).
    Shows the fundamental nature of sampled data.
    """
    min_val = min(sample_values) if sample_values else -1
    max_val = max(sample_values) if sample_values else 1
    if max_val == min_val:
        max_val = min_val + 1.0

    # Build grid
    grid = [[" " for _ in range(width)] for _ in range(height)]

    # Zero row
    if min_val < 0 < max_val:
        zero_row = int((max_val - 0) / (max_val - min_val) * (height - 1))
        zero_row = max(0, min(height - 1, zero_row))
        for c in range(width):
            grid[zero_row][c] = "·"
    else:
        zero_row = height - 1

    # Plot each sample as a stem
    for pos, val in zip(sample_positions, sample_values):
        col = int(pos / total_length * (width - 1))
        col = max(0, min(width - 1, col))
        row = int((max_val - val) / (max_val - min_val) * (height - 1))
        row = max(0, min(height - 1, row))

        # Draw stem from zero to value
        r_lo = min(zero_row, row)
        r_hi = max(zero_row, row)
        for r in range(r_lo, r_hi + 1):
            grid[r][col] = "│"
        grid[row][col] = "●"

    # Render
    for r in range(height):
        if r == 0:
            axis_label = f"{max_val:>8.2f}"
        elif r == height - 1:
            axis_label = f"{min_val:>8.2f}"
        else:
            axis_label = "        "
        print(f"    {axis_label} ┤{''.join(grid[r])}")
    print(f"    {'':>8} └{'─' * width}")


# =============================================================================
# PART 1: WHAT IS A SIGNAL?
# =============================================================================

def part1_what_is_a_signal():
    print(f"\n{SEPARATOR}")
    print("  PART 1: WHAT IS A SIGNAL?")
    print("  The continuous world before the bits")
    print(SEPARATOR)

    print("""
    A SIGNAL is any quantity that varies over time (or space):

      • Sound:       air pressure vs. time
      • Light:       electromagnetic field vs. position
      • Temperature: degrees vs. time
      • Stock price: dollars vs. time
      • WiFi:        voltage vs. time

    Mathematically, a signal is just a function: f(t) → value.

    The simplest signal is a SINE WAVE:

        f(t) = A · sin(2π · f · t + φ)

        A = amplitude   (how tall)
        f = frequency   (how fast — cycles per second, measured in Hz)
        φ = phase       (where in the cycle we start)

    Let's see one:
    """)

    # Generate a 3 Hz sine wave
    t = np.linspace(0, 1, 1000)
    signal = np.sin(2 * np.pi * 3 * t)
    ascii_plot(signal.tolist(), label="3 Hz sine wave over 1 second")

    print("""
    That's 3 complete cycles in 1 second → 3 Hz (Hertz).

    Now the KEY INSIGHT that Fourier had in 1807:
    """)

    print("""    ┌──────────────────────────────────────────────────────────┐
    │  ANY signal — no matter how complicated — can be         │
    │  written as a SUM of sine waves at different             │
    │  frequencies, amplitudes, and phases.                    │
    └──────────────────────────────────────────────────────────┘
    """)

    # Show superposition
    print("  ── Superposition: building complex signals from simple ones ──\n")

    s1 = np.sin(2 * np.pi * 3 * t)
    s2 = 0.5 * np.sin(2 * np.pi * 7 * t)
    s3 = 0.3 * np.sin(2 * np.pi * 13 * t)
    composite = s1 + s2 + s3

    print("    Component 1:  sin(2π·3·t)           — 3 Hz, amplitude 1.0")
    ascii_plot(s1.tolist(), height=9, label="")
    print("    Component 2:  0.5·sin(2π·7·t)       — 7 Hz, amplitude 0.5")
    ascii_plot(s2.tolist(), height=9, label="")
    print("    Component 3:  0.3·sin(2π·13·t)      — 13 Hz, amplitude 0.3")
    ascii_plot(s3.tolist(), height=9, label="")
    print("    SUM of all three:")
    ascii_plot(composite.tolist(), height=11, label="Composite: 3 Hz + 7 Hz + 13 Hz")

    print("""
    The composite looks complex, but it's just three simple waves added.
    EVERY sound you hear — music, speech, noise — is exactly this:
    sine waves at different frequencies, added together.

    A piano chord is 3 frequencies. A human voice is hundreds.
    A full orchestral recording is thousands.

    But they're ALL just sums of sines.
    """)


# =============================================================================
# PART 2: THE SAMPLING THEOREM
# =============================================================================

def part2_sampling_theorem():
    print(f"\n{SEPARATOR}")
    print("  PART 2: THE SAMPLING THEOREM")
    print("  How to capture infinity in a finite list of numbers")
    print(SEPARATOR)

    print("""
    A continuous signal has a value at EVERY point in time —
    uncountably infinite values. A computer can only store
    finitely many numbers. How do we bridge this gap?

    We SAMPLE: measure the signal at regular intervals.

        continuous:  f(t) for all real t
        sampled:     f(0), f(Δt), f(2Δt), f(3Δt), ...

    The spacing Δt is the SAMPLE PERIOD.
    Its inverse 1/Δt is the SAMPLE RATE (samples per second).

    CD audio: 44,100 samples/second (44.1 kHz)
    Phone:     8,000 samples/second  (8 kHz)
    Blu-ray:  96,000 samples/second (96 kHz)

    But HOW FAST do we need to sample? This is where the magic is.
    """)

    print("""    ┌──────────────────────────────────────────────────────────┐
    │  NYQUIST-SHANNON SAMPLING THEOREM (1949)                 │
    │                                                          │
    │  If a signal contains no frequencies higher than B Hz,   │
    │  it is COMPLETELY determined by samples taken at a       │
    │  rate of 2B samples per second.                          │
    │                                                          │
    │  f_sample ≥ 2 · f_max  →  PERFECT reconstruction        │
    └──────────────────────────────────────────────────────────┘

    Not approximate. PERFECT. Every last detail can be recovered
    from the samples alone. This is one of the deepest results
    in information theory.
    """)

    # Demonstrate proper sampling
    print("  ── Proper sampling: 10 Hz signal sampled at 25 Hz ──\n")

    freq = 10  # Hz
    duration = 0.5  # seconds
    t_cont = np.linspace(0, duration, 2000)  # "continuous"
    signal_cont = np.sin(2 * np.pi * freq * t_cont)

    sample_rate = 25  # samples per second (> 2 * 10)
    n_samples = int(duration * sample_rate)
    t_sampled = np.linspace(0, duration, n_samples, endpoint=False)
    signal_sampled = np.sin(2 * np.pi * freq * t_sampled)

    print(f"    Signal: {freq} Hz sine wave")
    print(f"    Nyquist rate: {2 * freq} Hz (minimum)")
    print(f"    Our sample rate: {sample_rate} Hz  ✓ above Nyquist")
    print(f"    Samples taken: {n_samples}")
    print()

    print("    Continuous signal:")
    ascii_plot(signal_cont.tolist(), height=9, width=60)
    print()
    print("    Sampled version (● = sample points):")
    ascii_stem(t_sampled.tolist(), signal_sampled.tolist(),
               duration, height=9, width=60)

    print("""
    With 25 samples per second, we captured a 10 Hz wave.
    Nyquist says 20 would suffice. We have margin.
    From these samples alone, we can reconstruct the original
    continuous signal PERFECTLY.
    """)


# =============================================================================
# PART 3: ALIASING — WHEN SAMPLING GOES WRONG
# =============================================================================

def part3_aliasing():
    print(f"\n{SEPARATOR}")
    print("  PART 3: ALIASING — WHEN SAMPLING GOES WRONG")
    print("  The demon that haunts undersampled signals")
    print(SEPARATOR)

    print("""
    What happens if we sample TOO SLOWLY — below the Nyquist rate?

    The signal FOLDS BACK. High frequencies masquerade as low
    frequencies. This is ALIASING, and it's irreversible.

    You've seen it: car wheels that appear to spin backward in
    video. That's temporal aliasing — the frame rate is too low
    to capture the true rotation speed.
    """)

    freq = 10  # Hz
    duration = 1.0
    t_cont = np.linspace(0, duration, 5000)
    signal_cont = np.sin(2 * np.pi * freq * t_cont)

    print(f"    The true signal: {freq} Hz sine wave\n")
    ascii_plot(signal_cont[:2500].tolist(), height=9, width=60, label="True: 10 Hz")

    # Undersample
    sample_rates = [25, 15, 12]
    for sr in sample_rates:
        n_samp = int(duration * sr)
        t_samp = np.linspace(0, duration, n_samp, endpoint=False)
        s_samp = np.sin(2 * np.pi * freq * t_samp)

        # Reconstruct (naive linear interpolation to show effect)
        reconstructed = np.interp(t_cont, t_samp, s_samp)

        nyquist_ok = "✓" if sr >= 2 * freq else "✗ ALIASED"
        print(f"\n    Sampled at {sr} Hz ({n_samp} samples) — {nyquist_ok}")
        ascii_plot(reconstructed[:2500].tolist(), height=9, width=60)

    print("""
    At 25 Hz: perfect capture (above 2×10 = 20 Hz Nyquist rate).
    At 15 Hz: BELOW Nyquist. The 10 Hz signal aliases to 5 Hz!
    At 12 Hz: even worse. The reconstructed signal is 2 Hz.

    The aliased frequency is:  f_alias = |f_signal - n·f_sample|
    where n is the nearest integer multiple.

    ┌──────────────────────────────────────────────────────────┐
    │  ALIASING IS IRREVERSIBLE                                │
    │                                                          │
    │  Once you've sampled too slowly, the original frequency  │
    │  information is GONE. No algorithm can recover it.       │
    │  This is why CD players have anti-aliasing filters       │
    │  that remove frequencies above 22.05 kHz BEFORE          │
    │  sampling at 44.1 kHz.                                   │
    └──────────────────────────────────────────────────────────┘

    ► Connection to repo 1: Aliasing is an information-theoretic
      limit, like Shannon's entropy bound for compression. You
      can't compress below entropy; you can't sample below Nyquist.
      Both are fundamental limits on representation.
    """)


# =============================================================================
# PART 4: QUANTIZATION — WHERE BITS MEET WAVES
# =============================================================================

def part4_quantization():
    print(f"\n{SEPARATOR}")
    print("  PART 4: QUANTIZATION — WHERE BITS MEET WAVES")
    print("  How continuous amplitudes become discrete numbers")
    print(SEPARATOR)

    print("""
    Sampling discretizes TIME. But the amplitude values are still
    continuous real numbers. To store them digitally, we must also
    discretize AMPLITUDE. This is QUANTIZATION.

    With n bits per sample, we get 2^n possible amplitude levels:

      • 8-bit audio:  256 levels      (telephone quality)
      • 16-bit audio: 65,536 levels   (CD quality)
      • 24-bit audio: 16,777,216 levels (studio quality)
      • 32-bit float: ~7 decimal digits (scientific instruments)

    ► Connection to repo 1: This is where IEEE 754 from
      bit_tricks_demo.py enters the signal processing world!
      Each sample is stored as a float or integer, with the
      same bit-level structure we explored there.
    """)

    # Demonstrate quantization at different bit depths
    t = np.linspace(0, 1, 1000)
    signal = np.sin(2 * np.pi * 3 * t) + 0.3 * np.sin(2 * np.pi * 11 * t)

    print("    Original signal (infinite precision):")
    ascii_plot(signal.tolist(), height=11, width=60)
    print()

    bit_depths = [2, 4, 8, 16]
    for bits in bit_depths:
        levels = 2 ** bits
        # Quantize: map [-max, +max] to [0, levels-1], then back
        sig_min, sig_max = signal.min(), signal.max()
        normalized = (signal - sig_min) / (sig_max - sig_min)
        quantized_int = np.round(normalized * (levels - 1)).astype(int)
        quantized = quantized_int / (levels - 1) * (sig_max - sig_min) + sig_min

        error = signal - quantized
        snr = 10 * np.log10(np.mean(signal**2) / np.mean(error**2))

        print(f"    {bits}-bit quantization ({levels} levels)  "
              f"SNR = {snr:.1f} dB")
        ascii_plot(quantized.tolist(), height=9, width=60)
        print()

    print("""
    The theoretical SNR for uniform quantization is:

        SNR ≈ 6.02 · n + 1.76 dB     (n = bit depth)

    Every extra bit of depth gives about 6 dB of dynamic range.
    That's why 16-bit CD audio (96 dB range) sounds "good enough"
    for most people — it covers the full dynamic range of hearing.

    ┌──────────────────────────────────────────────────────────┐
    │  BIT DEPTH IS RESOLUTION                                 │
    │                                                          │
    │  8 bits  → crude, audibly noisy (telephone)              │
    │  16 bits → clean, covers human hearing range (CD)        │
    │  24 bits → professional, exceeds perception (studio)     │
    │  32-bit float → scientific precision (IEEE 754!)         │
    └──────────────────────────────────────────────────────────┘

    ► The quantization error is mathematically identical to the
      rounding error in floating-point arithmetic from repo 1.
      Both are consequences of mapping continuous values to
      discrete representations.
    """)


# =============================================================================
# PART 5: RECONSTRUCTION — GETTING THE CONTINUOUS BACK
# =============================================================================

def part5_reconstruction():
    print(f"\n{SEPARATOR}")
    print("  PART 5: RECONSTRUCTION")
    print("  From discrete samples back to the continuous signal")
    print(SEPARATOR)

    print("""
    The Nyquist theorem promises PERFECT reconstruction. But how?

    The answer is the SINC FUNCTION:

        sinc(x) = sin(πx) / (πx)

    It looks like a splash in a pond — a central peak with ripples
    that decay toward zero.
    """)

    # Plot sinc function
    t = np.linspace(-5, 5, 1000)
    sinc_vals = np.where(np.abs(t) < 1e-10, 1.0, np.sin(np.pi * t) / (np.pi * t))
    ascii_plot(sinc_vals.tolist(), height=13, width=64, label="sinc(t) = sin(πt) / (πt)")

    print("""
    The Whittaker-Shannon interpolation formula says:

        f(t) = Σ f[n] · sinc(f_s · t − n)

    Each sample creates a sinc function centered at its position.
    The sinc functions ADD UP to perfectly reconstruct the original
    continuous signal. The ripples of each sinc cancel out exactly
    where the other samples are — this is called ORTHOGONALITY.
    """)

    # Demonstrate reconstruction
    print("  ── Reconstruction Demo ──\n")

    freq = 5
    duration = 1.0
    sample_rate = 15  # well above Nyquist for 5 Hz

    # Create samples
    n_samples = int(duration * sample_rate)
    t_samp = np.arange(n_samples) / sample_rate
    samples = np.sin(2 * np.pi * freq * t_samp)

    # Reconstruct at high resolution using sinc interpolation
    t_recon = np.linspace(0, duration, 2000)
    reconstructed = np.zeros_like(t_recon)

    for n in range(n_samples):
        sinc_arg = sample_rate * t_recon - n
        # sinc function: sin(pi*x)/(pi*x), with sinc(0)=1
        pi_arg = np.pi * sinc_arg
        sinc_vals_n = np.ones_like(sinc_arg)
        nonzero = np.abs(sinc_arg) > 1e-10
        sinc_vals_n[nonzero] = np.sin(pi_arg[nonzero]) / pi_arg[nonzero]
        reconstructed += samples[n] * sinc_vals_n

    # True signal for comparison
    true_signal = np.sin(2 * np.pi * freq * t_recon)

    # Measure error
    error = np.sqrt(np.mean((reconstructed - true_signal) ** 2))

    print(f"    Signal: {freq} Hz sine")
    print(f"    Samples: {n_samples} at {sample_rate} Hz")
    print(f"    Reconstruction error (RMSE): {error:.2e}")
    print()

    print("    Samples:")
    ascii_stem(t_samp.tolist(), samples.tolist(), duration, height=9, width=60)
    print()
    print("    Reconstructed (via sinc interpolation):")
    ascii_plot(reconstructed[100:1900].tolist(), height=9, width=60)
    print()
    print("    True continuous signal:")
    ascii_plot(true_signal[100:1900].tolist(), height=9, width=60)
    print()

    print(f"""
    Reconstruction RMSE: {error:.2e} — essentially perfect!

    The tiny residual error is numerical, not theoretical.
    With infinite precision arithmetic, the reconstruction
    would be EXACT.

    ► This is mathematically proven. Not heuristic. Not approximate.
      Given samples at the Nyquist rate, sinc interpolation gives
      you back the EXACT original continuous signal.
    """)


# =============================================================================
# PART 6: THE COMPLETE ADC/DAC PIPELINE
# =============================================================================

def part6_adc_dac():
    print(f"\n{SEPARATOR}")
    print("  PART 6: THE ADC/DAC PIPELINE")
    print("  Everything comes together")
    print(SEPARATOR)

    print("""
    Every time you listen to music, make a phone call, or watch
    a video, this pipeline runs:

    ┌────────────────────────────────────────────────────────────────┐
    │  ANALOG → DIGITAL (ADC)                                       │
    │                                                                │
    │  1. Anti-alias filter:  Remove frequencies above f_s/2        │
    │  2. Sample:             Measure at f_s samples/second         │
    │  3. Quantize:           Round to nearest n-bit level          │
    │  4. Encode:             Store as binary (this is repo 1!)     │
    │                                                                │
    │  DIGITAL → ANALOG (DAC)                                       │
    │                                                                │
    │  5. Decode:             Read binary → amplitude values        │
    │  6. Reconstruct:        Sinc interpolation (or approximation) │
    │  7. Smoothing filter:   Remove reconstruction artifacts       │
    │  8. Output:             Drive speaker / display / etc.        │
    └────────────────────────────────────────────────────────────────┘

    Let's run the complete pipeline on a composite signal.
    """)

    # Create a realistic composite signal
    np.random.seed(42)
    duration = 0.1  # 100ms of audio-like signal
    f_true = 48000  # "true" continuous rate
    t_cont = np.linspace(0, duration, int(duration * f_true))

    # Simulate a musical note: fundamental + harmonics
    fundamental = 440  # A4 note
    signal_cont = (
        1.0 * np.sin(2 * np.pi * fundamental * t_cont) +
        0.5 * np.sin(2 * np.pi * 2 * fundamental * t_cont) +
        0.25 * np.sin(2 * np.pi * 3 * fundamental * t_cont) +
        0.12 * np.sin(2 * np.pi * 4 * fundamental * t_cont)
    )
    # Normalize
    signal_cont = signal_cont / np.max(np.abs(signal_cont))

    print("    Original signal: A4 (440 Hz) with harmonics")
    print(f"    Components: 440 Hz + 880 Hz + 1320 Hz + 1760 Hz")
    print(f"    Highest frequency: {4 * fundamental} Hz")
    print()

    # Show a few cycles
    show_n = int(0.01 * f_true)  # 10ms
    ascii_plot(signal_cont[:show_n].tolist(), height=11, width=64,
               label="First 10ms of continuous signal")
    print()

    # Step 1: Anti-alias filter (simulated by band-limiting)
    f_sample = 8000  # 8 kHz (telephone quality)
    f_nyquist = f_sample / 2
    print(f"    STEP 1: Anti-alias filter (remove > {f_nyquist:.0f} Hz)")
    print(f"    → Frequencies 440, 880, 1320, 1760 Hz")
    print(f"    → All below {f_nyquist:.0f} Hz Nyquist limit ✓")
    print()

    # Step 2: Sample
    n_samples = int(duration * f_sample)
    t_samp = np.arange(n_samples) / f_sample
    # Compute samples from the continuous signal (nearest sample)
    samp_indices = (t_samp * f_true).astype(int)
    samp_indices = np.clip(samp_indices, 0, len(signal_cont) - 1)
    samples_float = signal_cont[samp_indices]

    print(f"    STEP 2: Sample at {f_sample} Hz → {n_samples} samples")
    show_samp_n = int(0.01 * f_sample)
    ascii_stem(list(range(show_samp_n)),
               samples_float[:show_samp_n].tolist(),
               show_samp_n, height=9, width=64)
    print()

    # Step 3: Quantize
    bit_depth = 16
    levels = 2 ** bit_depth
    samples_normalized = (samples_float + 1) / 2  # map [-1, 1] to [0, 1]
    samples_quantized_int = np.round(samples_normalized * (levels - 1)).astype(int)
    samples_quantized = samples_quantized_int / (levels - 1) * 2 - 1

    quant_error = np.sqrt(np.mean((samples_float - samples_quantized) ** 2))
    quant_snr = 10 * np.log10(np.mean(samples_float**2) /
                               np.mean((samples_float - samples_quantized)**2 + 1e-30))

    print(f"    STEP 3: Quantize to {bit_depth} bits ({levels:,} levels)")
    print(f"    Quantization error: {quant_error:.2e}")
    print(f"    Quantization SNR:   {quant_snr:.1f} dB")
    print()

    # Step 4: Encode (show binary representation of a sample)
    example_idx = n_samples // 4
    example_int = samples_quantized_int[example_idx]
    example_bits = f"{example_int:016b}"
    print(f"    STEP 4: Encode as binary")
    print(f"    Sample #{example_idx}: value = {samples_quantized[example_idx]:.6f}")
    print(f"    → Quantized integer: {example_int}")
    print(f"    → Binary: {example_bits[:4]} {example_bits[4:8]} "
          f"{example_bits[8:12]} {example_bits[12:]}")
    print(f"    → This is where repo 1 begins: bits in memory.")
    print()

    # Steps 5-7: Reconstruct
    t_recon = np.linspace(0, duration, int(duration * f_true))
    reconstructed = np.zeros_like(t_recon)

    # Use sinc interpolation (truncated for speed)
    for n in range(n_samples):
        sinc_arg = f_sample * t_recon - n
        pi_arg = np.pi * sinc_arg
        sinc_vals = np.ones_like(sinc_arg)
        nonzero = np.abs(sinc_arg) > 1e-10
        sinc_vals[nonzero] = np.sin(pi_arg[nonzero]) / pi_arg[nonzero]
        # Truncate sinc for speed (real DACs use finite impulse response)
        window = np.abs(sinc_arg) < 20
        reconstructed += samples_quantized[n] * sinc_vals * window

    print(f"    STEPS 5-7: Decode → Reconstruct → Smooth")
    print()
    print("    Reconstructed signal (first 10ms):")
    ascii_plot(reconstructed[:show_n].tolist(), height=11, width=64)
    print()

    # Measure fidelity
    valid = min(len(signal_cont), len(reconstructed))
    error = np.sqrt(np.mean((signal_cont[:valid] - reconstructed[:valid]) ** 2))
    snr = 10 * np.log10(
        np.mean(signal_cont[:valid]**2) /
        (np.mean((signal_cont[:valid] - reconstructed[:valid])**2) + 1e-30)
    )

    print(f"    End-to-end RMSE:  {error:.4f}")
    print(f"    End-to-end SNR:   {snr:.1f} dB")

    # Storage
    raw_bytes = n_samples * bit_depth // 8
    print(f"\n    Storage: {n_samples} samples × {bit_depth} bits = "
          f"{raw_bytes:,} bytes for {duration*1000:.0f}ms of audio")
    bitrate = f_sample * bit_depth
    print(f"    Bitrate: {bitrate:,} bits/sec = {bitrate/1000:.0f} kbps")


# =============================================================================
# PART 7: THE PHILOSOPHY — CONTINUOUS ↔ DISCRETE
# =============================================================================

def part7_philosophy():
    print(f"\n{SEPARATOR}")
    print("  CONCLUSION: THE BRIDGE")
    print(SEPARATOR)

    print("""
    This script has shown the bridge between the continuous world
    of waves and the discrete world of bits:

    ┌──────────────────────────────────────────────────────────┐
    │  CONTINUOUS                │  DISCRETE                    │
    │  ─────────                │  ────────                    │
    │  Infinite precision       │  Finite bits                 │
    │  Analog signal            │  Digital samples             │
    │  Smooth curves            │  Staircase quantization      │
    │  Anti-alias filter        │  Bit manipulation            │
    │  Sinc interpolation       │  Integer arithmetic          │
    │                           │                              │
    │  Connected by:  SAMPLING THEOREM                         │
    │  f_sample ≥ 2·f_max → perfect reconstruction            │
    └──────────────────────────────────────────────────────────┘

    KEY INSIGHTS FROM THIS CHAPTER:

    1. SAMPLING discretizes time. The Nyquist rate is the minimum.
    2. QUANTIZATION discretizes amplitude. Bit depth = precision.
    3. ALIASING is irreversible — it's the sampling world's
       equivalent of lossy compression below Shannon entropy.
    4. RECONSTRUCTION via sinc interpolation is PERFECT — given
       enough samples, you lose nothing.
    5. The ADC/DAC pipeline is the gateway between the physical
       world and the digital world of repo 1.

    ┌──────────────────────────────────────────────────────────┐
    │  Repo 1 started with bits and asked:                     │
    │    "What structure is hidden in this representation?"     │
    │                                                          │
    │  This repo starts with waves and asks:                   │
    │    "How do we capture and decompose them?"               │
    │                                                          │
    │  Same question. Different domain.                        │
    │  Same answer: exploit the structure of representation.   │
    └──────────────────────────────────────────────────────────┘

    NEXT: Now that we can capture waves as discrete samples,
    how do we find out WHICH frequencies are in them?
    That's the Fourier Transform → fourier_transform.py
    """)


# =============================================================================
# RUN EVERYTHING
# =============================================================================

if __name__ == "__main__":
    print("\n" + "█" * 72)
    print("█  SIGNALS AND SAMPLING                                              █")
    print("█  The bridge between the continuous world and digital representation █")
    print("█" * 72)

    part1_what_is_a_signal()
    part2_sampling_theorem()
    part3_aliasing()
    part4_quantization()
    part5_reconstruction()
    part6_adc_dac()
    part7_philosophy()
