# Bit Tricks and Wave Functions

**How a French mathematician's 200-year-old insight connects signal processing, compression, quantum computing, and the structure of information itself.**

```
X[k] = Σ x[n] · e^(−i·2π·k·n/N)     — "Any signal is a sum of spinning arrows."
```

This repo picks up where [Exploring: From FISR to Conway and Beyond](https://github.com/c0re-i5/exploring-from-fisr-to-conway-and-beyond) left off. That repo showed:

> **The representation of data has structure. That structure is a tool.**

This repo crosses from discrete bits into the continuous world and adds a second law:

> **Every signal is a sum of waves. Decompose it right, and the impossible becomes trivial.**

Bit tricks exploit the structure of binary representation. Wave transforms exploit the structure of frequency decomposition. They meet at the FFT, at JPEG, and at quantum computing.

Each script is self-contained and meant to be read as much as run. `numpy` is the only dependency.

## The Scripts

Run them in order for the best experience, or jump to whatever catches your eye.

### 1. [`signals_and_sampling.py`](signals_and_sampling.py) — The Analog Bridge

Where the continuous world becomes discrete — and why that works.

**Topics:** sine waves as building blocks · sampling theorem (Nyquist-Shannon) · aliasing · quantization and bit depth · sinc interpolation · the ADC/DAC pipeline

### 2. [`fourier_transform.py`](fourier_transform.py) — Seeing Frequencies

The most important algorithm most programmers never learned.

**Topics:** Euler's formula · the DFT by hand · Cooley-Tukey FFT · bit-reversal permutation · spectral analysis · convolution theorem · windowing · Parseval's theorem

### 3. [`wavelets_and_uncertainty.py`](wavelets_and_uncertainty.py) — Having It Both Ways

Fourier tells you *what* frequencies exist. Wavelets tell you *when* they happen.

**Topics:** Heisenberg-Gabor uncertainty · Short-Time Fourier Transform · Haar wavelet · multiresolution analysis · wavelet denoising · signal compression

### 4. [`practical_wave_applications.py`](practical_wave_applications.py) — Tools That Work

Six benchmarked, working implementations built from the principles in the first three scripts.

| # | Application | What it demonstrates |
|---|---|---|
| 1 | JPEG-style compression | DCT + quantization — **85% size reduction** |
| 2 | Audio synthesizer | Additive & FM synthesis from pure math |
| 3 | Spectral noise filter | FFT denoise — recover signals buried in noise |
| 4 | Template matching | FFT correlation — **50× faster** than brute force |
| 5 | Spectrogram | Time-frequency ASCII art visualization |
| 6 | Digital filters | FIR low-pass, high-pass, band-pass |

### 5. [`quantum_wave_functions.py`](quantum_wave_functions.py) — When Waves Compute

Classical bits are 0 or 1. Qubits are waves that interfere.

**Topics:** qubits and superposition · quantum gates · entanglement · Quantum Fourier Transform · Shor's algorithm (period finding) · Grover's algorithm (amplitude amplification) · quantum error correction

### 6. [`the_full_spectrum.py`](the_full_spectrum.py) — The Capstone

Chains five techniques into one end-to-end pipeline that samples, analyzes, compresses, corrupts, denoises, and perfectly recovers a signal.

```
Signal → Sample (Nyquist) → FFT analyze → DCT compress → Add noise
    → Wavelet denoise → Reconstruct → ✓ Original signal recovered
```

Every stage pulls from a different chapter of this repo, demonstrating how these techniques compose in the real world — the same pipeline your phone runs when you stream music.

### Visual Guide

[`visual_guide.ipynb`](visual_guide.ipynb) — A Jupyter notebook with matplotlib plots for every major concept: sampling & aliasing, Fourier spectra, wavelet decomposition, spectrograms, Grover's algorithm, and the full pipeline. Run it alongside the scripts for a richer visual understanding.

### Reference

[`euler.txt`](euler.txt) — Euler's formula and why it's the bridge between bits and waves.

## Running

Python 3.10+ and `numpy` required. Add `matplotlib` for the visual guide notebook.

```bash
pip install numpy matplotlib
python signals_and_sampling.py
python fourier_transform.py
python wavelets_and_uncertainty.py
python practical_wave_applications.py
python quantum_wave_functions.py
python the_full_spectrum.py
```

Pre-generated output is in the [`output/`](output/) folder if you just want to read the results.

## The Connection Map

Every topic traces back to the same starting point:

```
Euler's e^(iθ) = cos(θ) + i·sin(θ)
    │
    ├── Signal decomposition (Fourier)
    │     ├── FFT — O(n²) → O(n log n)
    │     ├── Spectral analysis (find hidden frequencies)
    │     └── Convolution theorem (filter in frequency domain)
    │
    ├── The uncertainty principle
    │     ├── Gabor limit (Δt · Δf ≥ ½)
    │     ├── Short-Time Fourier Transform
    │     └── Wavelets (multiresolution analysis)
    │
    ├── Practical compression & processing
    │     ├── DCT → JPEG, MP3           ├── Spectral filtering
    │     ├── Edge detection             └── Huffman + DCT = JPEG  [← repo 1]
    │
    ├── Quantum wave functions
    │     ├── Superposition (qubits as waves)
    │     ├── Quantum Fourier Transform
    │     ├── Shor's algorithm (period finding)
    │     └── Grover's algorithm (amplitude amplification)
    │
    └── Bridge to Bit Tricks  [← repo 1]
          ├── ADC/DAC (continuous ↔ discrete)
          ├── FFT butterfly = bit-reversal permutation
          ├── Hamming codes ↔ quantum error correction
          └── Shannon entropy ↔ spectral energy
```

## License

[MIT](LICENSE)
