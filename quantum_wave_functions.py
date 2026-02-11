"""
=============================================================================
  QUANTUM WAVE FUNCTIONS — When Waves Compute
  Classical bits are 0 or 1. Qubits are waves that interfere.
=============================================================================

  In the classical world, a bit is either 0 or 1.
  In the quantum world, a qubit is BOTH at once — it exists as
  a WAVE FUNCTION that encodes complex amplitudes for each state.

  When qubits interact, their wave functions INTERFERE —
  constructively reinforcing correct answers and destructively
  canceling wrong ones. This is quantum computing.

  This script builds a qubit simulator from scratch using numpy
  and walks through the key algorithms that exploit wave
  interference for computation.
"""

import math
import cmath
import numpy as np

SEPARATOR = "=" * 72
SUBSEP = "─" * 72


# =============================================================================
# THE QUBIT SIMULATOR
# =============================================================================

class QubitRegister:
    """
    A simple quantum register simulator.

    The state of n qubits is a complex vector of length 2^n.
    Each entry is the probability amplitude for that basis state.
    Quantum gates are unitary matrices applied to this vector.

    This is exponentially expensive classically (2^n amplitudes),
    which is exactly WHY quantum computers are powerful —
    they maintain this state in actual physics.
    """

    def __init__(self, n_qubits):
        self.n = n_qubits
        self.size = 2 ** n_qubits
        # Initialize to |000...0⟩
        self.state = np.zeros(self.size, dtype=complex)
        self.state[0] = 1.0 + 0j

    def reset(self):
        """Reset to |000...0⟩."""
        self.state = np.zeros(self.size, dtype=complex)
        self.state[0] = 1.0 + 0j

    def probabilities(self):
        """Return measurement probabilities for each basis state."""
        return np.abs(self.state) ** 2

    def measure(self):
        """
        Simulate a measurement — collapse the wave function.
        Returns the observed basis state as an integer.
        """
        probs = self.probabilities()
        result = np.random.choice(self.size, p=probs)
        # Collapse: after measurement, the state IS the result
        self.state = np.zeros(self.size, dtype=complex)
        self.state[result] = 1.0
        return result

    def apply_single(self, gate, target):
        """
        Apply a single-qubit gate to the target qubit.

        Uses the tensor product structure:
            Full gate = I ⊗ ... ⊗ gate ⊗ ... ⊗ I

        But we compute it efficiently without building the full matrix.
        """
        new_state = np.zeros_like(self.state)
        for i in range(self.size):
            # Bit 'target' of state index i
            bit = (i >> target) & 1
            # The partner state (with bit flipped)
            partner = i ^ (1 << target)
            if bit == 0:
                new_state[i] += gate[0, 0] * self.state[i] + \
                                gate[0, 1] * self.state[partner]
            else:
                new_state[i] += gate[1, 0] * self.state[partner] + \
                                gate[1, 1] * self.state[i]
        self.state = new_state

    def apply_controlled(self, gate, control, target):
        """
        Apply a controlled single-qubit gate.
        The gate acts on 'target' only when 'control' is |1⟩.
        """
        new_state = self.state.copy()
        for i in range(self.size):
            ctrl_bit = (i >> control) & 1
            tgt_bit = (i >> target) & 1
            if ctrl_bit == 1:
                partner = i ^ (1 << target)
                if tgt_bit == 0:
                    new_state[i] = gate[0, 0] * self.state[i] + \
                                   gate[0, 1] * self.state[partner]
                else:
                    new_state[i] = gate[1, 0] * self.state[partner] + \
                                   gate[1, 1] * self.state[i]
        self.state = new_state

    def swap(self, q1, q2):
        """Swap two qubits."""
        new_state = np.zeros_like(self.state)
        for i in range(self.size):
            b1 = (i >> q1) & 1
            b2 = (i >> q2) & 1
            if b1 != b2:
                j = i ^ (1 << q1) ^ (1 << q2)
                new_state[j] = self.state[i]
            else:
                new_state[i] = self.state[i]
        self.state = new_state

    def state_str(self, threshold=0.001):
        """Pretty-print the quantum state."""
        terms = []
        for i in range(self.size):
            amp = self.state[i]
            prob = abs(amp) ** 2
            if prob > threshold:
                basis = f"|{i:0{self.n}b}⟩"
                if abs(amp.imag) < 1e-10:
                    terms.append(f"{amp.real:+.4f}{basis}")
                else:
                    terms.append(f"({amp:.3f}){basis}")
        return " ".join(terms)


# ── Standard gates ───────────────────────────────────────────────────────────

# Pauli-X (NOT gate): flips |0⟩ ↔ |1⟩
X = np.array([[0, 1], [1, 0]], dtype=complex)

# Pauli-Z: flips phase of |1⟩
Z = np.array([[1, 0], [0, -1]], dtype=complex)

# Hadamard: creates equal superposition
H = np.array([[1, 1], [1, -1]], dtype=complex) / math.sqrt(2)

# Phase gate: adds a phase to |1⟩
def phase_gate(theta):
    return np.array([[1, 0], [0, cmath.exp(1j * theta)]], dtype=complex)

# Rotation gates
def Rz(theta):
    """Rotation around Z-axis."""
    return np.array([
        [cmath.exp(-1j * theta / 2), 0],
        [0, cmath.exp(1j * theta / 2)]
    ], dtype=complex)


# =============================================================================
# PART 1: CLASSICAL BITS → QUBITS
# =============================================================================

def part1_qubits():
    print(f"\n{SEPARATOR}")
    print("  PART 1: FROM BITS TO QUBITS")
    print("  0 and 1 become waves")
    print(SEPARATOR)

    print("""
    A classical bit is either 0 or 1.
    A qubit is BOTH simultaneously, with complex amplitudes:

    ┌──────────────────────────────────────────────────────────┐
    │  |ψ⟩ = α|0⟩ + β|1⟩                                      │
    │                                                          │
    │  α, β are COMPLEX NUMBERS (amplitudes)                   │
    │  |α|² = probability of measuring 0                       │
    │  |β|² = probability of measuring 1                       │
    │  |α|² + |β|² = 1  (probabilities sum to 1)              │
    │                                                          │
    │  Before measurement: the qubit is a WAVE FUNCTION         │
    │  After measurement: it COLLAPSES to 0 or 1               │
    └──────────────────────────────────────────────────────────┘

    The key: α and β are COMPLEX. They have both magnitude AND
    phase. The phase is what enables quantum interference.
    """)

    qr = QubitRegister(1)

    # |0⟩ state
    print(f"    Initial state: {qr.state_str()}")
    print(f"    Probabilities: P(0)={abs(qr.state[0])**2:.4f}, "
          f"P(1)={abs(qr.state[1])**2:.4f}")
    print()

    # Apply Hadamard → equal superposition
    qr.apply_single(H, 0)
    print(f"    After Hadamard: {qr.state_str()}")
    print(f"    Probabilities:  P(0)={abs(qr.state[0])**2:.4f}, "
          f"P(1)={abs(qr.state[1])**2:.4f}")
    print()

    # Measure many times to show probability
    print("    Measuring 10,000 times:\n")
    counts = {0: 0, 1: 0}
    for _ in range(10000):
        qr.reset()
        qr.apply_single(H, 0)
        result = qr.measure()
        counts[result] += 1

    for state, count in sorted(counts.items()):
        bar = "█" * (count // 200)
        print(f"      |{state}⟩: {count:>5} ({count/100:.1f}%) {bar}")

    print("""
    50/50 — the Hadamard gate creates PERFECT superposition.

    This is NOT randomness. The qubit IS in both states
    simultaneously, with DEFINITE amplitudes. The randomness
    only appears when we measure — until then, the wave
    function evolves deterministically.

    ► Connection to Fourier: a qubit in superposition is like
      a signal with two frequency components. The amplitudes
      are like Fourier coefficients. Measurement is like
      sampling — it collapses the wave to a single value.
    """)


# =============================================================================
# PART 2: QUANTUM GATES
# =============================================================================

def part2_gates():
    print(f"\n{SEPARATOR}")
    print("  PART 2: QUANTUM GATES — ROTATING WAVES")
    print("  All computation as unitary matrix operations")
    print(SEPARATOR)

    print("""
    Classical gates: AND, OR, NOT, XOR → irreversible
    Quantum gates: unitary matrices → REVERSIBLE and LOSSLESS

    Every quantum gate is a ROTATION in complex vector space.
    This is Euler's formula at work: e^(iθ) rotates a complex number.

    ┌──────────────────────────────────────────────────────────┐
    │  KEY QUANTUM GATES                                       │
    │                                                          │
    │  Hadamard (H):  Creates superposition                    │
    │    |0⟩ → (|0⟩+|1⟩)/√2    |1⟩ → (|0⟩-|1⟩)/√2           │
    │                                                          │
    │  Pauli-X:  Quantum NOT (bit flip)                        │
    │    |0⟩ → |1⟩    |1⟩ → |0⟩                               │
    │                                                          │
    │  Pauli-Z:  Phase flip                                    │
    │    |0⟩ → |0⟩    |1⟩ → -|1⟩                              │
    │                                                          │
    │  CNOT:  Controlled-NOT (entanglement creator)            │
    │    |00⟩ → |00⟩  |01⟩ → |01⟩  |10⟩ → |11⟩  |11⟩ → |10⟩ │
    │    (flips target when control=1, like classical XOR!)    │
    └──────────────────────────────────────────────────────────┘
    """)

    # Demonstrate gate effects
    gates = [
        ("Pauli-X (NOT)", X),
        ("Pauli-Z (phase flip)", Z),
        ("Hadamard", H),
    ]

    for name, gate in gates:
        print(f"    Gate: {name}")
        print(f"    Matrix: [{gate[0,0]:.3f}  {gate[0,1]:.3f}]")
        print(f"            [{gate[1,0]:.3f}  {gate[1,1]:.3f}]")

        for init, init_name in [(0, "|0⟩"), (1, "|1⟩")]:
            qr = QubitRegister(1)
            if init == 1:
                qr.apply_single(X, 0)
            qr.apply_single(gate, 0)
            print(f"      {init_name} → {qr.state_str()}")
        print()

    # Demonstrate CNOT
    print("    CNOT (Controlled-NOT):")
    print("    Control qubit 1, Target qubit 0\n")

    for init in range(4):
        qr = QubitRegister(2)
        if init & 1:
            qr.apply_single(X, 0)
        if init & 2:
            qr.apply_single(X, 1)
        input_state = qr.state_str()
        qr.apply_controlled(X, 1, 0)  # CNOT: control=1, target=0
        print(f"      {input_state}  →  {qr.state_str()}")

    print("""
    ► CNOT is the quantum XOR! Same truth table as classical XOR.
    ► Connection to repo 1: XOR was the foundation of Gray codes,
      Zobrist hashing, and CRC. In quantum computing, CNOT (quantum XOR)
      is the foundation of entanglement and error correction.
    """)


# =============================================================================
# PART 3: ENTANGLEMENT
# =============================================================================

def part3_entanglement():
    print(f"\n{SEPARATOR}")
    print("  PART 3: ENTANGLEMENT — CORRELATED WAVES")
    print("  The quantum resource that has no classical equivalent")
    print(SEPARATOR)

    print("""
    Entanglement: two qubits become CORRELATED so that measuring
    one instantly determines the other, regardless of distance.

    The simplest entangled state is the BELL STATE:

        |Φ+⟩ = (|00⟩ + |11⟩) / √2

    Measure qubit 0 → get 0 with 50% probability → qubit 1 is ALSO 0
    Measure qubit 0 → get 1 with 50% probability → qubit 1 is ALSO 1

    The outcomes are PERFECTLY correlated. Always. This isn't
    pre-arranged — it's a fundamental property of quantum mechanics.

    Recipe for Bell state: H on qubit 1, then CNOT(control=1, target=0):
    """)

    qr = QubitRegister(2)
    print(f"    Start:        {qr.state_str()}")

    qr.apply_single(H, 1)
    print(f"    After H(q1):  {qr.state_str()}")

    qr.apply_controlled(X, 1, 0)
    print(f"    After CNOT:   {qr.state_str()}")
    print()

    # Verify correlation by measuring many times
    print("    Measuring 10,000 times:\n")
    results = {(0, 0): 0, (0, 1): 0, (1, 0): 0, (1, 1): 0}
    for _ in range(10000):
        qr.reset()
        qr.apply_single(H, 1)
        qr.apply_controlled(X, 1, 0)
        outcome = qr.measure()
        b0 = outcome & 1
        b1 = (outcome >> 1) & 1
        results[(b1, b0)] += 1

    for (b1, b0), count in sorted(results.items()):
        bar = "█" * (count // 200)
        print(f"      |{b1}{b0}⟩: {count:>5} ({count/100:.1f}%) {bar}")

    print("""
    |00⟩ and |11⟩ each appear ~50%. |01⟩ and |10⟩ NEVER appear.
    The qubits are perfectly correlated — entangled.

    This has no classical explanation. There's no hidden variable
    that pre-determines the outcome (Bell's theorem proves this).

    Entanglement is used in:
      • Quantum teleportation (transfer quantum states)
      • Quantum key distribution (unhackable encryption)
      • Quantum error correction (protect quantum data)
      • Superdense coding (2 classical bits per qubit)

    ► Connection to Fourier: entanglement creates CORRELATIONS
      between qubits, just as Fourier analysis reveals correlations
      between frequency components of a signal.
    """)


# =============================================================================
# PART 4: QUANTUM FOURIER TRANSFORM
# =============================================================================

def part4_qft():
    print(f"\n{SEPARATOR}")
    print("  PART 4: THE QUANTUM FOURIER TRANSFORM")
    print("  The classical FFT, made quantum")
    print(SEPARATOR)

    print("""
    The QFT is the quantum version of the DFT:

    ┌──────────────────────────────────────────────────────────┐
    │  CLASSICAL DFT:                                          │
    │    X[k] = Σ x[n] · e^(-i2πkn/N)                         │
    │    Input: N complex numbers → Output: N complex numbers  │
    │    Cost: O(N log N) via FFT                              │
    │                                                          │
    │  QUANTUM FOURIER TRANSFORM:                              │
    │    |x⟩ → (1/√N) Σ e^(i2πxk/N) |k⟩                      │
    │    Input: n qubits → Output: n qubits (superposition!)  │
    │    Cost: O(n²) gates where N = 2^n                       │
    │         That's O(log²N) — EXPONENTIALLY FASTER            │
    └──────────────────────────────────────────────────────────┘

    The QFT circuit:
    For each qubit j (from MSB to LSB):
      1. Apply Hadamard to qubit j
      2. Apply controlled phase rotations from higher qubits
      3. Swap qubits at the end (bit reversal!)
    """)

    def apply_qft(qr, qubits):
        """
        Apply the Quantum Fourier Transform to the specified qubits.

        This is the quantum circuit that implements the DFT
        on the amplitudes of the quantum state.
        """
        n = len(qubits)
        for j in range(n - 1, -1, -1):
            # Hadamard on qubit j
            qr.apply_single(H, qubits[j])

            # Controlled phase rotations
            for k in range(j - 1, -1, -1):
                angle = math.pi / (2 ** (j - k))
                qr.apply_controlled(phase_gate(angle), qubits[k], qubits[j])

        # Reverse qubit order (bit reversal!)
        for i in range(n // 2):
            qr.swap(qubits[i], qubits[n - 1 - i])

    # Demonstrate QFT on 3 qubits
    n_qubits = 3
    N = 2 ** n_qubits  # 8 states

    print("  ── QFT on 3 qubits ──\n")

    # Input: |3⟩ = |011⟩
    test_input = 3
    qr = QubitRegister(n_qubits)
    # Prepare |3⟩ by applying X to bits that should be 1
    for bit in range(n_qubits):
        if (test_input >> bit) & 1:
            qr.apply_single(X, bit)

    print(f"    Input state: |{test_input}⟩ = |{test_input:0{n_qubits}b}⟩")
    print(f"    State: {qr.state_str()}")
    print()

    # Apply QFT
    apply_qft(qr, list(range(n_qubits)))

    print(f"    After QFT:")
    print(f"    State: {qr.state_str()}")
    print(f"\n    Probabilities:")
    probs = qr.probabilities()
    for k in range(N):
        bar = "█" * int(probs[k] * 50)
        phase = cmath.phase(qr.state[k]) if abs(qr.state[k]) > 0.01 else 0
        print(f"      |{k:0{n_qubits}b}⟩: {probs[k]:.4f}  "
              f"phase={phase/math.pi:.2f}π  {bar}")

    print("""
    After QFT, all states have EQUAL probability (1/8 each),
    but DIFFERENT PHASES. The information is encoded in the phases!

    This is fundamental: the QFT spreads amplitude equally but
    arranges phases to encode the input. Quantum algorithms then
    use INTERFERENCE to extract useful phase information.

    ► Connection to FFT: the QFT does EXACTLY what the classical
      FFT does, but on quantum amplitudes. The bit-reversal
      at the end is the same bit-reversal permutation from
      fourier_transform.py — now implemented with qubit swaps!
    """)

    # Verify against classical DFT
    print("  ── Verification: QFT vs classical DFT ──\n")

    # Classical DFT of basis vector |3⟩
    classical_input = np.zeros(N, dtype=complex)
    classical_input[test_input] = 1.0
    classical_dft = np.fft.fft(classical_input) / math.sqrt(N)

    print(f"    {'State':>8} {'QFT amplitude':>20} {'Classical DFT':>20} {'Match':>7}")
    print(f"    {'─'*8} {'─'*20} {'─'*20} {'─'*7}")
    for k in range(N):
        q_amp = qr.state[k]
        c_amp = classical_dft[k]
        match = "✓" if abs(q_amp - c_amp) < 0.001 else "✗"
        print(f"    |{k:0{n_qubits}b}⟩  {q_amp.real:+.4f}{q_amp.imag:+.4f}i"
              f"    {c_amp.real:+.4f}{c_amp.imag:+.4f}i  {match:>5}")

    print("""
    ✓ Perfect match! The QFT and classical DFT produce identical
      results. But the QFT does it in O(n²) = O(log²N) gates
      instead of O(N log N) operations — exponentially faster
      for large N.
    """)


# =============================================================================
# PART 5: GROVER'S ALGORITHM — SEARCH BY WAVE INTERFERENCE
# =============================================================================

def part5_grover():
    print(f"\n{SEPARATOR}")
    print("  PART 5: GROVER'S ALGORITHM — SEARCH BY INTERFERENCE")
    print("  Finding a needle in a haystack with wave amplification")
    print(SEPARATOR)

    print("""
    Problem: search an unsorted database of N items for one target.
    Classical: O(N) — check each item.
    Grover:    O(√N) — amplify the target's probability.

    The trick is AMPLITUDE AMPLIFICATION through interference:

    1. Start with equal superposition (all items equally likely)
    2. MARK the target (flip its amplitude's sign)
    3. REFLECT about the mean amplitude
    4. Repeat ~√N times
    5. The target's probability grows while others shrink

    This is constructive interference for the answer and
    destructive interference for everything else.
    """)

    def grover_search(n_qubits, target):
        """
        Run Grover's algorithm to find 'target' among 2^n states.
        Returns the measurement result.
        """
        N = 2 ** n_qubits
        optimal_iterations = max(1, round(math.pi / 4 * math.sqrt(N)))

        qr = QubitRegister(n_qubits)

        # Step 1: Create equal superposition
        for q in range(n_qubits):
            qr.apply_single(H, q)

        # Steps 2-3: Grover iterations
        for iteration in range(optimal_iterations):
            # Oracle: flip the sign of |target⟩
            qr.state[target] *= -1

            # Diffusion operator: reflect about the mean
            # 2|s⟩⟨s| - I where |s⟩ is uniform superposition
            mean_amp = np.mean(qr.state)
            qr.state = 2 * mean_amp - qr.state

        return qr, optimal_iterations

    # Demo with 4 qubits (search space = 16)
    n_qubits = 4
    N = 2 ** n_qubits
    target = 11  # We're looking for |1011⟩

    print(f"    Search space: {N} items ({n_qubits} qubits)")
    print(f"    Target: |{target:0{n_qubits}b}⟩ = {target}")
    print()

    qr, n_iterations = grover_search(n_qubits, target)

    print(f"    Grover iterations: {n_iterations} (≈ π/4 · √{N} = "
          f"{math.pi/4*math.sqrt(N):.1f})")
    print()

    probs = qr.probabilities()
    print("    Final probabilities:")
    for i in range(N):
        bar_len = int(probs[i] * 50)
        marker = " ← TARGET" if i == target else ""
        bar = "█" * bar_len
        if probs[i] > 0.01 or i == target:
            print(f"      |{i:0{n_qubits}b}⟩ ({i:>2}): "
                  f"{probs[i]:.4f} {bar}{marker}")

    print()

    # Verify by repeated measurement
    print("    Measuring 1000 times:")
    counts = {}
    for _ in range(1000):
        qr_trial, _ = grover_search(n_qubits, target)
        result = qr_trial.measure()
        counts[result] = counts.get(result, 0) + 1

    success = counts.get(target, 0)
    print(f"      Found target: {success}/1000 ({success/10:.1f}%)")
    print(f"      Classical would need ~{N//2} checks on average")
    print(f"      Grover needs ~{n_iterations} iterations")

    print("""
    With just 3 iterations, Grover's algorithm finds the target
    with ~96% probability in a space of 16 items.

    For N = 1,000,000 items:
      Classical: ~500,000 checks
      Grover:    ~785 iterations    (640× speedup)

    For N = 10^18 (a quintillion):
      Classical: ~5×10^17 checks
      Grover:    ~10^9 iterations   (a billion-fold speedup)

    ┌──────────────────────────────────────────────────────────┐
    │  HOW IT WORKS — WAVE INTERFERENCE                        │
    │                                                          │
    │  The oracle INVERTS the target amplitude (flip sign).    │
    │  The diffusion operator REFLECTS everything about the    │
    │  mean, which pushes the target amplitude UP and all      │
    │  others DOWN.                                            │
    │                                                          │
    │  This is CONSTRUCTIVE INTERFERENCE for the right answer  │
    │  and DESTRUCTIVE INTERFERENCE for wrong answers.         │
    │                                                          │
    │  Same physics as noise-canceling headphones:             │
    │  create a wave that cancels what you don't want and      │
    │  reinforces what you do.                                 │
    └──────────────────────────────────────────────────────────┘
    """)


# =============================================================================
# PART 6: SHOR'S ALGORITHM — PERIOD FINDING
# =============================================================================

def part6_shor():
    print(f"\n{SEPARATOR}")
    print("  PART 6: SHOR'S ALGORITHM (SIMPLIFIED)")
    print("  Breaking RSA with quantum period finding")
    print(SEPARATOR)

    print("""
    Shor's algorithm factors large numbers in polynomial time.
    RSA encryption depends on factoring being HARD.
    A quantum computer running Shor's renders RSA obsolete.

    The key insight: factoring reduces to PERIOD FINDING.
    Period finding reduces to the QUANTUM FOURIER TRANSFORM.

    ┌──────────────────────────────────────────────────────────┐
    │  SHOR'S ALGORITHM (simplified)                           │
    │                                                          │
    │  To factor N:                                            │
    │  1. Pick random a < N                                    │
    │  2. Find the PERIOD r of f(x) = a^x mod N               │
    │     (this is the hard part — QFT makes it fast)          │
    │  3. If r is even, compute gcd(a^(r/2) ± 1, N)           │
    │  4. With high probability, this gives a factor of N      │
    └──────────────────────────────────────────────────────────┘

    Let's demonstrate the classical part with a small example:
    """)

    def find_period_classical(a, N):
        """Find the period of a^x mod N (brute force)."""
        x = 1
        power = a % N
        while power != 1 and x < N:
            power = (power * a) % N
            x += 1
        return x if power == 1 else None

    # Factor 15 (= 3 × 5)
    N = 15
    print(f"    Factoring N = {N}")
    print()

    # Try several random bases
    tried = []
    for a in [2, 4, 7, 8, 11, 13]:
        if math.gcd(a, N) > 1:
            print(f"    a = {a:>2}: gcd({a}, {N}) = {math.gcd(a, N)} "
                  f"→ trivial factor!")
            continue

        r = find_period_classical(a, N)
        if r is None:
            continue

        # Show the sequence
        sequence = [pow(a, x, N) for x in range(r + 2)]
        print(f"    a = {a:>2}: sequence = {sequence[:r]}  → period r = {r}")

        if r % 2 == 0:
            x = pow(a, r // 2, N)
            f1 = math.gcd(x - 1, N)
            f2 = math.gcd(x + 1, N)
            if f1 > 1 and f1 < N:
                print(f"          r is even → a^(r/2) = {x}")
                print(f"          gcd({x}-1, {N}) = {f1},  "
                      f"gcd({x}+1, {N}) = {f2}")
                if f1 * f2 == N:
                    print(f"          ✓ FACTORS: {N} = {f1} × {f2}")
                tried.append((a, r, f1, f2))
            else:
                print(f"          (trivial factor)")
        else:
            print(f"          r is odd → try another a")
        print()

    print("""
    On a QUANTUM computer, step 2 (period finding) is where the
    QFT gives an exponential speedup:

    Classical period finding: O(√N) at best
    Quantum (Shor's):        O((log N)³)

    For RSA-2048 (a 2048-bit key):
      Classical: ~2^1024 operations (heat death of universe)
      Quantum:   ~2048³ ≈ 10^10 operations (minutes)

    This is why quantum computing is an existential threat to
    current encryption, and why post-quantum cryptography is
    being developed RIGHT NOW.

    ► Connection to Fourier: Shor's algorithm literally uses the
      QFT to find the period of a modular function — the SAME
      operation as finding the frequency of a signal, just applied
      to number theory instead of audio processing.
    """)


# =============================================================================
# PART 7: QUANTUM ERROR CORRECTION
# =============================================================================

def part7_error_correction():
    print(f"\n{SEPARATOR}")
    print("  PART 7: QUANTUM ERROR CORRECTION")
    print("  Hamming codes for qubits")
    print(SEPARATOR)

    print("""
    ┌──────────────────────────────────────────────────────────┐
    │  CONNECTION TO REPO 1: HAMMING → QUANTUM ERROR CODES     │
    │                                                          │
    │  In repo 1's deeper_foundations.py, we built Hamming     │
    │  codes: place parity bits at power-of-2 positions to     │
    │  detect and correct single-bit errors.                   │
    │                                                          │
    │  Quantum error correction is the same idea, but HARDER:  │
    │                                                          │
    │  1. You CAN'T copy a qubit (no-cloning theorem)          │
    │  2. Measuring a qubit DESTROYS the superposition         │
    │  3. Errors are CONTINUOUS (not just bit flips)            │
    │                                                          │
    │  Despite these obstacles, quantum error correction WORKS  │
    │  by spreading quantum information across multiple qubits. │
    └──────────────────────────────────────────────────────────┘

    The simplest quantum error correction: the 3-QUBIT BIT-FLIP CODE.
    Encode 1 logical qubit into 3 physical qubits:

        |0⟩ → |000⟩ (logical zero)
        |1⟩ → |111⟩ (logical one)

    If one qubit flips, majority vote corrects it.
    """)

    def demo_bit_flip_code(initial_state, error_qubit):
        """Demonstrate the 3-qubit bit-flip error correction code."""
        qr = QubitRegister(3)

        # Encode: prepare initial state on qubit 2
        if initial_state == 1:
            qr.apply_single(X, 2)

        # Spread with CNOTs: |ψ⟩ → |ψψψ⟩
        qr.apply_controlled(X, 2, 1)
        qr.apply_controlled(X, 2, 0)
        encoded = qr.state_str()

        # Introduce error: flip one qubit
        if error_qubit >= 0:
            qr.apply_single(X, error_qubit)
        errored = qr.state_str()

        # Syndrome measurement: check parity
        # In a real quantum computer, this uses ancilla qubits.
        # Here we check parities directly.
        probs = qr.probabilities()
        most_likely = np.argmax(probs)
        b0 = most_likely & 1
        b1 = (most_likely >> 1) & 1
        b2 = (most_likely >> 2) & 1

        # Majority vote to find error
        if b0 == b1 == b2:
            correction = -1  # no error
        elif b0 == b1:
            correction = 2  # qubit 2 flipped
        elif b0 == b2:
            correction = 1  # qubit 1 flipped
        else:
            correction = 0  # qubit 0 flipped

        # Apply correction
        if correction >= 0:
            qr.apply_single(X, correction)
        corrected = qr.state_str()

        # Decode: undo entanglement
        qr.apply_controlled(X, 2, 0)
        qr.apply_controlled(X, 2, 1)

        # Read out logical qubit (qubit 2)
        decoded_probs = qr.probabilities()
        logical_0_prob = sum(decoded_probs[i] for i in range(8)
                             if not (i >> 2) & 1)
        logical_1_prob = sum(decoded_probs[i] for i in range(8)
                             if (i >> 2) & 1)
        logical_result = 1 if logical_1_prob > logical_0_prob else 0

        return {
            'encoded': encoded,
            'errored': errored,
            'corrected': corrected,
            'error_qubit': error_qubit,
            'correction': correction,
            'logical_result': logical_result,
            'success': logical_result == initial_state
        }

    print("  ── 3-Qubit Bit-Flip Code Demo ──\n")
    print(f"    {'Input':>6} {'Error on':>10} {'Corrected to':>15} {'Result':>8} {'Status':>8}")
    print(f"    {'─'*6} {'─'*10} {'─'*15} {'─'*8} {'─'*8}")

    for init in [0, 1]:
        for err in [-1, 0, 1, 2]:
            r = demo_bit_flip_code(init, err)
            err_str = f"qubit {err}" if err >= 0 else "none"
            init_str = f"|{init}⟩"
            result_str = f"|{r['logical_result']}⟩"
            status = "✓" if r['success'] else "✗"
            print(f"    {init_str:>6} {err_str:>10} "
                  f"{r['corrected']:>15} {result_str:>8} {status:>8}")

    print("""
    Every single-qubit error is corrected! This is the quantum
    version of the repetition code — the simplest error correction.

    Real quantum computers use more sophisticated codes:

      ┌────────────────────┬──────────┬──────────────────────┐
      │ Code               │ Qubits   │ Corrects             │
      ├────────────────────┼──────────┼──────────────────────┤
      │ 3-qubit repetition │ 3 for 1  │ 1 bit-flip error     │
      │ Shor code          │ 9 for 1  │ 1 arbitrary error    │
      │ Steane code        │ 7 for 1  │ 1 arbitrary error    │
      │ Surface code       │ ~1000    │ Many errors (google's)│
      └────────────────────┴──────────┴──────────────────────┘

    ► The Steane code is literally Hamming [7,4] from repo 1,
      extended to the quantum domain! The parity check matrix
      is the SAME — it's Hamming's insight, protecting wave
      functions instead of bit strings.
    """)


# =============================================================================
# PART 8: THE PHILOSOPHY
# =============================================================================

def part8_philosophy():
    print(f"\n{SEPARATOR}")
    print("  CONCLUSION: THE WAVE NATURE OF COMPUTATION")
    print(SEPARATOR)

    print("""
    This script has shown that quantum computing is not about
    "trying all answers at once." It's about WAVE INTERFERENCE:

    ┌──────────────────────────────────────────────────────────┐
    │  Constructive interference → amplify correct answers     │
    │  Destructive interference → suppress wrong answers       │
    │                                                          │
    │  This is the SAME physics as:                            │
    │    Noise-canceling headphones (destructive for noise)    │
    │    Radio tuning (constructive for desired station)       │
    │    Laser coherence (constructive for one wavelength)     │
    │    FFT spectral peaks (constructive at true frequencies) │
    └──────────────────────────────────────────────────────────┘

    THE FULL ARC OF BOTH REPOS:

    ┌──────────────────────────────────────────────────────────┐
    │  LEVEL 1 — Bits (Repo 1)                                 │
    │    Representation: binary integers, IEEE 754              │
    │    Exploit: bit manipulation (XOR, shift, mask)          │
    │    Applications: hash tables, error correction, PRNG     │
    │                                                          │
    │  LEVEL 2 — Waves (This repo, scripts 1-4)               │
    │    Representation: frequency decomposition                │
    │    Exploit: Fourier Transform, wavelets                  │
    │    Applications: compression, filtering, synthesis       │
    │                                                          │
    │  LEVEL 3 — Quantum waves (This script)                   │
    │    Representation: complex amplitudes (wave functions)    │
    │    Exploit: interference (constructive/destructive)       │
    │    Applications: search, factoring, simulation           │
    │                                                          │
    │  Each level subsumes the previous.                        │
    │  Each level exploits REPRESENTATION in a deeper way.     │
    │  The unifying principle is always the same:              │
    │                                                          │
    │     "The representation of data has structure.            │
    │      That structure is a tool."                           │
    └──────────────────────────────────────────────────────────┘

    NEXT: Let's chain everything together in one pipeline.
    → the_full_spectrum.py
    """)


# =============================================================================
# RUN EVERYTHING
# =============================================================================

if __name__ == "__main__":
    print("\n" + "█" * 72)
    print("█  QUANTUM WAVE FUNCTIONS                                            █")
    print("█  When waves compute: qubits, interference, and quantum algorithms  █")
    print("█" * 72)

    np.random.seed(2026)

    part1_qubits()
    part2_gates()
    part3_entanglement()
    part4_qft()
    part5_grover()
    part6_shor()
    part7_error_correction()
    part8_philosophy()
