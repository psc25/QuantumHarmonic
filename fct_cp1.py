"""Numerical example for f : CP^1 -> C and the approximation Q_{K,L} f.

We identify CP^1 with the Bloch sphere S^2, consistently with the metric
normalization used in the paper.  The numerical construction is:

1. Expand f in spherical harmonics Y_l^m on S^2.
2. P_K f is the truncation to degrees l <= K.
3. For a physical state [x], choose the standard PU(2) transformation
       U_x = R_z(phi) R_y(theta)
   carrying [0] to [x].
4. C_L([x]) is the nearest projective Clifford+T word of length <= L.
5. Pi_L([x]) = C_L([x]) [0] is prepared with Qiskit's Statevector.
6. Q_{K,L} f([x]) = (P_K f)(Pi_L([x])).

The script plots the empirical RMSE for several K and L and writes
    cp1_rmse.png
    cp1_rmse.csv
    
Dependencies:
    pip install qiskit numpy scipy matplotlib
"""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.optimize import least_squares
from scipy.special import sph_harm_y
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
sns.set_style("white")
plt.rcParams.update({"text.usetex": True, "font.family": "serif", "font.serif": ["Computer Modern Roman"]})

try:
    from qiskit import QuantumCircuit
    from qiskit.circuit.library import HGate, TGate, TdgGate
    from qiskit.quantum_info import Operator, Statevector
except ImportError as exc:
    raise SystemExit(
        "Qiskit is required. Install dependencies with\n"
        "    pip install qiskit numpy scipy matplotlib"
    ) from exc

# ---------------------------------------------------------------------------
# Experiment parameters
# ---------------------------------------------------------------------------

SEED = 0
K_VALUES = [0, 1, 2, 3, 4]
L_VALUES = [8, 12, 16, 20, 24, 28]
N_TEST = 2000

# The Kuperberg improvement of Solovay--Kitaev allows any
# c > log_phi(2) ~= 1.44042. We fix c=1.5 and fit the unknown
# constant in epsilon_1(L) <= exp(-a L^(1/c)).
L_FIT = 8

# Deterministic quadrature for spherical-harmonic coefficients.
N_Z_QUAD = 48
N_PHI_QUAD = 96

OUTPUT_DIR = Path(__file__).resolve().parent
SQRT_4PI = np.sqrt(4.0*np.pi)

# ---------------------------------------------------------------------------
# Target function f : CP^1 ~= S^2 -> C
# ---------------------------------------------------------------------------

DELTA = 0.02
def target_function_xyz(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
    """Smooth complex-valued function on CP^1 ~= S^2.
    f([psi]) = exp(4 i <a, r([psi])>),
    where a = (1,2,3)/sqrt(14) and r is the Bloch vector.
    """
    return np.sqrt(2.0 + DELTA) * ((1.0 - z)/2.0)**((1.0 + DELTA)/2.0)

def angles_to_xyz(theta: np.ndarray, phi: np.ndarray) -> tuple[np.ndarray, ...]:
    sin_theta = np.sin(theta)
    return sin_theta*np.cos(phi), sin_theta*np.sin(phi), np.cos(theta)

def state_to_bloch(state: np.ndarray) -> tuple[float, float, float]:
    """Convert a normalized one-qubit statevector to its Bloch vector."""
    a, b = state
    x = 2.0*np.real(np.conj(a)*b)
    y = 2.0*np.imag(np.conj(a)*b)
    z = abs(a)**2 - abs(b)**2
    return float(x), float(y), float(z)

def xyz_to_angles(x: float, y: float, z: float) -> tuple[float, float]:
    theta = float(np.arccos(np.clip(z, -1.0, 1.0)))
    phi = float(np.mod(np.arctan2(y, x), 2.0*np.pi))
    return theta, phi

# ---------------------------------------------------------------------------
# PU(2) utilities and Qiskit word bank A_{1,L}
# ---------------------------------------------------------------------------

def rz(theta: float) -> np.ndarray:
    return np.array([[np.exp(-0.5j*theta), 0.0], [0.0, np.exp(0.5j*theta)]], dtype=complex)

def ry(theta: float) -> np.ndarray:
    c = np.cos(theta/2.0)
    s = np.sin(theta/2.0)
    return np.array([[c, -s], [s, c]], dtype=complex)

def state_transport(theta: float, phi: float) -> np.ndarray:
    """Standard U_x with U_x[|0>] equal to the desired projective state."""
    return rz(phi) @ ry(theta)

def projective_key(U: np.ndarray, decimals: int = 11) -> tuple[float, ...]:
    flat = U.ravel()
    pivot = next((z for z in flat if abs(z) > 1e-12), 1.0 + 0.0j)
    V = U*np.exp(-1j*np.angle(pivot))
    data = np.concatenate((V.real.ravel(), V.imag.ravel()))
    return tuple(np.round(data, decimals=decimals))

@dataclass(frozen=True)
class WordElement:
    word: tuple[str, ...]
    unitary: np.ndarray
    depth: int

def qiskit_gate_matrices() -> dict[str, np.ndarray]:
    return {
        "h": np.asarray(Operator(HGate()).data, dtype=complex),
        "t": np.asarray(Operator(TGate()).data, dtype=complex),
        "tdg": np.asarray(Operator(TdgGate()).data, dtype=complex),
    }

def generate_word_bank(max_depth: int) -> list[WordElement]:
    gate_matrices = qiskit_gate_matrices()
    identity = np.eye(2, dtype=complex)
    first = WordElement((), identity, 0)
    bank: dict[tuple[float, ...], WordElement] = {projective_key(identity): first}
    frontier = [first]
    for depth in range(1, max_depth + 1):
        new_frontier: list[WordElement] = []
        for element in frontier:
            for name, gate in gate_matrices.items():
                new_unitary = gate @ element.unitary
                new_word = element.word + (name,)
                key = projective_key(new_unitary)
                if key not in bank:
                    new_element = WordElement(new_word, new_unitary, depth)
                    bank[key] = new_element
                    new_frontier.append(new_element)
                    
        frontier = new_frontier
        
    result = list(bank.values())
    result.sort(key=lambda x: (x.depth, x.word))
    return result

def nearest_word_index(U: np.ndarray, bank_unitaries: np.ndarray, bank_depths: np.ndarray, max_depth: int) -> int:
    candidates = np.flatnonzero(bank_depths <= max_depth)
    matrices = bank_unitaries[candidates]
    overlaps = np.abs(np.einsum("ab,nab->n", U.conj(), matrices))
    return int(candidates[np.argmax(overlaps)])

def circuit_from_word(word: tuple[str, ...]) -> QuantumCircuit:
    """Construct the actual Qiskit circuit corresponding to a word."""
    circuit = QuantumCircuit(1)
    for gate in word:
        if gate == "h":
            circuit.h(0)
        elif gate == "t":
            circuit.t(0)
        elif gate == "tdg":
            circuit.tdg(0)
        else:
            raise ValueError(f"Unknown gate {gate!r}")
            
    return circuit

def qiskit_state_from_word(word: tuple[str, ...]) -> np.ndarray:
    """Prepare C_L([x])|0> with Qiskit's exact statevector simulator."""
    circuit = circuit_from_word(word)
    return np.asarray(Statevector.from_instruction(circuit).data, dtype=complex)

# ---------------------------------------------------------------------------
# Spherical harmonics on CP^1 \cong S^2
# ---------------------------------------------------------------------------

def normalized_spherical_harmonic(l: int, m: int, theta: np.ndarray, phi: np.ndarray) -> np.ndarray:
    """ONB element for the normalized probability measure dOmega/(4 pi)."""
    return SQRT_4PI*sph_harm_y(l, m, theta, phi)

def compute_spherical_coefficients(max_K: int) -> dict[tuple[int, int], complex]:
    """Compute coefficients using product quadrature for normalized area measure."""
    z_nodes, z_weights = leggauss(N_Z_QUAD)
    phis = 2.0*np.pi*np.arange(N_PHI_QUAD)/N_PHI_QUAD
    coeffs: dict[tuple[int, int], complex] = {
        (l, m): 0.0j
        for l in range(max_K + 1)
        for m in range(-l, l + 1)
    }
    for z, wz in zip(z_nodes, z_weights):
        theta = np.arccos(np.clip(z, -1.0, 1.0))
        theta_array = np.full_like(phis, theta, dtype=float)
        x, y, z_arr = angles_to_xyz(theta_array, phis)
        f_values = target_function_xyz(x, y, z_arr)
        # Normalized measure dmu = dphi/(2pi)*dz/2.
        weight = 0.5*wz/N_PHI_QUAD
        for l in range(max_K + 1):
            for m in range(-l, l + 1):
                basis = normalized_spherical_harmonic(l, m, theta_array, phis)
                coeffs[(l, m)] += weight*np.sum(f_values*np.conj(basis))
                
    return coeffs

def spectral_projection_value(theta: float, phi: float, K: int, coeffs: dict[tuple[int, int], complex]) -> complex:
    value = 0.0j
    for l in range(K + 1):
        for m in range(-l, l + 1):
            value += coeffs[(l, m)]*normalized_spherical_harmonic(l, m, theta, phi)
            
    return complex(value)

# ---------------------------------------------------------------------------
# Theorem + Solovay--Kitaev rate fit
# ---------------------------------------------------------------------------

def theorem_rate_model(K: np.ndarray | float, L: np.ndarray | float, A: float, B: float, a: float, s_eff: float, c: float) -> np.ndarray:
    K_arr = np.asarray(K, dtype=float)
    L_arr = np.asarray(L, dtype=float)
    d = 2.0 # for CP^1
    # Exact spectral truncation rate:
    # (1 + (K+1)(K+d))^{-s/2}
    spectral = A*(1.0 + (K_arr + 1.0)*(K_arr + d))**(-s_eff/2.0)
    sk = np.exp(-a*np.maximum(L_arr, 0.0)**(1.0/c))
    circuit = B*(1.0 + K_arr)**2*sk
    return spectral + circuit

def fit_theorem_rate(rmse: np.ndarray) -> dict[str, float]:
    """Fit A, B, a, s_eff, and the effective SK exponent c."""
    K_data: list[float] = []
    L_data: list[float] = []
    y_data: list[float] = []
    for k_idx, K in enumerate(K_VALUES):
        if K <= 0:
            continue
        
        for l_idx, L in enumerate(L_VALUES):
            if L < L_FIT:
                continue
            
            K_data.append(float(K))
            L_data.append(float(L))
            y_data.append(float(rmse[k_idx, l_idx]))
            
    K_arr = np.asarray(K_data)
    L_arr = np.asarray(L_data)
    y_arr = np.asarray(y_data)
    def unpack(params):
        A = np.exp(params[0])
        B = np.exp(params[1])
        a = np.exp(params[2])
        s_eff = np.exp(params[3])
        c = params[4]
        return A, B, a, s_eff, c
    
    def residual(params):
        A, B, a, s_eff, c = unpack(params)
        prediction = theorem_rate_model(K_arr, L_arr, A, B, a, s_eff, c)
        return np.log(prediction + 1e-15) - np.log(y_arr + 1e-15)
    
    initial = np.asarray([np.log(1.0), np.log(0.1), np.log(1.0), np.log(2.0), 1.5])
    lower = np.asarray([np.log(1e-8), np.log(1e-8), np.log(1e-4), np.log(0.1), np.log(2.0)/np.log((1.0 + np.sqrt(5.0))/2.0) + 1e-10])
    upper = np.asarray([np.log(1e8), np.log(1e8), np.log(1e2), np.log(20.0), 5.0])
    result = least_squares(residual, initial, bounds=(lower, upper), max_nfev=50000)
    A, B, a, s_eff, c = unpack(result.x)
    C_sk_eff = a**(-c)
    return {
        "A": A,
        "B": B,
        "a": a,
        "s_eff": s_eff,
        "c": c,
        "C_sk_eff": C_sk_eff,
        "cost": float(result.cost),
    }

# ---------------------------------------------------------------------------
# Experiment
# ---------------------------------------------------------------------------

rng = np.random.default_rng(SEED)

print("Generating Clifford+T projective word bank ...")
bank = generate_word_bank(max(L_VALUES))
bank_unitaries = np.stack([element.unitary for element in bank])
bank_depths = np.asarray([element.depth for element in bank], dtype=int)
print(f"  {len(bank)} distinct PU(2) words up to depth {max(L_VALUES)}")

print("Computing spherical-harmonic coefficients ...")
coeffs = compute_spherical_coefficients(max(K_VALUES))

# Haar-uniform physical one-qubit states = uniform points on S^2.
z_test = rng.uniform(-1.0, 1.0, size=N_TEST)
phi_test = rng.uniform(0.0, 2.0*np.pi, size=N_TEST)
theta_test = np.arccos(z_test)
x_test, y_test, _ = angles_to_xyz(theta_test, phi_test)
target = target_function_xyz(x_test, y_test, z_test)

transport = np.stack([state_transport(theta, phi) for theta, phi in zip(theta_test, phi_test)])
nearest: dict[int, np.ndarray] = {}
for L in L_VALUES:
    nearest[L] = np.asarray([nearest_word_index(U, bank_unitaries, bank_depths, L) for U in transport], dtype=int)
    
# Qiskit prepares every circuit-generated projective state that is used.
selected_indices = sorted(set(np.concatenate(list(nearest.values())).tolist()))
state_angles: dict[int, tuple[float, float]] = {}
for idx in selected_indices:
    state = qiskit_state_from_word(bank[idx].word)
    x, y, z = state_to_bloch(state)
    state_angles[idx] = xyz_to_angles(x, y, z)
    
rmse = np.zeros((len(K_VALUES), len(L_VALUES)), dtype=float)

projection_cache: dict[tuple[int, int], complex] = {}
for K in K_VALUES:
    for idx in selected_indices:
        theta_q, phi_q = state_angles[idx]
        projection_cache[(K, idx)] = spectral_projection_value(theta_q, phi_q, K, coeffs)
        
for k_idx, K in enumerate(K_VALUES):
    for l_idx, L in enumerate(L_VALUES):
        approximation = np.asarray([projection_cache[(K, idx)] for idx in nearest[L]], dtype=complex)
        rmse[k_idx, l_idx] = np.sqrt(np.mean(np.abs(target - approximation)**2))
        
rate_fit = fit_theorem_rate(rmse)
print(
    "\nTheorem + Solovay--Kitaev fit\n"
    f"  A        = {rate_fit['A']:.6g}\n"
    f"  B        = {rate_fit['B']:.6g}\n"
    f"  a        = {rate_fit['a']:.6g}\n"
    f"  s_eff    = {rate_fit['s_eff']:.6g}\n"
    f"  c        = {rate_fit['c']:.6g}\n"
    f"  C_SK,eff = {rate_fit['C_sk_eff']:.6g}"
)

print("\nEmpirical RMSE on uniform Bloch-sphere test samples")
header = "K \\ L | " + " | ".join(f"{L:>10d}" for L in L_VALUES)
print(header)
print("-"*len(header))
for k_idx, K in enumerate(K_VALUES):
    row = " | ".join(f"{rmse[k_idx, l_idx]:10.3e}" for l_idx in range(len(L_VALUES)))
    print(f"{K:>5d} | {row}")
    
csv_path = OUTPUT_DIR/"cp1_rmse.csv"
table = np.column_stack([np.asarray(K_VALUES, dtype=float), rmse])
np.savetxt(csv_path, table, delimiter=",")

fig, ax = plt.subplots(figsize=(8.2, 5.0))
cmap = LinearSegmentedColormap.from_list('mycmap', ["limegreen", "mediumturquoise", "blue"])
N_K_VALUES = [(K+1)**2 for K in K_VALUES]

for k_idx, K in enumerate(K_VALUES):
    N_K = N_K_VALUES[k_idx]
    ax.plot(L_VALUES, rmse[k_idx], marker="o", color=cmap(k_idx/len(K_VALUES)), label=f"$K={K}$ ($N_K={N_K}$)")
    if K > 0:
        fitted = theorem_rate_model(K, L_VALUES, rate_fit["A"], rate_fit["B"], rate_fit["a"], rate_fit["s_eff"], rate_fit["c"])
        ax.plot(L_VALUES, fitted, linestyle="--", color=cmap(k_idx/len(K_VALUES)), alpha=0.9)
        
ax.plot(np.nan, np.nan, linestyle="None", label=" ")
ax.plot(np.nan, np.nan, linestyle="-", marker="o", color="black", label="RMSE")
ax.plot(np.nan, np.nan, linestyle="--", color="black", label="SK")

ax.set_yscale("log")
ax.set_ylim([0.01, 0.8])
ax.set_xlabel("Circuit depth $L$")
ax.set_ylabel("$\\mathrm{RMSE}_{K,L}$")
ax.grid(True, which="both", alpha=0.25)
ax.legend(loc = "upper right", ncol = 4)
fit_text = (
    fr"$\widehat{{c}} \approx {rate_fit['c']:.5f}$"
    "\n"
    fr"$\widehat{{s}} \approx {rate_fit['s_eff']:.5f}$"
)
ax.text(0.175, 0.1, fit_text, transform=ax.transAxes, ha="right", va="center", fontsize=11,
    bbox=dict(
        boxstyle="round,pad=0.4",
        facecolor="white",
        edgecolor="gray",
        alpha=0.9,
    ),
)

figure_path = OUTPUT_DIR/"cp1_rmse.png"
fig.savefig(figure_path, dpi=300, bbox_inches="tight")
print(f"\nSaved {figure_path}")
print(f"Saved {csv_path}")