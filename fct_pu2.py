"""Numerical example for f : PU(2) -> C and the approximation Q_{K,L} f.

The implementation follows the structure used in the paper:

1. P_K f is computed from numerical Peter--Weyl coefficients.
2. A_{1,L} consists of projective one-qubit words of length at most L in
   {H, T, T-dagger}.  Q_L([U]) is the nearest word in the projective
   Frobenius metric.
3. Q_{K,L} f([U]) = (P_K f)(Q_L([U])).

The script plots the empirical RMSE for several K and L and writes
    pu2_rmse.png
    pu2_rmse.csv

Dependencies:
    pip install qiskit numpy scipy matplotlib
"""

from __future__ import annotations

from dataclasses import dataclass
from math import comb
from pathlib import Path

import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.optimize import least_squares
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
sns.set_style("white")
plt.rcParams.update({"text.usetex": True, "font.family": "serif", "font.serif": ["Computer Modern Roman"]})

try:
    from qiskit.circuit.library import HGate, TGate, TdgGate
    from qiskit.quantum_info import Operator
except ImportError as exc:  # pragma: no cover - helpful message for users
    raise SystemExit(
        "Qiskit is required. Install dependencies with\n"
        "    pip install qiskit numpy scipy matplotlib"
    ) from exc

# ---------------------------------------------------------------------------
# Experiment parameters
# ---------------------------------------------------------------------------

SEED = 0
K_VALUES = [0, 2, 4, 6, 8]
L_VALUES = [8, 12, 16, 20, 24]
N_TEST = 2000

# The Kuperberg improvement of Solovay--Kitaev allows any
# c > log_phi(2) ~= 1.44042. We fix c=1.5 and fit the unknown
# constant in epsilon_1(L) <= exp(-a L^(1/c)).
L_FIT = 8
THEOREM_K_POWER = 2.5  # (d^2+1)/2 for d=2

# Deterministic quadrature for the Peter--Weyl coefficients on PU(2) ~= SO(3).
N_ALPHA = 18
N_BETA = 18
N_GAMMA = 18

OUTPUT_DIR = Path(__file__).resolve().parent

# ---------------------------------------------------------------------------
# Target function f : PU(2) -> C
# ---------------------------------------------------------------------------

def target_function(U: np.ndarray) -> complex:
    """Smooth complex-valued function on PU(2):
        f([U]) = exp(0.9 (|tr U|^2 - 2)) + 0.30 i (|U_{00}|^2 - |U_{01}|^2).
    """
    class_coordinate = abs(np.trace(U))**2
    z_coordinate = abs(U[0, 0])**2 - abs(U[0, 1])**2
    return np.exp(0.9*(class_coordinate - 2.0)) + 0.30j*z_coordinate

# ---------------------------------------------------------------------------
# Basic PU(2)/SU(2) utilities
# ---------------------------------------------------------------------------

def rz(theta: float) -> np.ndarray:
    """SU(2) R_z matrix."""
    return np.array([[np.exp(-0.5j*theta), 0.0], [0.0, np.exp(0.5j*theta)]], dtype=complex)

def ry(theta: float) -> np.ndarray:
    """SU(2) R_y matrix."""
    c = np.cos(theta/2.0)
    s = np.sin(theta/2.0)
    return np.array([[c, -s], [s, c]], dtype=complex)

def su_representative(U: np.ndarray) -> np.ndarray:
    """Return a determinant-one representative of a projective 2x2 unitary.
    The square root leaves a possible sign ambiguity, but all PU(2)
    representations below have even highest weight, so that sign disappears.
    """
    det = np.linalg.det(U)
    root = np.exp(0.5j*np.angle(det))
    return U/root

def projective_key(U: np.ndarray, decimals: int = 11) -> tuple[float, ...]:
    """Hash a 2x2 unitary modulo global phase."""
    flat = U.ravel()
    pivot = next((z for z in flat if abs(z) > 1e-12), 1.0 + 0.0j)
    V = U*np.exp(-1j*np.angle(pivot))
    data = np.concatenate((V.real.ravel(), V.imag.ravel()))
    return tuple(np.round(data, decimals=decimals))

def projective_distance(U: np.ndarray, V: np.ndarray) -> float:
    """Projective Frobenius distance on PU(2).
    min_{|z|=1} ||U-zV||_F = sqrt(4 - 2 |tr(U^* V)|).
    """
    overlap = abs(np.vdot(U, V))  # = |tr(U^* V)|
    return float(np.sqrt(max(0.0, 4.0 - 2.0*overlap)))

def random_haar_su2(rng: np.random.Generator) -> np.ndarray:
    """Sample Haar SU(2) via a uniform unit quaternion."""
    q = rng.normal(size=4)
    q /= np.linalg.norm(q)
    a, b, c, d = q
    return np.array([[a + 1j*b, c + 1j*d], [-c + 1j*d, a - 1j*b]], dtype=complex)

# ---------------------------------------------------------------------------
# Symmetric-power realization of the PU(2) irreducible representations
# ---------------------------------------------------------------------------

def laplace_eigenvalue(m: int) -> float:
    """Eigenvalue for highest weight (m,0) on SU(2)/PU(2)."""
    return 0.5*m*(m + 2)


def symmetric_power_representation(U: np.ndarray, m: int) -> np.ndarray:
    """Matrix of Sym^m(U) in the normalized Dicke basis.
    The basis vector indexed by r=0,...,m contains r copies of |1> and m-r copies 
    of |0>. This avoids explicitly forming the 2^m-dimensional tensor power while 
    being exactly the restriction of the m-fold tensor power of U to Sym^m(C^2).
    """
    if m == 0:
        return np.ones((1, 1), dtype=complex)
    
    a, b = U[0, 0], U[0, 1]
    c, d = U[1, 0], U[1, 1]
    rep = np.zeros((m + 1, m + 1), dtype=complex)
    # Input basis index r = number of |1>'s; output index s likewise.
    for r in range(m + 1):
        for s in range(m + 1):
            coefficient = 0.0j
            p_min = max(0, s - r)
            p_max = min(s, m - r)
            for p in range(p_min, p_max + 1):
                q = s - p
                coefficient += comb(m - r, p)*comb(r, q)*a**(m - r - p)*c**p*b**(r - q)*d**q
                
            rep[s, r] = np.sqrt(comb(m, r)/comb(m, s))*coefficient
            
    return rep

def active_weights(K: int) -> list[int]:
    """Even m with lambda_m <= K^2, i.e. the PU(2) spectral cutoff."""
    result: list[int] = []
    m = 0
    while laplace_eigenvalue(m) <= K*K + 1e-12:
        result.append(m)
        m += 2
        
    return result

# ---------------------------------------------------------------------------
# Qiskit word bank A_{1,L}
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class WordElement:
    word: tuple[str, ...]
    unitary: np.ndarray
    depth: int

def qiskit_gate_matrices() -> dict[str, np.ndarray]:
    """Use Qiskit as the source of the H, T and T^dagger matrices."""
    return {
        "h": np.asarray(Operator(HGate()).data, dtype=complex),
        "t": np.asarray(Operator(TGate()).data, dtype=complex),
        "tdg": np.asarray(Operator(TdgGate()).data, dtype=complex),
    }

def generate_word_bank(max_depth: int) -> list[WordElement]:
    """Generate distinct projective words of length at most max_depth.
    Breadth-first search ensures that the stored depth is minimal.  Words are
    deduplicated modulo global phase, which is the natural equivalence in PU(2).
    """
    gate_matrices = qiskit_gate_matrices()
    identity = np.eye(2, dtype=complex)
    first = WordElement((), identity, 0)
    bank: dict[tuple[float, ...], WordElement] = {projective_key(identity): first}
    frontier = [first]
    for depth in range(1, max_depth + 1):
        new_frontier: list[WordElement] = []
        for element in frontier:
            for name, gate in gate_matrices.items():
                # Appending a gate to a circuit means left multiplication.
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
    """Index of Q_L([U]) in the word bank."""
    candidates = np.flatnonzero(bank_depths <= max_depth)
    matrices = bank_unitaries[candidates]
    # abs(vdot(U,V)) = |tr(U^* V)|; maximizing this minimizes d_PU.
    overlaps = np.abs(np.einsum("ab,nab->n", U.conj(), matrices))
    return int(candidates[np.argmax(overlaps)])

# ---------------------------------------------------------------------------
# Numerical Peter--Weyl transform on PU(2)
# ---------------------------------------------------------------------------

def compute_fourier_coefficients(max_K: int) -> dict[int, np.ndarray]:
    """Compute the Fourier coefficient matrix f_hat_m by deterministic Haar quadrature on SO(3).
    ZYZ Euler angles give normalized Haar measure:
        dmu = d alpha/(2 pi)*d gamma/(2 pi)*(1/2) sin(beta) d beta.
    Gauss--Legendre quadrature is used in x = cos(beta).
    """
    weights_m = active_weights(max_K)
    coeffs = {m: np.zeros((m + 1, m + 1), dtype=complex) for m in weights_m}
    x_nodes, x_weights = leggauss(N_BETA)
    alphas = 2.0*np.pi*np.arange(N_ALPHA)/N_ALPHA
    gammas = 2.0*np.pi*np.arange(N_GAMMA)/N_GAMMA
    for x, wx in zip(x_nodes, x_weights):
        beta = np.arccos(np.clip(x, -1.0, 1.0))
        beta_weight = 0.5*wx
        Ry = ry(beta)
        for alpha in alphas:
            Rza = rz(alpha)
            for gamma in gammas:
                U = Rza @ Ry @ rz(gamma)
                weight = beta_weight/(N_ALPHA*N_GAMMA)
                f_value = target_function(U)
                for m in weights_m:
                    rep = symmetric_power_representation(U, m)
                    coeffs[m] += weight*f_value*rep.conj().T
                    
    return coeffs

def spectral_projection_value(U_projective: np.ndarray, K: int, coeffs: dict[int, np.ndarray]) -> complex:
    """Evaluate P_K f at a projective unitary."""
    U = su_representative(U_projective)
    value = 0.0j
    for m in active_weights(K):
        rep = symmetric_power_representation(U, m)
        value += (m + 1)*np.trace(coeffs[m] @ rep)
    return value

# ---------------------------------------------------------------------------
# Theorem + Solovay--Kitaev rate fit
# ---------------------------------------------------------------------------

def theorem_rate_model(K: np.ndarray | float, L: np.ndarray | float, A: float, B: float, a: float, s_eff: float, c: float) -> np.ndarray:
    """Theorem-shaped L2/RMSE rate after inserting the Solovay--Kitaev rate.
    Corollary SK gives asymptotically epsilon_1(L) <= exp(-a L^(1/c)),
    with c=SK_EXPONENT_C and an unknown a>0. For n=1, the theorem gives
    an L2/RMSE bound of the form (1+K)^(-s) + (1+K)^q epsilon_1(L),
    where q=5/2 on PU(2) and q=2 on CP^1. The numerical quantity
    computed below is RMSE, so the L2/RMSE rate is used directly.
    
    A and B absorb unknown theorem constants and the Sobolev norm. Because
    the chosen target function is analytic, s_eff is only an effective
    finite-range spectral exponent, not a maximal Sobolev regularity.
    """
    K_arr = np.asarray(K, dtype=float)
    L_arr = np.asarray(L, dtype=float)
    spectral = A*(1.0 + K_arr)**(-s_eff)
    sk = np.exp(-a*np.maximum(L_arr, 0.0)**(1.0/c))
    circuit = B*(1.0 + K_arr)**THEOREM_K_POWER*sk
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

print("Computing Peter--Weyl coefficients ...")
coeffs = compute_fourier_coefficients(max(K_VALUES))

print("Sampling Haar test points ...")
test_U = np.stack([random_haar_su2(rng) for _ in range(N_TEST)])
target = np.asarray([target_function(U) for U in test_U])

nearest: dict[int, np.ndarray] = {}
for L in L_VALUES:
    nearest[L] = np.asarray([nearest_word_index(U, bank_unitaries, bank_depths, L) for U in test_U], dtype=int)
    
rmse = np.zeros((len(K_VALUES), len(L_VALUES)), dtype=float)

# Cache spectral values on every word that is actually selected.
selected_indices = sorted(set(np.concatenate(list(nearest.values())).tolist()))
projection_cache: dict[tuple[int, int], complex] = {}
for K in K_VALUES:
    for idx in selected_indices:
        projection_cache[(K, idx)] = spectral_projection_value(bank[idx].unitary, K, coeffs)

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

print("\nEmpirical RMSE on Haar test samples")
header = "K \\ L | " + " | ".join(f"{L:>10d}" for L in L_VALUES)
print(header)
print("-"*len(header))
for k_idx, K in enumerate(K_VALUES):
    row = " | ".join(f"{rmse[k_idx, l_idx]:10.3e}" for l_idx in range(len(L_VALUES)))
    print(f"{K:>5d} | {row}")

csv_path = OUTPUT_DIR/"pu2_rmse.csv"
table = np.column_stack([np.asarray(K_VALUES, dtype=float), rmse])
np.savetxt(csv_path, table, delimiter=",")

cmap = LinearSegmentedColormap.from_list('mycmap', ["limegreen", "mediumturquoise", "blue"])
def NK(K: int) -> int:
    """Dimension of the PU(2) spectral space with lambda_m <= K^2."""
    return sum((m + 1)**2 for m in active_weights(K))

N_K_VALUES = [NK(K) for K in K_VALUES]
fig, ax = plt.subplots(figsize=(8.2, 5.0))

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
ax.set_ylim([0.04, 2.0])
ax.set_xlabel("Circuit depth $L$")
ax.set_ylabel("$\\mathrm{RMSE}_{K,L}$")
ax.grid(True, which="both", alpha=0.25)
ax.legend(loc = "upper right", ncol = 4)
fit_text = (
    fr"$\widehat{{c}} \approx {rate_fit['c']:.5f}$"
    "\n"
    fr"$\widehat{{s}} \approx {rate_fit['s_eff']:.5f}$"
)
ax.text(0.17, 0.1, fit_text, transform=ax.transAxes, ha="right", va="center", fontsize=11,
    bbox=dict(
        boxstyle="round,pad=0.4",
        facecolor="white",
        edgecolor="gray",
        alpha=0.9,
    ),
)

figure_path = OUTPUT_DIR/"pu2_rmse.png"
fig.savefig(figure_path, dpi=300, bbox_inches="tight")
print(f"\nSaved {figure_path}")
print(f"Saved {csv_path}")