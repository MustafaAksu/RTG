#!/usr/bin/env python3
# -------------------------------------------------------------
#  RTG Heat‑Kernel Test  (v8)   —   scalar channel + graviton prefactor
#  * automatic spectrum shift‑and‑scale
#  * Chebyshev/Lanczos logarithm
#  * stochastic Hutchinson trace
#  * symmetric finite‑difference for  A₂  (EH coefficient)
#  (c) 2025  RTG Collaboration
# -------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix, identity
from scipy.sparse.linalg import eigsh, LinearOperator

# ---------- user‑tunable parameters ----------
L        = 8           # 4‑torus size (L^4 sites).  Use 12–16 for production
delta_eps= 5.0e-3      # curvature amplitude for symmetric derivative
num_vec  = 128         # stochastic Z2 vectors (↓ variance ∝ 1/√num_vec)
cheb_N   = 60          # Chebyshev order for log polynomial
Delta    = 1.45e23     # RTG cutoff  (s⁻¹)  — sets kernel scale
# ---------------------------------------------

V  = L**4                          # total lattice sites
a  = 3.0e8 / Delta                 # relational lattice spacing  (metres)
eps_reg = 1.0/Delta**2             # UV regulator to avoid log(0)

# ---- helpers ---------------------------------------------------------------
def _site_index(t, x, y, z):
    """flatten 4D site index"""
    return (((t % L)*L + (x % L))*L + (y % L))*L + (z % L)

def curvature_profile():
    """smooth, zero‑mean bump — used to modulate link weights"""
    grid = np.indices((L,)*4, dtype=float) - L/2.0
    r2   = (grid**2).sum(axis=0)
    f    = np.exp(-r2 / (0.3*L)**2)
    return f - f.mean()            # ensure ⟨f⟩=0

CURV_FIELD = curvature_profile().flatten()   # cached

def build_shifted_laplacian(epsilon=0.0):
    """
    Return a sparse 4‑D Laplacian   L = −∇²  (positive‑definite)
    with curvature modulation  (link weight 1+ε f̄)
    """
    rows, cols, data = [], [], []
    for t, x, y, z in np.ndindex((L,)*4):
        i   = _site_index(t, x, y, z)
        diag = 0.0
        for dt,dx,dy,dz in [(1,0,0,0),(-1,0,0,0),
                            (0,1,0,0),(0,-1,0,0),
                            (0,0,1,0),(0,0,-1,0),
                            (0,0,0,1),(0,0,0,-1)]:
            j = _site_index(t+dt, x+dx, y+dy, z+dz)
            # symmetric link factor   (1 + ε/2 (f_i+f_j))
            w = 1.0 + 0.5*epsilon * (CURV_FIELD[i] + CURV_FIELD[j])
            rows.append(i); cols.append(j); data.append(-w)  # off‑diag (negative)
            diag += w
        rows.append(i); cols.append(i); data.append(diag)
    Lmat = csr_matrix((data, (rows, cols)), shape=(V, V))

    # ----------- shift so that spec(L) ⊂ (λ_shift, ∞)  with λ_shift>0 ----------
    λ_min = eigsh(Lmat, k=1, which='SA', return_eigenvectors=False)[0]
    if λ_min <= 0:
        shift = 1.05 * (-λ_min) + 1e-6
        Lmat = Lmat + shift * identity(V, format='csr')
    return Lmat

# ---- Chebyshev log‑det (vector‑apply) --------------------------------------
def cheb_log_apply(vec, Lmat, N=cheb_N):
    """
    Return  y = log(Lmat) · vec   using Chebyshev expansion (+Lanczos scaling)
    Spectrum is mapped to [−1,1] automatically.
    """
    # quick two‑extreme Lanczos
    λ_max = eigsh(Lmat, k=1, which='LA', return_eigenvectors=False)[0]
    λ_min = eigsh(Lmat, k=1, which='SA', return_eigenvectors=False)[0]
    a   = (λ_max - λ_min)/2.0
    b   = (λ_max + λ_min)/2.0

    def mv(x):             # rescaled operator  (λ → (λ-b)/a)
        return (Lmat @ x - b*x)/a
    Lop = LinearOperator((V,V), matvec=mv, dtype=float)

    # Chebyshev coeffs for log on [−1,1]   (Clenshaw–Curtis quadrature)
    k    = np.arange(N)
    theta= (k + 0.5) * np.pi / N
    ck   = (2.0/N) * np.sum(np.cos(np.outer(k, theta)) *
                            np.log(a*np.cos(theta) + b), axis=1)
    ck[0] *= 0.5

    # Clenshaw recurrence
    y0 = np.zeros_like(vec)
    y1 = np.zeros_like(vec)
    for coeff in ck[::-1]:
        y0, y1 = coeff*vec + 2.0*Lop @ y0 - y1, y0
    return (y0 - y1)/2.0

def stochastic_logdet(Lmat, nv=num_vec):
    """
    Hutchinson trace estimate:
        logdet ≈ 1/N  ∑ ξᵀ log(L) ξ
    with ξ in {+1,−1}^V
    """
    acc = 0.0
    for _ in range(nv):
        ξ = np.random.choice([-1.0, 1.0], size=V)
        acc += ξ @ cheb_log_apply(ξ, Lmat)
    return acc / nv

# ---- main workflow ---------------------------------------------------------
print("Building Laplacians …")
L_plus  = build_shifted_laplacian(+delta_eps)
L_minus = build_shifted_laplacian(-delta_eps)
L_zero  = build_shifted_laplacian(0.0)

print("Estimating log‑det traces …")
logdet_p = stochastic_logdet(L_plus)
logdet_m = stochastic_logdet(L_minus)
logdet_0 = stochastic_logdet(L_zero)

# scalar‑channel coefficient  A₂_scalar  (per site)
A2_scalar = (logdet_p - logdet_m) / (2.0*delta_eps) / V
print(f"A₂_scalar /site : {A2_scalar: .4e}")

# ---- convert to graviton + ghost system  -----------------------------------
#  heuristic  ( +5 TT  −4 ghosts  +1 conformal )  → factor 2
A2_grav   = 2.0 * A2_scalar
inv16πG   = A2_grav / ((4*np.pi)**2 * a**2)
G_extr    = 1.0 / (16*np.pi * inv16πG)
print(f"G_extracted   ≈ {G_extr: .3e}  m³ kg⁻¹ s⁻²")

# ---- quick sanity plot ------------------------------------------------------
eps_list = np.array([-delta_eps, 0.0, +delta_eps])
log_list = np.array([logdet_m,    logdet_0, logdet_p])
plt.plot(eps_list, log_list, 'o-')
plt.xlabel("ε  (curvature scale)")
plt.ylabel("stochastic  log det L")
plt.title("RTG scalar log‑det vs curvature (v8)")
plt.grid(True)
plt.tight_layout()
plt.savefig("logdet_curv.png")
plt.show()
