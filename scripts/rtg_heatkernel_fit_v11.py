#!/usr/bin/env python3
"""
rtg_heatkernel_fit_v11.py
-------------------------
✓ symmetric curvature links  w = 1 ± ½ ε(f_i+f_j)          (sign fixed!)
✓ shared Z₂ noise for ±ε
✓ Chebyshev–Clenshaw log‑det
✓ constant‑trace subtraction
✓ diagnostic λ₁ / Tr L print‑outs
"""

import numpy as np, matplotlib.pyplot as plt
from  scipy.sparse        import csr_matrix, identity
from  scipy.sparse.linalg import eigsh, LinearOperator

# ───────── lattice & physical constants ─────────
L          = 8                 # raise to 12 or 16 for production
V          = L**4
Delta      = 1.45e23           # RTG spectral cutoff  s⁻¹
eps_mag    = 5.0e-3            # curvature scale  ε = ±eps_mag
num_vec    = 128               # stochastic Z₂ vectors
cheb_N     = 60                # Chebyshev order for log
a          = 3e8 / Delta       # lattice spacing  (m)

# ───────── curvature field (zero mean) ─────────
grid = np.indices((L,)*4, dtype=float) - L/2
curv = np.exp(-(grid**2).sum(0)/(0.3*L)**2).flatten()
curv -= curv.mean()                    # 〈f〉 = 0

def laplacian(eps=0.0):
    """Positive‑definite 4‑D Laplacian with symmetric curvature‑weighted links."""
    rows, cols, data = [], [], []
    def idx(t,x,y,z): return (((t%L)*L+x%L)*L + y%L)*L + z%L
    for t,x,y,z in np.ndindex((L,)*4):
        i   = idx(t,x,y,z)
        diag = 0.0
        for dt,dx,dy,dz in [(1,0,0,0),(-1,0,0,0),(0,1,0,0),(0,-1,0,0),
                            (0,0,1,0),(0,0,-1,0),(0,0,0,1),(0,0,0,-1)]:
            j = idx(t+dt,x+dx,y+dy,z+dz)
            # ---------- sign FIX:  + ½ ε (…)  ----------
            w = 1.0 + 0.5*eps*(curv[i] + curv[j])   # symmetric link factor
            rows.append(i); cols.append(j); data.append(-w)
            diag += w
        rows.append(i); cols.append(i); data.append(diag)
    Lmat = csr_matrix((data,(rows,cols)),shape=(V,V))

    # shift if necessary so λ_min > 0
    lam_min = eigsh(Lmat, k=1, which='SA', return_eigenvectors=False)[0]
    if lam_min <= 0.0:
        Lmat += (1.05*(-lam_min) + 1e-6) * identity(V)
    return Lmat

# ───────── Chebyshev–Clenshaw  log(A) · v  ─────────
def cheb_log_apply(vec, A, N=cheb_N):
    lam_max = eigsh(A, k=1, which='LA', return_eigenvectors=False)[0]
    lam_min = eigsh(A, k=1, which='SA', return_eigenvectors=False)[0]
    a_scale = (lam_max - lam_min)/2
    b_shift = (lam_max + lam_min)/2
    Lop = LinearOperator((V,V),
                         matvec=lambda x:(A @ x - b_shift*x)/a_scale)

    k      = np.arange(N)
    theta  = (k + 0.5)*np.pi/N
    ck     = (2/N)*np.sum(np.cos(np.outer(k,theta)) *
                          np.log(a_scale*np.cos(theta)+b_shift), axis=1)
    ck[0] *= 0.5           # c₀ halves

    y0 = np.zeros_like(vec)
    y1 = np.zeros_like(vec)
    for c_k in ck[::-1]:
        y0, y1 = c_k*vec + 2*Lop @ y0 - y1, y0          # Clenshaw
    return (y0 - y1)/2

def stochastic_logdet(A, Z):
    return sum(xi @ cheb_log_apply(xi, A) for xi in Z) / len(Z)

# ───────── Z₂ noise vectors (same for ±ε) ─────────
rng = np.random.default_rng(20250731)
Z = [rng.choice([-1.0,1.0], size=V) for _ in range(num_vec)]

# ───────── build Laplacians & diagnostics ─────────
print("Building Laplacians …", flush=True)
L_plus  = laplacian(+eps_mag)
L_minus = laplacian(-eps_mag)

λ1_plus  = eigsh(L_plus ,  k=1, which='SA', return_eigenvectors=False)[0]
λ1_minus = eigsh(L_minus, k=1, which='SA', return_eigenvectors=False)[0]
Tr_plus  = L_plus .diagonal().sum()
Tr_minus = L_minus.diagonal().sum()
print(f"λ₁(+)={λ1_plus:.3e}  λ₁(−)={λ1_minus:.3e}")
print(f"Tr L(+)={Tr_plus:.3e}  Tr L(−)={Tr_minus:.3e}")

# ───────── stochastic log‑det (constant subtracted) ─────────
print("Estimating log‑det traces …", flush=True)
ld_plus  = stochastic_logdet(L_plus,  Z)
ld_minus = stochastic_logdet(L_minus, Z)

const = 0.5*(np.log(L_plus.diagonal()).sum() +
             np.log(L_minus.diagonal()).sum())
ld_plus  -= const
ld_minus -= const

A2_site  = (ld_plus - ld_minus) / (2*eps_mag) / V
print(f"A₂_scalar /site  = {A2_site:+.3e}")

# ───────── convert to G_N  ─────────
A2_grav  = 2.0 * A2_site                     # (+5 TT −4 ghost +1 conf.)
inv16piG = A2_grav / ((4*np.pi)**2 * a**2)   # heat‑kernel prefactor
G_extract= 1.0 / (16*np.pi*inv16piG)
print(f"G_extracted ≈ {G_extract:.3e}  m³ kg⁻¹ s⁻²")

# ───────── quick visualization ─────────
eps_vals = [-eps_mag, 0.0, +eps_mag]
ld_vals  = [ld_minus, 0.0, ld_plus]
plt.plot(eps_vals, ld_vals, 'o-')
plt.title('RTG scalar log‑det vs curvature (v11)')
plt.xlabel('ε  (curvature scale)')
plt.ylabel('stochastic  log det L')
plt.grid(True)
plt.savefig('logdet_curv.png')
plt.show()
