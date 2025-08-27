#!/usr/bin/env python3
"""
rtg_heatkernel_fit_v12.py   –  common shift + 3‑point slope fit
"""

import numpy as np, matplotlib.pyplot as plt
from scipy.sparse import csr_matrix, identity
from scipy.sparse.linalg import eigsh, LinearOperator

# ── lattice / physics ────────────────────────────
L = 8
V = L**4
Delta   = 1.45e23
eps_mag = 5.0e-3
num_vec = 256
cheb_N  = 60
a = 3e8 / Delta

# ── curvature profile (⟨f⟩=0) ────────────────────
grid = np.indices((L,)*4,dtype=float) - L/2
curv = np.exp(-(grid**2).sum(0)/(0.3*L)**2).flatten()
curv -= curv.mean()

def laplacian(eps=0.0, safety_shift=False):
    rows,cols,data=[],[],[]
    idx = lambda t,x,y,z:(((t%L)*L+x%L)*L+y%L)*L+z%L
    for t,x,y,z in np.ndindex((L,)*4):
        i  = idx(t,x,y,z); diag = 0.0
        for dt,dx,dy,dz in [(1,0,0,0),(-1,0,0,0),(0,1,0,0),(0,-1,0,0),
                            (0,0,1,0),(0,0,-1,0),(0,0,0,1),(0,0,0,-1)]:
            j  = idx(t+dt,x+dx,y+dy,z+dz)
            w  = 1.0 + 0.5*eps*(curv[i]+curv[j])   # sign OK
            rows.append(i); cols.append(j); data.append(-w)
            diag += w
        rows.append(i); cols.append(i); data.append(diag)
    Lmat = csr_matrix((data,(rows,cols)),shape=(V,V))
    if safety_shift:
        lam_min = eigsh(Lmat, k=1, which='SA', return_eigenvectors=False)[0]
        if lam_min<=0: Lmat += (1.05*(-lam_min)+1e-6)*identity(V)
    return Lmat

# ── Chebyshev log(A)·v ───────────────────────────
def cheb_log_apply(v, A, N=cheb_N):
    lmax = eigsh(A,k=1,which='LA',return_eigenvectors=False)[0]
    lmin = eigsh(A,k=1,which='SA',return_eigenvectors=False)[0]
    aS, bS = (lmax-lmin)/2, (lmax+lmin)/2
    Lop = LinearOperator((V,V), matvec=lambda x:(A@x-bS*x)/aS)
    k   = np.arange(N); θ=(k+0.5)*np.pi/N
    ck  = (2/N)*np.sum(np.cos(np.outer(k,θ))*np.log(aS*np.cos(θ)+bS),axis=1); ck[0]*=0.5
    y0=y1=np.zeros_like(v)
    for c in ck[::-1]: y0,y1 = c*v + 2*Lop@y0 - y1, y0
    return (y0 - y1)/2

def stochastic_logdet(A, Z): return sum(xi@cheb_log_apply(xi,A) for xi in Z)/len(Z)

# ── Z₂ noise (fixed) ─────────────────────────────
rng=np.random.default_rng(20250731)
Z=[rng.choice([-1.,1.],size=V) for _ in range(num_vec)]

# ── build raw Laplacians ─────────────────────────
print("Building raw Laplacians …")
L_raw = {ε: laplacian(ε, safety_shift=False) for ε in (-eps_mag,0.0,+eps_mag)}

# common positive shift
lam_mins=[eigsh(M,k=1,which='SA',return_eigenvectors=False)[0] for M in L_raw.values()]
shift = max(0.0, 1.05*max(-min(lam_mins),0)+1e-6)
L = {ε: M + shift*identity(V) for ε,M in L_raw.items()}

print(f"common shift  = {shift:.3e}")
for ε in (-eps_mag,0.0,+eps_mag):
    lam1 = eigsh(L[ε],k=1,which='SA',return_eigenvectors=False)[0]
    print(f"λ₁({ε:+.3e}) = {lam1:.3e}")

# ── constant‑trace subtraction (ε‑indep.) ────────
const = np.mean([np.log(M.diagonal()).sum() for M in L.values()])

# log‑detimates
print("Estimating stochastic log‑dets …")
ld = {ε: stochastic_logdet(M,Z)-const for ε,M in L.items()}

# three‑point slope
eps_arr = np.array(list(ld.keys()))
ld_arr  = np.array(list(ld.values()))
slope   = np.polyfit(eps_arr, ld_arr, 1)[0]
A2_site = slope / V
print(f"A₂_scalar /site = {A2_site:+.3e}")

# ── convert to G_N ───────────────────────────────
A2_grav  = 2.0 * A2_site
inv16πG  = A2_grav / ((4*np.pi)**2 * a**2)
G_extr   = 1.0 / (16*np.pi*inv16πG)
print(f"G_extracted ≈ {G_extr:.3e}  m³ kg⁻¹ s⁻²")

# optional plot
plt.plot(eps_arr, ld_arr, 'o-')
plt.title('RTG scalar log‑det vs curvature (v12)')
plt.xlabel('ε (curvature scale)')
plt.ylabel('stochastic log det L')
plt.grid(True)
plt.savefig('logdet_curv.png')
plt.show()
