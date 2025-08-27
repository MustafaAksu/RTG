# rtg_heatkernel_fit_v9.py
# ------------------------
# * identical Z2 noise vectors for +/- eps
# * per-epsilon spectrum rescaling
# * trace-subtracted logdet   (removes huge constant piece)
# * volume‑normalised  A2_scalar

import numpy as np, matplotlib.pyplot as plt
from scipy.sparse import csr_matrix, identity
from scipy.sparse.linalg import eigsh, LinearOperator

L, Delta = 8, 1.45e23
V, a     = L**4, 3e8/Delta
eps_mag  = 5.0e-3
num_vec, cheb_N = 128, 60

# ---------- curvature field (zero mean) ----------
grid = np.indices((L,)*4,dtype=float)-L/2
curv = np.exp(-(grid**2).sum(0)/(0.3*L)**2).flatten()
curv -= curv.mean()

def laplacian(eps=0.0):
    rows, cols, data = [], [], []
    def idx(t,x,y,z): return (((t%L)*L+x%L)*L+y%L)*L+z%L
    for t,x,y,z in np.ndindex((L,)*4):
        i   = idx(t,x,y,z); diag=0.
        for dt,dx,dy,dz in [(1,0,0,0),(-1,0,0,0),(0,1,0,0),(0,-1,0,0),
                            (0,0,1,0),(0,0,-1,0),(0,0,0,1),(0,0,0,-1)]:
            j = idx(t+dt,x+dx,y+dy,z+dz)
            w = 1.+0.5*eps*(curv[i]+curv[j])
            rows.append(i); cols.append(j); data.append(-w)
            diag += w
        rows.append(i); cols.append(i); data.append(diag)
    Lmat = csr_matrix((data,(rows,cols)),shape=(V,V))
    λmin = eigsh(Lmat, k=1, which='SA', return_eigenvectors=False)[0]
    if λmin<=0: Lmat += (1.05*(-λmin)+1e-6)*identity(V)
    return Lmat

def cheb_log_apply(vec, A, N=cheb_N):
    λmax = eigsh(A,k=1,which='LA',return_eigenvectors=False)[0]
    λmin = eigsh(A,k=1,which='SA',return_eigenvectors=False)[0]
    a,b = (λmax-λmin)/2, (λmax+λmin)/2
    Lop = LinearOperator((V,V), matvec=lambda x:(A@x-b*x)/a)

    k   = np.arange(N); θ=(k+0.5)*np.pi/N
    ck  = (2/N)*np.sum(np.cos(np.outer(k,θ))*np.log(a*np.cos(θ)+b),1); ck[0]*=.5
    y0=y1=np.zeros_like(vec)
    for c in ck[::-1]:
        y0,y1 = c*vec+2*Lop@y0-y1, y0
    return (y0-y1)/2

def stoch_trace_log(A, Z):
    acc=0.
    for ξ in Z:
        acc += ξ @ cheb_log_apply(ξ, A)
    return acc/len(Z)

# -------- prepare common Z2 noise vectors ----------
Z = [np.random.choice([-1.,1.],V) for _ in range(num_vec)]

L_plus  = laplacian(+eps_mag)
L_minus = laplacian(-eps_mag)

log_plus  = stoch_trace_log(L_plus,  Z)
log_minus = stoch_trace_log(L_minus, Z)

# subtract trace( log diag(A) )  (constant)  before diff
def diag_log(A): return np.sum(np.log(A.diagonal()))
const = 0.5*(diag_log(L_plus)+diag_log(L_minus))
log_plus  -= const
log_minus -= const

A2_scalar = (log_plus - log_minus)/(2*eps_mag)/V   # per site
print(f"A₂_scalar /site  = {A2_scalar:+.3e}")

A2_grav = 2*A2_scalar
inv16πG = A2_grav/((4*np.pi)**2*a**2)
G_extr  = 1/(16*np.pi*inv16πG)
print(f"G_extracted ≈ {G_extr:.3e}  m³ kg⁻¹ s⁻²")
