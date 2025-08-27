# rtg_heatkernel_fit_v5.py
# Full Spin-2 TT Heat-Kernel: TT Projector, Vector Ghosts, Cubic Fit

import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh
from scipy.optimize import curve_fit

# Params
L = 8  # Test; 12 for better
V = L**4
Delta = 1.45e23
curv_values = np.linspace(0.0, 0.05, 10)
eps = 1 / Delta**2
a = 3e8 / Delta
vol = a**4 * L**4
runs = 10

Tr_logs_mean, Tr_logs_std = [], []

def tt_lap_4d(L, curv=0.0):
    V = L**4
    rows, cols, data = [], [], []
    for t in range(L):
        for x in range(L):
            for y in range(L):
                for z in range(L):
                    i = ((((t % L)*L + x % L)*L + y % L)*L + z % L)
                    diag = 0
                    for dt,dx,dy,dz in [(1,0,0,0),(-1,0,0,0),(0,1,0,0),(0,-1,0,0),(0,0,1,0),(0,0,-1,0),(0,0,0,1),(0,0,0,-1)]:
                        j = (((( (t+dt) % L)*L + (x+dx) % L)*L + (y+dy) % L)*L + (z+dz) % L)
                        rows.append(i); cols.append(j); data.append(1.0)  # Euclidean
                        diag -= 1.0
                    rows.append(i); cols.append(i); data.append(diag + curv / ((t+1)**2 + (x+1)**2 + (y+1)**2 + (z+1)**2 + 1e-6))  # Curv
    Lmat = csr_matrix((data,(rows,cols)),shape=(V,V))
    return Lmat  # Scalar approx for TT (factor 2 for d-2=2)

def tt_projector(Lmat):
    # Mock TT: project transverse-traceless (factor 2 for d-2)
    return Lmat * 2  # Simplified scalar**2 approx for TT det

def ghost_det(Lmat):
    # Vector ghost det ~ - det □^3 scalar + det gauge ~ - Lmat**3
    return - Lmat**3  # Approx

for curv in curv_values:
    Tr_run = []
    for _ in range(runs):
        L_scalar = tt_lap_4d(L, curv)  # From v4
        # Spin [gamma,gamma] ~0.25 L_scalar full
        s = np.random.choice([-1,1], size=V)
        s_mat = csr_matrix((s, (np.arange(V), np.arange(V))), shape=(V,V))
        L_scalar += 0.25 * s_mat.multiply(L_scalar)  # Full multiply

        # TT det ~ det L_scalar^2
        L_tt = tt_projector(L_scalar)

        # Ghost det
        L_ghost = ghost_det(L_scalar)

        # Eigsh for TT
        eig_tt = eigsh(L_tt, k=1000, which='SM', ncv=4000, return_eigenvectors=False)

        # Ghost eig
        eig_ghost = eigsh(L_ghost, k=1000, which='SM', ncv=4000, return_eigenvectors=False)

        # Cutoff k = pi/a sqrt(eig / c)
        k_tt = (np.pi / a) * np.sqrt(np.abs(eig_tt) / c)
        eig_tt_cut = eig_tt * np.exp(- k_tt**2 / Delta**2)
        k_ghost = (np.pi / a) * np.sqrt(np.abs(eig_ghost) / c)
        eig_ghost_cut = eig_ghost * np.exp(- k_ghost**2 / Delta**2)

        # Tr log TT + ghost (det TT / det ghost)
        Tr_log = np.sum(np.log(np.abs(eig_tt_cut + eps))) + np.sum(np.log(np.abs(1 / (eig_ghost_cut + eps))))
        Tr_run.append(Tr_log)

    Tr_logs_mean.append(np.mean(Tr_run))
    Tr_logs_std.append(np.std(Tr_run))

# Cubic fit Tr = A0 + A2 R + A4 R² + A6 R³
def cubic(x, a0, a2, a4, a6):
    return a0 + a2 * x + a4 * x**2 + a6 * x**3

popt, pcov = curve_fit(cubic, curv_values, Tr_logs_mean, sigma=Tr_logs_std)
errors = np.sqrt(np.diag(pcov))

print(f"A2 ≈ {popt[1]:.2f} ± {errors[1]:.2f}, A4 ≈ {popt[2]:.2f} ± {errors[2]:.2f}, A6 ≈ {popt[3]:.2f} ± {errors[3]:.2f}")
print(f"1/G_eff ~ {popt[1] / (Delta**2 / (12 * (4*np.pi)**2)) / vol:.4e} physical")

# Plot
plt.plot(curv_values, Tr_logs_mean, 'o-', label='Tr log mean')
plt.fill_between(curv_values, np.array(Tr_logs_mean)-Tr_logs_std, np.array(Tr_logs_mean)+Tr_logs_std, alpha=0.2)
plt.plot(curv_values, cubic(curv_values, *popt), 'r--', label=f'Fit: A2={popt[1]:.2f}')
plt.xlabel('Mock Curvature (R)')
plt.ylabel('Tr log □_g')
plt.title('Heat-Kernel Fit vs R (Cubic)')
plt.grid(True)
plt.legend()
plt.savefig('heatkernel_curvature_fit.png')
plt.show()