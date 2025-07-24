#!/usr/bin/env python
"""
RTG-v2 real-time MD (binary spin kernel)
Velocity–Verlet + Langevin, coherent spin flips, dynamic KD-tree neighbours
"""

import numpy as np, math, argparse, time, yaml
from pathlib import Path
from scipy.spatial import cKDTree

# ───── CLI ───────────────────────────────────────────────────────────
ap = argparse.ArgumentParser()
ap.add_argument('--L', type=int, default=32)
ap.add_argument('--ticks', type=int, default=20000)
ap.add_argument('--dt', type=float, default=5e-5)
ap.add_argument('--M_omega', type=float, default=30.0)
ap.add_argument('--M_phi', type=float, default=30.0)
ap.add_argument('--flip_every', type=int, default=100)
ap.add_argument('--T', type=float, default=0.02)
ap.add_argument('--gamma', type=float, default=0.1)
ap.add_argument('--kappa', type=float, default=5000.0)
ap.add_argument('--graph_update', type=int, default=200)
ap.add_argument('--r_cut', type=float, default=1.7)
ap.add_argument('--save_every', type=int, default=200)
ap.add_argument('--cfgdir', default='cfg_md')
args = ap.parse_args()

rng = np.random.default_rng(1234)

# ───── RTG constants (recursion_v2.yaml) ────────────────────────────
_pkg = Path(__file__).resolve().parents[1] / 'src' / 'rtg'
K_over_J = float(yaml.safe_load((_pkg/'recursion_v2.yaml').read_text())['K_over_J'])
J, K = 1.0, K_over_J
Δω_star = 1.0

# ───── lattice fields ───────────────────────────────────────────────
L, V = args.L, args.L**3
ω = rng.uniform(0.72, 0.78, size=V)
φ = rng.uniform(0, 2*math.pi, size=V)
s = rng.choice([+1, -1], size=V)
π_ω = rng.normal(0, 1, size=V)
π_φ = rng.normal(0, 1, size=V)

# Minimal fix (1): identical spins for Bell pair
s[0], s[1] = +1, +1

# Minimal fix (2): Centre spring at Δφ = 0
φ[0] = 0.0
φ[1] = 0.0
π_φ[0] = π_φ[1] = 0.0

# positions for KD-tree (periodic box [0,L)³)
r = rng.uniform(0, L, size=(V, 3))

M_ω, M_φ = args.M_omega, args.M_phi
cfgdir = Path(args.cfgdir); cfgdir.mkdir(exist_ok=True)

# ───── neighbour helpers ───────────────────────────────────────────
def build_kdtree():
    tree = cKDTree(r, boxsize=L)
    pairs = tree.query_pairs(args.r_cut, output_type='ndarray')
    if pairs.size == 0:
        return np.empty(0, np.int32), np.empty(0, np.int32)
    return pairs[:,0], pairs[:,1]

if args.graph_update > 0:
    i_idx, j_idx = build_kdtree()

# ───── kernels ─────────────────────────────────────────────────────
def resonance(ii, jj):
    cos = np.cos(φ[ii] - φ[jj])
    spin = 1 + s[ii]*s[jj]
    gauss = np.exp(-((ω[ii]-ω[jj])/Δω_star)**2)
    return 0.75 * (1 + cos) * spin * gauss

def F_ω():
    f = np.zeros_like(ω)
    dω = ω[i_idx] - ω[j_idx]
    np.add.at(f, i_idx, K*np.sign(dω))
    np.add.at(f, j_idx, -K*np.sign(dω))
    R = resonance(i_idx, j_idx)
    dR = R * (-2 * dω / Δω_star**2)
    np.add.at(f, i_idx, J*dR)
    np.add.at(f, j_idx, -J*dR)
    return -f

def F_φ():
    f = np.zeros_like(φ)
    sin = np.sin(φ[i_idx] - φ[j_idx])
    coeff = -0.75*(1+s[i_idx]*s[j_idx])*np.exp(-((ω[i_idx]-ω[j_idx])/Δω_star)**2)
    np.add.at(f, i_idx, J*coeff*sin)
    np.add.at(f, j_idx, -J*coeff*sin)
    angle = (φ[1] - φ[0])  # spring at Δφ=0
    spring = -args.kappa * math.sin(angle)
    f[0] += spring; f[1] -= spring
    return -f

def total_E():
    kin = (π_ω**2).sum()/(2*M_ω) + (π_φ**2).sum()/(2*M_φ)
    pot = (K*np.abs(ω[i_idx]-ω[j_idx])).sum() + (J*resonance(i_idx,j_idx)).sum()
    V_spring = args.kappa * (1 - math.cos(φ[1]-φ[0]))
    return kin + pot + V_spring

E_prev = total_E()
t0 = time.time()

for tick in range(1, args.ticks+1):

    if args.graph_update and tick % args.graph_update == 0:
        i_idx, j_idx = build_kdtree()

    π_ω += 0.5*args.dt * F_ω()
    π_φ += 0.5*args.dt * F_φ()

    ω += args.dt * π_ω / M_ω
    φ += args.dt * π_φ / M_φ
    φ %= 2*math.pi

    if args.flip_every and tick % args.flip_every == 0:
        if rng.random() < 0.5:
            s[0] *= -1
            s[1] *= -1

    if args.T > 0.0:
        sigma = math.sqrt(2*args.gamma*args.T*args.dt)
        π_ω += -args.gamma*π_ω*args.dt + sigma*rng.normal(size=V)
        π_φ += -args.gamma*π_φ*args.dt + sigma*rng.normal(size=V)

    π_ω += 0.5*args.dt * F_ω()
    π_φ += 0.5*args.dt * F_φ()

    if tick % args.save_every == 0:
        np.savez(cfgdir/f"snap_{tick:06d}.npz", omega=ω, phi=φ, spin=s, pi_omega=π_ω, pi_phi=π_φ)

print(f"\nFinished {args.ticks} ticks in {time.time()-t0:.1f}s\n")
