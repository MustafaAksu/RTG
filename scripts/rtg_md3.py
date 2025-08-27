#!/usr/bin/env python
"""
Final stable RTG-v2 MD script.
"""

import numpy as np, math, argparse, yaml
from pathlib import Path
from scipy.spatial import cKDTree

# CLI
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
ap.add_argument('--flip_threshold', type=float, default=0.28)
args = ap.parse_args()

rng = np.random.default_rng(1234)

# Constants
_pkg = Path(__file__).resolve().parents[1] / 'src' / 'rtg'
K_over_J = float(yaml.safe_load((_pkg/'recursion_v2.yaml').read_text())['K_over_J'])
J, K = 1.0, K_over_J
delta_omega_star = 1.0

# Lattice setup
L, V = args.L, args.L**3
omega = rng.uniform(0.72, 0.78, V)
phi = rng.uniform(0, 2*math.pi, V)
spin = rng.choice([+1, -1], V)
pi_omega = rng.normal(0, 1, V)
pi_phi = rng.normal(0, 1, V)

# Bell pair (aligned spins)
spin[0], spin[1] = +1, +1
phi[0], phi[1] = 0.0, math.pi
pi_phi[[0, 1]] *= 0.0

# Positions
r = rng.uniform(0, L, (V, 3))

# KD-tree neighbors
def build_kdtree():
    tree = cKDTree(r, boxsize=L)
    pairs = tree.query_pairs(args.r_cut, output_type='ndarray')
    if pairs.size == 0:
        return np.empty(0,int), np.empty(0,int)
    return pairs[:,0], pairs[:,1]

i_idx, j_idx = build_kdtree()

# Resonance
def resonance(ii, jj):
    cos_term = np.cos(phi[ii] - phi[jj])
    spin_term = 1 + spin[ii]*spin[jj]
    gauss = np.exp(-((omega[ii]-omega[jj])/delta_omega_star)**2)
    return 0.75*(1+cos_term)*spin_term*gauss

# Forces
def force_omega():
    f = np.zeros(V)
    d_omega = omega[i_idx]-omega[j_idx]
    R = resonance(i_idx, j_idx)
    np.add.at(f, i_idx, K*np.sign(d_omega)+J*R*(-2*d_omega/delta_omega_star**2))
    np.add.at(f, j_idx, -K*np.sign(d_omega)-J*R*(-2*d_omega/delta_omega_star**2))
    return -f

def force_phi():
    f = np.zeros(V)
    sin_term = np.sin(phi[i_idx]-phi[j_idx])
    coeff = -0.75*(1+spin[i_idx]*spin[j_idx])*np.exp(-((omega[i_idx]-omega[j_idx])/delta_omega_star)**2)
    np.add.at(f, i_idx, J*coeff*sin_term)
    np.add.at(f, j_idx, -J*coeff*sin_term)
    spring = -args.kappa * math.sin(phi[1]-phi[0]-math.pi)
    f[0] += spring; f[1] -= spring
    return -f

# Total energy (corrected)
def total_energy():
    kin = (pi_omega**2).sum()/(2*args.M_omega) + (pi_phi**2).sum()/(2*args.M_phi)
    pot = (K*np.abs(omega[i_idx]-omega[j_idx])).sum() + (J*resonance(i_idx,j_idx)).sum()
    spring_energy = args.kappa*(1-math.cos(phi[1]-phi[0]-math.pi))
    return kin+pot+spring_energy

cfgdir=Path(args.cfgdir); cfgdir.mkdir(exist_ok=True)

# MD loop
for tick in range(1, args.ticks+1):
    if args.graph_update and tick % args.graph_update == 0:
        i_idx, j_idx = build_kdtree()

    pi_omega+=0.5*args.dt*force_omega()
    pi_phi+=0.5*args.dt*force_phi()
    omega+=args.dt*pi_omega/args.M_omega
    phi=(phi+args.dt*pi_phi/args.M_phi)%(2*math.pi)

    if tick%args.flip_every==0:
        w=np.exp(-((omega[i_idx]-omega[j_idx])/delta_omega_star)**2)
        gated=w>np.exp(-(args.flip_threshold)**2)
        h=np.zeros(V)
        np.add.at(h,i_idx[gated],w[gated]*spin[j_idx[gated]])
        np.add.at(h,j_idx[gated],w[gated]*spin[i_idx[gated]])
        delta_E=-2*J*spin*h
        beta=1/max(args.T,1e-5) # robust beta
        p_flip=1/(1+np.exp(beta*delta_E))
        mask=rng.random(V)<p_flip
        if mask[0]or mask[1]:mask[0]=mask[1]=True
        spin[mask]*=-1

    if tick%args.save_every==0:
        flip_rate=np.mean(mask)
        spin_corr=np.mean(spin[i_idx]*spin[j_idx])
        energy=total_energy()
        print(f"Tick {tick}: Flip={flip_rate:.3f}, Spin_corr={spin_corr:.3f}, E={energy:.3f}")
        np.savez(cfgdir/f"snap_{tick:06d}.npz", omega=omega,phi=phi,spin=spin,pi_omega=pi_omega,pi_phi=pi_phi,flip_mask=mask)

print("MD complete.")
