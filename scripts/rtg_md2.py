#!/usr/bin/env python
"""
RTG-v2 MD script with improved spin-flip dynamics, final version.
Velocity–Verlet + Langevin dynamics, coherent spin flips, dynamic KD-tree neighbors.
"""

import numpy as np, math, argparse, time, yaml
from pathlib import Path
from scipy.spatial import cKDTree

# CLI arguments
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
ap.add_argument('--quantum_flip', action='store_true')
ap.add_argument('--flip_threshold', type=float, default=0.28)
ap.add_argument('--tunnel_delta', type=float, default=1.0)
ap.add_argument('--tunnel_alpha', type=float, default=1.0)
ap.add_argument('--flip_mode', default='glauber', choices=['glauber', 'metropolis'])
ap.add_argument('--gauge_phi', action='store_true')
ap.add_argument('--bell_pin', action='store_true', default=True, help='Pin Bell pair to prevent flipping')
ap.add_argument('--curvature_strength', type=float, default=0.0, help='Curvature strength for additional potential term')
args = ap.parse_args()

rng = np.random.default_rng(1234)

# Constants from recursion_v2.yaml
_pkg = Path(__file__).resolve().parents[1] / 'src' / 'rtg'
K_over_J = float(yaml.safe_load((_pkg/'recursion_v2.yaml').read_text())['K_over_J'])
J, K = 1.0, K_over_J
delta_omega_star = 1.0

# Lattice setup
L, V = args.L, args.L**3
omega = rng.uniform(0.72, 0.78, size=V)
phi = rng.uniform(0, 2*math.pi, size=V)
spin = rng.choice([+1, -1], size=V)
pi_omega = rng.normal(0, 1, size=V)
pi_phi = rng.normal(0, 1, size=V)

# Bell pair setup (aligned spins for CHSH)
spin[0], spin[1] = +1, +1
phi[0], phi[1] = 0.0, math.pi
pi_phi[[0, 1]] *= 0.0

# Positions
r = rng.uniform(0, L, size=(V, 3))

# KD-tree neighbor build function
def build_kdtree():
    tree = cKDTree(r, boxsize=L)
    pairs = tree.query_pairs(args.r_cut, output_type='ndarray')
    if pairs.size == 0:
        return np.empty(0, int), np.empty(0, int)
    return pairs[:,0], pairs[:,1]

i_idx, j_idx = build_kdtree()

# Resonance calculation
def resonance(ii, jj):
    cos_term = np.cos(phi[ii] - phi[jj])
    spin_term = 1 + spin[ii]*spin[jj]
    gauss = np.exp(-((omega[ii]-omega[jj])/delta_omega_star)**2)
    return 0.75 * (1 + cos_term) * spin_term * gauss

# Force calculations
def force_omega():
    f = np.zeros(V)
    d_omega = omega[i_idx] - omega[j_idx]
    np.add.at(f, i_idx, K*np.sign(d_omega))
    np.add.at(f, j_idx, -K*np.sign(d_omega))
    R = resonance(i_idx, j_idx)
    dR = R * (-2 * d_omega / delta_omega_star**2)
    np.add.at(f, i_idx, J*dR)
    np.add.at(f, j_idx, -J*dR)
    return -f

def force_phi():
    f = np.zeros(V)
    sin_term = np.sin(phi[i_idx] - phi[j_idx])
    coeff = -0.75*(1+spin[i_idx]*spin[j_idx])*np.exp(-((omega[i_idx]-omega[j_idx])/delta_omega_star)**2)
    np.add.at(f, i_idx, J*coeff*sin_term)
    np.add.at(f, j_idx, -J*coeff*sin_term)
    spring = -args.kappa * math.sin(phi[1] - phi[0] - math.pi)
    f[0] += spring; f[1] -= spring
    return -f

# Total energy including spin-energy term
def total_energy():
    kin = (pi_omega**2).sum()/(2*args.M_omega) + (pi_phi**2).sum()/(2*args.M_phi)
    pot = (K*np.abs(omega[i_idx]-omega[j_idx])).sum() + (J*resonance(i_idx,j_idx)).sum()
    spring_energy = args.kappa * (1 - math.cos(phi[1]-phi[0]-math.pi))
    return kin + pot + spring_energy

# Main MD loop
cfgdir = Path(args.cfgdir); cfgdir.mkdir(exist_ok=True)

E0 = 0.0
h=0
for tick in range(1, args.ticks+1):
    if args.graph_update and tick % args.graph_update == 0:
        i_idx, j_idx = build_kdtree()

    pi_omega += 0.5*args.dt*force_omega()
    pi_phi += 0.5*args.dt*force_phi()
    omega += args.dt*pi_omega/args.M_omega
    phi = (phi + args.dt*pi_phi/args.M_phi) % (2*math.pi)

    # Spin-flip dynamics
    if tick % args.flip_every == 0:
        candidates = rng.choice(V, size=V//20, replace=False)  # 5% of sites
        mask = np.zeros(V, dtype=bool)

        for i in candidates:
            # neighbours of i
            neigh = np.concatenate([j_idx[i_idx == i], i_idx[j_idx == i]])
            if neigh.size == 0:
                continue

            # energy BEFORE
            s_old = spin[i]
            R_old = resonance(np.full_like(neigh, i), neigh)
            E_old = (J * R_old).sum()

            # energy AFTER hypothetical flip
            s_new = -s_old
            spin[i] = s_new  # temporary
            R_new = resonance(np.full_like(neigh, i), neigh)
            E_new = (J * R_new).sum()
            spin[i] = s_old  # revert unless accepted

            dE = E_new - E_old

            # Metropolis acceptance
            beta = 1.0/args.T if args.T > 0 else np.inf
            p_acc = 1.0 if dE < 0 else math.exp(-beta*dE)

            if rng.random() < p_acc:
                spin[i] = s_new
                mask[i] = True

                # kinetic compensation
                if dE != 0:
                    pi_phi[i] += np.sign(-dE) * np.sqrt(2*args.M_phi * np.abs(dE))

        if args.quantum_flip:
            # Apply quantum tunnel to the same candidates
            for i in candidates:
                if not mask[i]:
                    dE_tunnel = dE  # Reuse or recalculate if needed
                    tunnel_prob = args.tunnel_delta * math.exp(-args.tunnel_alpha * math.abs(omega[i] - np.mean(omega[neigh])) / delta_omega_star) * args.dt
                    if rng.random() < tunnel_prob:
                        spin[i] *= -1
                        mask[i] = True
                        if dE_tunnel != 0:
                            pi_phi[i] += np.sign(-dE_tunnel) * np.sqrt(2*args.M_phi * np.abs(dE_tunnel))

        # Bell pair coherence
        if args.bell_pin:
            if mask[0] or mask[1]: mask[0]=mask[1]=True
        spin[mask] *= -1

    # Logging and Saving
    if tick % args.save_every == 0:
        flip_rate = np.mean(mask)
        spin_corr = np.mean(spin[i_idx]*spin[j_idx])
        energy = total_energy()
        print(f"Tick {tick}: Flip rate={flip_rate:.4f}, Spin corr={spin_corr:.4f}, Energy={energy:.4f}, dE/E0={(energy - E0)/E0 if E0 != 0 else 0:.2e}")
        np.savez(cfgdir/f"snap_{tick:06d}.npz", omega=omega, phi=phi, spin=spin, pi_omega=pi_omega, pi_phi=pi_phi, flip_mask=mask, local_h=h)
        if tick % 100 == 0:
            E0 = energy

print("\nMD run complete.")