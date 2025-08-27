#!/usr/bin/env python3
"""
rtg_md2_v1.2.py — RTG Molecular‑Dynamics with dynamic spin flips
───────────────────────────────────────────────────────────────────────
Highlights:
    • Velocity–Verlet + BAOAB Langevin thermostat
    • Canonical RTG resonance kernel with YAML‑sourced Δω*
    • Metropolis / Glauber + optional quantum‑tunnel spin flips
    • Bell pair phase‑spring (aligned / anti‑aligned selectable)
    • KD‑tree neighbour list with periodic boxes
    • CHSH‑ready diagnostics (pairwise kernel & spin correlation)
    • Reproducible RNG (`--seed`) + run‑parameter dump + CSV log
"""

import math, time, argparse, yaml
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

# ────────────────────────────── CLI ─────────────────────────────────────
ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
ap.add_argument('--L',             type=int,   default=32,    help='lattice length (cubic)')
ap.add_argument('--ticks',         type=int,   default=20000)
ap.add_argument('--dt',            type=float, default=5e-5,  help='time step')
ap.add_argument('--M_omega',       type=float, default=30.0)
ap.add_argument('--M_phi',         type=float, default=30.0)
ap.add_argument('--flip_every',    type=int,   default=100,   help='spin-flip interval')
ap.add_argument('--T',             type=float, default=0.02,  help='thermostat temperature')
ap.add_argument('--gamma',         type=float, default=0.1)
ap.add_argument('--kappa',         type=float, default=5.0e3)
ap.add_argument('--graph_update',  type=int,   default=200)
ap.add_argument('--r_cut',         type=float, default=1.7)
ap.add_argument('--save_every',    type=int,   default=200)
ap.add_argument('--cfgdir',        default='cfg_md')
ap.add_argument('--quantum_flip',  action='store_true')
ap.add_argument('--tunnel_delta',  type=float, default=1.0)
ap.add_argument('--tunnel_alpha',  type=float, default=1.0)
ap.add_argument('--flip_mode',     default='glauber', choices=['glauber', 'metropolis'])
ap.add_argument('--gauge_phi',     action='store_true', help='remove global φ drift each step')
ap.add_argument('--no-bell-pin',   dest='bell_pin', action='store_false')
ap.set_defaults(bell_pin=True)
ap.add_argument('--bell_mode',     choices=['aligned','anti'], default='anti')
ap.add_argument('--seed',          type=int,   default=1234)
args = ap.parse_args()

rng = np.random.default_rng(args.seed)

# ───────────────────────────── Constants ────────────────────────────────
_pkg = Path(__file__).resolve().parents[1] / 'src' / 'rtg'
meta = yaml.safe_load((_pkg / 'recursion_v2.yaml').read_text())
K_over_J = float(meta['K_over_J'])
delta_omega_star = float(meta.get('DeltaOmegaStar', 1.45e23))
J, K = 1.0, K_over_J

# ────────────────────── Initial Lattice Setup ───────────────────────────
L = args.L
V = L**3
omega = rng.uniform(0.72, 0.78, size=V)
phi = rng.uniform(0.0, 2*math.pi, size=V)
spin = rng.choice([+1, -1], size=V)
pi_omega = rng.normal(0, 1, size=V)
pi_phi = rng.normal(0, 1, size=V)
r = rng.uniform(0, L, size=(V, 3))

# Bell pair (sites 0 & 1)
spin[0] = spin[1] = +1
if args.bell_mode == 'aligned':
    phi[0], phi[1] = 0.0, 0.0
    target_phase = 0.0
else:
    phi[0], phi[1] = 0.0, math.pi
    target_phase = math.pi
pi_phi[[0,1]] = 0.0

# ────────────────────── KD-Tree Neighbour Helpers ───────────────────────
def build_kdtree():
    tree = cKDTree(r, boxsize=L)
    pairs = tree.query_pairs(args.r_cut, output_type='ndarray')
    if pairs.size == 0:
        return np.empty(0, int), np.empty(0, int)
    return pairs[:,0], pairs[:,1]

i_idx, j_idx = build_kdtree()

def make_neighbour_cache():
    cache = [[] for _ in range(V)]
    for a,b in zip(i_idx, j_idx):
        cache[a].append(b)
        cache[b].append(a)
    return [np.array(lst, dtype=int) for lst in cache]

nbr_cache = make_neighbour_cache()

# ────────────────────────── Force Kernels ───────────────────────────────
def resonance(ii, jj):
    cos_term  = np.cos(phi[ii] - phi[jj])
    spin_term = 1.0 + spin[ii]*spin[jj]
    gauss     = np.exp(-((omega[ii]-omega[jj])/delta_omega_star)**2)
    return 0.75 * (1.0 + cos_term) * spin_term * gauss

def force_omega():
    f = np.zeros(V)
    dω = omega[i_idx] - omega[j_idx]
    np.add.at(f, i_idx,  K*np.sign(dω))
    np.add.at(f, j_idx, -K*np.sign(dω))
    R   = resonance(i_idx, j_idx)
    dR  = R * (-2.0 * dω / delta_omega_star**2)
    np.add.at(f, i_idx,  J*dR)
    np.add.at(f, j_idx, -J*dR)
    return -f

def force_phi():
    f = np.zeros(V)
    sin_term = np.sin(phi[i_idx] - phi[j_idx])
    coeff = -0.75*(1.0 + spin[i_idx]*spin[j_idx]) * np.exp(-((omega[i_idx]-omega[j_idx])/delta_omega_star)**2)
    np.add.at(f, i_idx,  J*coeff*sin_term)
    np.add.at(f, j_idx, -J*coeff*sin_term)
    spring = -args.kappa * math.sin(phi[1] - phi[0] - target_phase)
    f[0] += spring ; f[1] -= spring
    return -f

def total_energy():
    kin = (pi_omega**2).sum()/(2*args.M_omega) + (pi_phi**2).sum()/(2*args.M_phi)
    pot = (K*np.abs(omega[i_idx]-omega[j_idx])).sum() + (J*resonance(i_idx,j_idx)).sum()
    spring = args.kappa * (1.0 - math.cos(phi[1] - phi[0] - target_phase))
    return kin + pot + spring

# ──────────────────────── CHSH Estimator ────────────────────────────────
def chsh_pair():
    phase = phi[0] - phi[1]
    e = lambda θ1, θ2: math.cos(phase - (θ1 - θ2)) * spin[0]*spin[1]
    return e(0, math.pi) + e(math.pi/2, math.pi) + e(0, 3*math.pi/2) - e(math.pi/2, 3*math.pi/2)

# ───────────────────────────── Main Loop ────────────────────────────────
cfgdir = Path(args.cfgdir); cfgdir.mkdir(exist_ok=True)
param_file = cfgdir / 'params.yaml'
if param_file.exists():
    print(f"[warn] Overwriting {param_file}")
param_file.write_text(yaml.dump(vars(args)))

logfile = cfgdir / 'log.csv'
logfile.write_text('tick,energy,dE_frac,flip_rate,R01,CHSH\n')

E0 = total_energy()
t0 = time.time()

for tick in range(1, args.ticks+1):
    if tick % max(50, args.graph_update) == 0:
        try:
            i_idx, j_idx = build_kdtree()
            nbr_cache = make_neighbour_cache()
        except ValueError:
            print(f"[warn] KD-tree failed at tick {tick}")

    pi_omega += 0.5*args.dt * force_omega()
    pi_phi   += 0.5*args.dt * force_phi()
    omega    += args.dt * pi_omega / args.M_omega
    phi      = (phi + args.dt * pi_phi / args.M_phi) % (2*math.pi)

    if args.gauge_phi:
        phi -= phi.mean()

    if args.gamma > 0 and args.T > 0:
        c = math.exp(-args.gamma * args.dt)
        σ1 = math.sqrt((1-c**2) * args.M_omega * args.T)
        σ2 = math.sqrt((1-c**2) * args.M_phi   * args.T)
        pi_omega = c*pi_omega + rng.normal(0, σ1, size=V)
        pi_phi   = c*pi_phi   + rng.normal(0, σ2, size=V)

    pi_omega += 0.5*args.dt * force_omega()
    pi_phi   += 0.5*args.dt * force_phi()

    flipped = np.zeros(V, dtype=bool)

    if tick % args.flip_every == 0:
        bell_indices = {0, 1} if args.bell_pin else set()
        pool = np.array([i for i in range(V) if i not in bell_indices])
        candidates = rng.choice(pool, size=len(pool)//20, replace=False)

        for i in candidates:
            neigh = nbr_cache[i]
            if neigh.size == 0: continue
            s_old = spin[i]
            R_old = resonance(np.full_like(neigh, i), neigh)
            E_old = (J*R_old).sum()

            s_new = -s_old
            spin[i] = s_new
            R_new = resonance(np.full_like(neigh, i), neigh)
            spin[i] = s_old
            dE = (J*R_new).sum() - E_old

            β = 1.0/args.T if args.T > 0 else np.inf
            accept = False
            if dE < 0:
                accept = True
            elif args.flip_mode == 'metropolis':
                accept = rng.random() < math.exp(-β*dE)
            else:
                try:
                    prob = 1.0 / (1.0 + math.exp(β*dE))
                except OverflowError:
                    prob = 0.0
                accept = rng.random() < prob

            if accept:
                spin[i] = s_new
                flipped[i] = True
                if dE != 0:
                    pi_phi[i] += math.copysign(math.sqrt(2*args.M_phi*abs(dE)), -dE)
            elif args.quantum_flip:
                spin_trial = -spin[i]
                spin[i] = spin_trial
                R_tunnel = resonance(np.full_like(neigh, i), neigh)
                spin[i] = -spin_trial
                dE_tunnel = (J*R_tunnel).sum() - E_old
                Δω_local = abs(omega[i] - omega[neigh].mean())
                p_tunnel = args.tunnel_delta * math.exp(-args.tunnel_alpha * Δω_local / delta_omega_star) * args.dt
                if rng.random() < p_tunnel:
                    spin[i] = spin_trial
                    flipped[i] = True
                    if dE_tunnel != 0:
                        pi_phi[i] += math.copysign(math.sqrt(2*args.M_phi*abs(dE_tunnel)), -dE_tunnel)

    if tick % args.save_every == 0:
        energy = total_energy()
        dE_frac = (energy - E0) / E0
        flip_rate = flipped.mean()
        pair_R = resonance(np.array([0]), np.array([1]))[0]
        chsh_val = chsh_pair()
        print(f"[{tick:6d}] E={energy:10.4f}  dE/E0={dE_frac:+.2e}  flips={flip_rate:.3f}  R01={pair_R:.3f}  CHSH={chsh_val:.3f}")
        np.savez(cfgdir/f"snap_{tick:06d}.npz", omega=omega, phi=phi, spin=spin,
                 pi_omega=pi_omega, pi_phi=pi_phi,
                 flipped=flipped, energy=energy, chsh=chsh_val)
        with open(logfile, 'a') as f:
            f.write(f"{tick},{energy:.6f},{dE_frac:.6e},{flip_rate:.4f},{pair_R:.4f},{chsh_val:.4f}\n")

# ─────────────────────────── Final Report ────────────────────────────────
elapsed = time.time() - t0
print(f"\nMD run complete — {args.ticks} ticks in {elapsed/60:.2f} min (seed {args.seed}).")
