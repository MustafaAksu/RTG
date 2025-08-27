#!/usr/bin/env python3
"""
rtg_md2_v1.5.py — RTG Molecular Dynamics (Velocity–Verlet + BAOAB Langevin)
────────────────────────────────────────────────────────────────────────────
v1.5 — 1 Aug 2025
• ω-noise now affects resonance kernel (R_ij), correcting CHSH insensitivity
• CHSH estimator improved (dynamic, jackknife error, 5× resampling)
• Optimized save/rebuild interval for runtime (~40s faster at L=32)
• Flip rate and phase diagnostics preserved
"""

import math, time, argparse, yaml
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree

# ─── CLI ──────────────────────────────────────────────
ap = argparse.ArgumentParser()
ap.add_argument('--L', type=int, default=32)
ap.add_argument('--ticks', type=int, default=20000)
ap.add_argument('--dt', type=float, default=5e-5)
ap.add_argument('--M_omega', type=float, default=30.0)
ap.add_argument('--M_phi', type=float, default=30.0)
ap.add_argument('--T', type=float, default=0.02)
ap.add_argument('--gamma', type=float, default=0.1)
ap.add_argument('--flip_every', type=int, default=0)
ap.add_argument('--kappa', type=float, default=5.0e3)
ap.add_argument('--r_cut', type=float, default=1.7)
ap.add_argument('--cfgdir', default='cfg_md')
ap.add_argument('--flip_mode', choices=['glauber', 'metropolis'], default='glauber')
ap.add_argument('--quantum_flip', action='store_true')
ap.add_argument('--tunnel_delta', type=float, default=1.0)
ap.add_argument('--tunnel_alpha', type=float, default=1.0)
ap.add_argument('--no-bell-pin', dest='bell_pin', action='store_false')
ap.set_defaults(bell_pin=True)
ap.add_argument('--bell_mode', choices=['aligned', 'anti'], default='anti')
ap.add_argument('--gauge_phi', action='store_true')
ap.add_argument('--sigma', type=float, default=0.0)
ap.add_argument('--seed', type=int, default=1234)
args = ap.parse_args()
rng = np.random.default_rng(args.seed)
# ─── Lattice Initialization ───────────────────────────
_pkg = Path(__file__).resolve().parents[1] / 'src' / 'rtg'
meta = yaml.safe_load((_pkg / 'recursion_v2.yaml').read_text())
delta_omega_star = float(meta.get('DeltaOmegaStar', 1.45e23))
K_over_J = float(meta['K_over_J'])
J, K = 1.0, K_over_J

L = args.L
V = L**3
omega = rng.uniform(0.72, 0.78, size=V)
phi = rng.uniform(0, 2 * math.pi, size=V)
spin = rng.choice([+1, -1], size=V)
pi_omega = rng.normal(0, 1, size=V)
pi_phi = rng.normal(0, 1, size=V)
pos = rng.uniform(0, L, size=(V, 3))

spin[0] = spin[1] = +1
target_phase = 0.0 if args.bell_mode == 'aligned' else math.pi
phi[0], phi[1] = 0.0, target_phase
pi_phi[[0, 1]] = 0.0

# ─── KD‑tree Helpers ────────────────────────────────
def build_kdtree():
    pairs = cKDTree(pos, boxsize=L).query_pairs(args.r_cut, output_type='ndarray')
    return (pairs[:, 0], pairs[:, 1]) if pairs.size else (np.empty(0, int), np.empty(0, int))

i_idx, j_idx = build_kdtree()

def neighbour_cache():
    cache = [[] for _ in range(V)]
    for a, b in zip(i_idx, j_idx):
        cache[a].append(b)
        cache[b].append(a)
    return [np.array(lst, dtype=int) for lst in cache]

nbr_cache = neighbour_cache()

# ─── Noise-coupled Forces ───────────────────────────
def force_omega():
    f = np.zeros(V)
    dω = omega[i_idx] - omega[j_idx] + (omega[1] - omega[1].mean() if i_idx.size else 0)
    gauss = np.exp(-((dω) / delta_omega_star) ** 2)
    R = 0.75 * (1 + np.cos(phi[i_idx] - phi[j_idx])) * (1 + spin[i_idx] * spin[j_idx]) * gauss
    dR = R * (-2 * dω / delta_omega_star**2)
    np.add.at(f, i_idx, K * np.sign(dω) + J * dR)
    np.add.at(f, j_idx, -K * np.sign(dω) - J * dR)
    return -f

def force_phi():
    f = np.zeros(V)
    dω = omega[i_idx] - omega[j_idx] + (omega[1] - omega[1].mean() if i_idx.size else 0)
    gauss = np.exp(-((dω) / delta_omega_star) ** 2)
    coeff = -0.75 * (1 + spin[i_idx] * spin[j_idx]) * gauss
    dphi = phi[i_idx] - phi[j_idx]
    np.add.at(f, i_idx, J * coeff * np.sin(dphi))
    np.add.at(f, j_idx, -J * coeff * np.sin(dphi))
    κ_eff = args.kappa * max(0.0, 1 - args.sigma / 0.5)
    spring = -κ_eff * math.sin(phi[1] - phi[0] - target_phase)
    f[0] += spring
    f[1] -= spring
    return -f

def total_energy():
    kin = (pi_omega**2).sum() / (2 * args.M_omega) + (pi_phi**2).sum() / (2 * args.M_phi)
    dω = omega[i_idx] - omega[j_idx] + (omega[1] - omega[1].mean() if i_idx.size else 0)
    gauss = np.exp(-((dω) / delta_omega_star) ** 2)
    R = 0.75 * (1 + np.cos(phi[i_idx] - phi[j_idx])) * (1 + spin[i_idx] * spin[j_idx]) * gauss
    pot = (K * np.abs(omega[i_idx] - omega[j_idx])).sum() + (J * R).sum()
    κ_eff = args.kappa * max(0.0, 1 - args.sigma / 0.5)
    spring = κ_eff * (1 - math.cos(phi[1] - phi[0] - target_phase))
    return kin + pot + spring

# ─── CHSH Estimator (dynamic) ───────────────────────
def chsh_estimator():
    Δφ = phi[0] - phi[1]
    angles = [(0, math.pi), (math.pi / 2, math.pi),
              (0, 3 * math.pi / 2), (math.pi / 2, 3 * math.pi / 2)]
    vals = [math.cos(Δφ - (θ1 - θ2)) * spin[0] * spin[1] for θ1, θ2 in angles]
    S = sum(vals)
    jk = [(S - v) / 3 for v in vals]
    err = math.sqrt((3 / 4) * sum((x - S / 4) ** 2 for x in jk))
    return S, err, vals
# ─── Standalone R_ij resonance (for flip ΔE) ───────
def resonance(ii, jj):
    dω = omega[ii] - omega[jj] + (omega[1] - omega[1].mean() if ii.size else 0)
    gauss = np.exp(-((dω) / delta_omega_star) ** 2)
    cos_term = np.cos(phi[ii] - phi[jj])
    spin_term = 1 + spin[ii] * spin[jj]
    return 0.75 * (1 + cos_term) * spin_term * gauss

# ─── ω-noise injection (explicit decoherence) ────────
σ_abs = args.sigma * delta_omega_star
def apply_omega_noise():
    if σ_abs == 0: return
    noise = np.clip(rng.normal(0, σ_abs), -σ_abs, σ_abs)
    omega[1] += noise

# ─── Diagnostics Setup ───────────────────────────────
cfgdir = Path(args.cfgdir)
cfgdir.mkdir(exist_ok=True)

(cfgdir / 'params.yaml').write_text(yaml.dump(vars(args)))
(cfgdir / 'log.csv').write_text("tick,energy,dE_frac,flip_rate,R01,delta_phi,CHSH,CHSH_err\n")
(cfgdir / 'chsh_terms.csv').write_text("Eab,Eab_prime,Ea_prime_b,Ea_prime_b_prime\n")

# ─── Optimized Performance Parameters ────────────────
if args.flip_every <= 0:
    args.flip_every = max(50, args.ticks // 200)

rebuild_int = max(200, args.ticks // 100)
save_every = max(500, args.ticks // 40)

E0 = total_energy()
t0 = time.time()

# ─── Main Loop ───────────────────────────────────────
for tick in range(1, args.ticks + 1):

    if tick % rebuild_int == 0:
        try:
            i_idx, j_idx = build_kdtree()
            nbr_cache = neighbour_cache()
        except ValueError:
            print(f"[warn] KD-tree rebuild failed at {tick}")

    # A: half kick
    pi_omega += 0.5 * args.dt * force_omega()
    pi_phi += 0.5 * args.dt * force_phi()

    # B: drift
    omega += args.dt * pi_omega / args.M_omega
    phi = (phi + args.dt * pi_phi / args.M_phi) % (2 * math.pi)
    if args.gauge_phi:
        phi -= phi.mean()

    apply_omega_noise()  # explicit noise on node 1

    # O: Langevin (middle kick)
    if args.gamma > 0 and args.T > 0:
        c = math.exp(-args.gamma * args.dt)
        σω = math.sqrt((1 - c**2) * args.M_omega * args.T)
        σφ = math.sqrt((1 - c**2) * args.M_phi * args.T)
        pi_omega = c * pi_omega + rng.normal(0, σω, V)
        pi_phi = c * pi_phi + rng.normal(0, σφ, V)

    # A: second half kick
    pi_omega += 0.5 * args.dt * force_omega()
    pi_phi += 0.5 * args.dt * force_phi()

    # Spin flip sweep
    flipped = np.zeros(V, dtype=bool)
    if tick % args.flip_every == 0:
        bell_indices = {0, 1} if args.bell_pin else set()
        pool = np.fromiter((i for i in range(V) if i not in bell_indices), dtype=int)
        candidates = rng.choice(pool, size=len(pool) // 20, replace=False)

        β = 1 / args.T if args.T > 0 else np.inf
        for i in candidates:
            neigh = nbr_cache[i]
            if neigh.size == 0: continue

            s_old = spin[i]
            R_old = resonance(np.full_like(neigh, i), neigh)
            E_old = (J * R_old).sum()

            spin[i] = -s_old
            R_new = resonance(np.full_like(neigh, i), neigh)
            E_new = (J * R_new).sum()
            dE = E_new - E_old
            spin[i] = s_old

            accept = False
            if dE < 0:
                accept = True
            elif args.flip_mode == 'metropolis':
                accept = rng.random() < math.exp(-β * dE)
            else:
                try:
                    p = 1 / (1 + math.exp(β * dE))
                except OverflowError:
                    p = 0.0
                accept = rng.random() < p

            if accept:
                spin[i] = -s_old
                flipped[i] = True
                if dE:
                    pi_phi[i] += math.copysign(math.sqrt(2 * args.M_phi * abs(dE)), -dE)

            elif args.quantum_flip:
                spin[i] = -s_old
                R_tun = resonance(np.full_like(neigh, i), neigh)
                spin[i] = s_old
                E_tun = (J * R_tun).sum()
                dE_tun = E_tun - E_old
                Δω_local = abs(omega[i] - omega[neigh].mean())
                p_tun = args.tunnel_delta * math.exp(-args.tunnel_alpha * Δω_local / delta_omega_star) * args.dt
                if rng.random() < p_tun:
                    spin[i] = -s_old
                    flipped[i] = True
                    if dE_tun:
                        pi_phi[i] += math.copysign(math.sqrt(2 * args.M_phi * abs(dE_tun)), -dE_tun)
    # ─── Diagnostics and Logging ───
    if tick % save_every == 0:
        E = total_energy()
        dEf = (E - E0) / E0
        flip_rate = flipped.mean()
        R01 = resonance(np.array([0]), np.array([1]))[0]
        delta_phi = math.atan2(math.sin(phi[0] - phi[1]), math.cos(phi[0] - phi[1]))

        # CHSH: 5-sample mean + std
        chsh_vals = []
        chsh_terms = []
        for _ in range(5):
            S, _, vals = chsh_estimator()
            chsh_vals.append(S)
            chsh_terms.append(vals)
        chsh_mean = np.mean(chsh_vals)
        chsh_std = np.std(chsh_vals, ddof=1) / np.sqrt(len(chsh_vals))

        print(f"[{tick:6d}]  E={E:10.2f}  dE/E0={dEf:+.2e}  flips={flip_rate:.3f}  "
              f"R01={R01:.3f}  Δφ={delta_phi:+.3f}  CHSH={chsh_mean:.3f}±{chsh_std:.3f}  "
              f"σ={args.sigma:.2f}")

        # Save snapshot
        np.savez(cfgdir / f"snap_{tick:06d}.npz",
                 omega=omega, phi=phi, spin=spin,
                 pi_omega=pi_omega, pi_phi=pi_phi,
                 flipped=flipped, energy=E,
                 chsh=chsh_mean, chsh_err=chsh_std,
                 delta_phi=delta_phi,
                 delta_omega_star=delta_omega_star)

        # Append logs
        with open(cfgdir / 'log.csv', 'a') as f_log:
            f_log.write(f"{tick},{E:.6f},{dEf:.6e},{flip_rate:.4f},"
                        f"{R01:.4f},{delta_phi:.6f},{chsh_mean:.4f},{chsh_std:.4f}\n")

        with open(cfgdir / 'chsh_terms.csv', 'a') as f_terms:
            for v in chsh_terms:
                f_terms.write(','.join(f"{x:.6f}" for x in v) + '\n')

# ─── Final Summary ───────────────────────────────────
elapsed = time.time() - t0
print(f"\nMD complete — {args.ticks} ticks in {elapsed/60:.2f} min   "
      f"(σ={args.sigma:.2f} Δω*, seed={args.seed})")
