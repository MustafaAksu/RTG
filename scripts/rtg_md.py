#!/usr/bin/env python
"""
Real-time MD for RTG-v2 (binary spin kernel).
Velocity-Verlet + optional Langevin thermostat + optional Ising flips.
"""

import numpy as np, math, argparse, time, yaml
from pathlib import Path

# ── CLI ──────────────────────────────────────────────────────────────
ap = argparse.ArgumentParser()
ap.add_argument('--L',           type=int,   default=32)
ap.add_argument('--ticks',       type=int,   default=10_000)
ap.add_argument('--dt',          type=float, default=5e-5)
ap.add_argument('--M_omega',     type=float, default=100.0)
ap.add_argument('--M_phi',       type=float, default=100.0)
ap.add_argument('--flip_every',  type=int,   default=0,    # 0 → no flips
                help='flip half the spins every N ticks')
ap.add_argument('--T',           type=float, default=0.0,  # Langevin T
                help='thermostat temperature (0 = off)')
ap.add_argument('--gamma',       type=float, default=0.1,  help='Langevin friction γ')
ap.add_argument('--kappa',       type=float, default=1.0e4,help='Bell-pair spring')  # Lowered as before
ap.add_argument('--save_every',  type=int,   default=1000)
ap.add_argument('--cfgdir',      default='cfg_md')
args = ap.parse_args()

rng = np.random.default_rng(1234)

# ── constants (from recursion_v2.yaml) ───────────────────────────────
_pkg = Path(__file__).resolve().parents[1] / 'src' / 'rtg'
K_over_J = float(yaml.safe_load((_pkg/'recursion_v2.yaml').read_text())['K_over_J'])
J, K = 1.0, K_over_J
Δω_star = 1.0

# ── lattice fields ───────────────────────────────────────────────────
L, V = args.L, args.L**3
ω     = rng.uniform(0.72, 0.78, size=V)
φ     = rng.uniform(0, 2*math.pi, size=V)
s     = rng.choice([1, -1], size=V)             # ±1  ≡  ±i
π_ω   = rng.normal(0, 1, size=V)
π_φ   = rng.normal(0, 1, size=V)
M_ω, M_φ = float(args.M_omega), float(args.M_phi)

cfgdir = Path(args.cfgdir); cfgdir.mkdir(exist_ok=True)

# ── neighbour list (static cubic for now) ────────────────────────────
def nn_pairs(L):
    idx = np.arange(L**3).reshape(L, L, L)
    out = []
    for x in range(L):
        for y in range(L):
            for z in range(L):
                i = idx[x,y,z]
                out += [(i, idx[(x+1)%L, y, z]),
                        (i, idx[x,(y+1)%L, z]),
                        (i, idx[x, y,(z+1)%L])]
    return np.asarray(out, np.int32)
pairs = nn_pairs(L)
i_idx, j_idx = pairs.T

# ── helper: resonance factor ─────────────────────────────────────────
def resonance(i, j):
    cos_dp = np.cos(φ[i] - φ[j])
    spin   = 1 + s[i]*s[j]                        # binary
    gauss  = np.exp(-((ω[i]-ω[j]) / Δω_star)**2)
    return 0.75 * (1 + cos_dp) * spin * gauss

# ── forces (return −∂H/∂q  = physical force) ────────────────────────
def F_ω():
    f  = np.zeros_like(ω)
    dω = ω[i_idx] - ω[j_idx]
    np.add.at(f, i_idx,  K*np.sign(dω))
    np.add.at(f, j_idx, -K*np.sign(dω))
    R  = resonance(i_idx, j_idx)
    dR = R * (-2 * dω / Δω_star**2)
    np.add.at(f, i_idx,  J*dR)
    np.add.at(f, j_idx, -J*dR)
    return -f

def F_φ():
    f   = np.zeros_like(φ)
    sin = np.sin(φ[i_idx] - φ[j_idx])
    coeff = -0.75*(1+s[i_idx]*s[j_idx])*np.exp(-((ω[i_idx]-ω[j_idx])/Δω_star)**2)
    np.add.at(f, i_idx,  J*coeff*sin)
    np.add.at(f, j_idx, -J*coeff*sin)
    # Bell-pair spring
    angle = (φ[1] - φ[0]) - math.pi
    spring = -args.kappa * math.sin(angle)  # Sign fixed here
    f[0] += spring;  f[1] -= spring
    return -f

# ── full Hamiltonian (for drift check) ───────────────────────────────
def total_E():
    kin = (π_ω**2).sum()/(2*M_ω) + (π_φ**2).sum()/(2*M_φ)
    pot = (K*np.abs(ω[i_idx]-ω[j_idx])).sum() + (J*resonance(i_idx,j_idx)).sum()
    V_spring = args.kappa * (1 - math.cos((φ[1]-φ[0]) - math.pi))
    return kin + pot + V_spring

# ── MD main loop ─────────────────────────────────────────────────────
dt, γ, T = args.dt, args.gamma, args.T
E0 = total_E();  t0 = time.time()

for tick in range(1, args.ticks+1):

    # first half-kick
    π_ω += 0.5*dt * F_ω()
    π_φ += 0.5*dt * F_φ()

    # drift
    ω  += dt * π_ω / M_ω
    φ  += dt * π_φ / M_φ
    φ %= 2*math.pi

    # spin flips
    if args.flip_every and tick % args.flip_every == 0:
        mask = rng.random(V) < 0.5
        s[mask] *= -1

    # Langevin thermostat (optional)
    if T > 0.0:
        σ = math.sqrt(2*γ*T*dt)
        π_ω += -γ*π_ω*dt + σ*rng.normal(0,1,size=V)
        π_φ += -γ*π_φ*dt + σ*rng.normal(0,1,size=V)

    # second half-kick
    π_ω += 0.5*dt * F_ω()
    π_φ += 0.5*dt * F_φ()

    # logging
    if tick % 1000 == 0 or tick == args.ticks:
        Etot = total_E()
        print(f"tick {tick:6d}/{args.ticks}  E_tot={Etot:.6e}  drift={(Etot-E0)/E0:+.3e}")

    # snapshot
    if tick % args.save_every == 0:
        np.savez(cfgdir/f"snap_{tick:06d}.npz",
                 omega=ω, phi=φ, spin=s, pi_omega=π_ω, pi_phi=π_φ)

print(f"\nFinished {args.ticks} ticks in {time.time()-t0:.1f}s"
      f"  (final drift {(total_E()-E0)/E0:+.3e})")