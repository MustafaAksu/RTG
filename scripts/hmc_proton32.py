#!/usr/bin/env python
"""
RTG-v2 Hybrid-Monte-Carlo lattice demo
-------------------------------------
* Uses constants_v2.yaml (Δω* = 5.8e22 s⁻¹) and recursion_v2.yaml (K/J = 0.27)
* Binary spins  s = ±i  → factor (1 + s_i s_j)
* Resonance     𝓡_{ij} = 1.5 (1 + cosΔφ)(1 + s_i s_j) exp[-((Δω)/Δω*)²]
* Action        Σ_{<ij>} [ K|Δω| + J 𝓡_{ij} ]

Run examples
------------
CPU : python scripts/hmc_proton32.py --L 32 --traj 200 --step 0.08 --nlf 20
GPU : python scripts/hmc_proton32.py --L 64 --traj 500 --step 0.04 --nlf 20 --cupy
"""

import argparse, math, time, yaml, numpy as np
try:
    import cupy as cp
    HAVE_CUPY = True
except ImportError:
    HAVE_CUPY = False

# ---------- CLI -------------------------------------------------------------
ap = argparse.ArgumentParser()
ap.add_argument('--L',    type=int,   default=32,  help='lattice size')
ap.add_argument('--traj', type=int,   default=200, help='HMC trajectories')
ap.add_argument('--step', type=float, default=0.08,help='leap-frog Δτ')
ap.add_argument('--nlf',  type=int,   default=20,  help='leap-frog steps/τ')
ap.add_argument('--cupy', action='store_true',     help='use GPU via CuPy')
args = ap.parse_args()

xp = cp if (args.cupy and HAVE_CUPY) else np
if args.cupy and not HAVE_CUPY:
    print("CuPy not found – running on CPU")

# ---------- constants -------------------------------------------------------
from pathlib import Path
_pkg = Path(__file__).resolve().parents[1] / 'src' / 'rtg'
C = yaml.safe_load((_pkg / 'constants_v2.yaml').read_text())
R = yaml.safe_load((_pkg / 'recursion_v2.yaml').read_text())
Δω_star  = float(C['delta_omega_star'])
K_over_J = float(R['K_over_J'])
J, K     = 1.0, K_over_J                           # set J = 1 unit

# ---------- lattice fields --------------------------------------------------
L, N = args.L, args.L**3
rng  = xp.random.default_rng(42)

ω = rng.uniform(0.7*Δω_star, 0.8*Δω_star, size=N).astype(xp.float64)
φ = rng.uniform(0, 2*math.pi,            size=N).astype(xp.float64)
s = rng.choice([+1, -1],                 size=N).astype(xp.int8)   # ±1 ≡ ±i

# nearest-neighbour bonds (+x, +y, +z) so each pair appears once
def nn_pairs(L: int):
    idx = np.arange(L**3).reshape(L, L, L)
    out = []
    for x in range(L):
        for y in range(L):
            for z in range(L):
                i = idx[x, y, z]
                out += [
                    (i, idx[(x+1)%L, y, z]),
                    (i, idx[x, (y+1)%L, z]),
                    (i, idx[x, y, (z+1)%L])
                ]
    return np.asarray(out, dtype=np.int32)

pairs     = xp.asarray(nn_pairs(L))
i_idx, j_idx = pairs.T                     # cached views

# ---------- energy & forces --------------------------------------------------
def resonance(i, j):
    cos_dp = xp.cos(φ[i] - φ[j])
    spin   = 1 + s[i]*s[j]
    gauss  = xp.exp(-((ω[i] - ω[j])/Δω_star)**2)
    return 1.5 * (1 + cos_dp) * spin * gauss

def total_E():
    Δω = xp.abs(ω[i_idx] - ω[j_idx])
    return xp.sum(K * Δω + J * resonance(i_idx, j_idx))

def force_ω():
    f = xp.zeros_like(ω)

    # linear K|Δω| contribution
    dw = ω[i_idx] - ω[j_idx]
    xp.add.at(f, i_idx,  K * xp.sign(dw))
    xp.add.at(f, j_idx, -K * xp.sign(dw))

    # J * 𝓡 derivative
    R  = resonance(i_idx, j_idx)
    dR = R * (-2 * dw / (Δω_star**2))
    xp.add.at(f, i_idx,  J * dR)
    xp.add.at(f, j_idx, -J * dR)
    return f

# ---------- HMC loop --------------------------------------------------------
accept = 0
start  = time.time()

for tr in range(args.traj):
    p  = rng.normal(0, 1, size=N).astype(xp.float64)
    ω0, p0 = ω.copy(), p.copy()

    # ---- OLD Hamiltonian (before update) -----------------------------------
    H_old = xp.sum(p0**2)/2 + total_E()

    # ---- leap-frog integrator ---------------------------------------------
    p -= 0.5 * args.step * force_ω()
    for _ in range(args.nlf):
        ω += args.step * p
        p -= args.step * force_ω()
    p -= 0.5 * args.step * force_ω()
    p *= -1                        # momentum reversal
    # -----------------------------------------------------------------------

    H_new = xp.sum(p**2)/2 + total_E()
    dH = float(H_new - H_old)

    if rng.random() < math.exp(-dH):
        accept += 1                # keep new ω
    else:
        ω[:] = ω0                  # revert

    if (tr+1) % 10 == 0:
        print(f"traj {tr+1:4d}/{args.traj}  ΔH={dH:+.3f}  acc={(accept/(tr+1))*100:5.1f}%")

# ---------- summary ---------------------------------------------------------
end = time.time()
print("\n---- run summary ------------------------------")
print(f"Lattice      : {L}^3  nodes = {N}")
print(f"Traj / steps : {args.traj} / {args.nlf}")
print(f"Acceptance   : {accept/args.traj:.3f}")
print(f"Wall-time    : {end-start:.1f} s  ({'GPU' if xp is cp else 'CPU'})")
