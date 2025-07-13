#!/usr/bin/env python
"""
RTG-v2 Hybrid-Monte-Carlo on a 32³ lattice (dimension-less ω̃, continuous spin)

  * K / J  = 0.27  (recursion_v2.yaml)
  * Resonance
        R = 0.75 (1 + cos Δφ) (1 + cos(θi + θj))  exp[-(Δω̃)²]
  * Bell pair (sites 0,1) pinned at Δφ = π  with a wrap-safe spring

Typical CPU run
    python hmc_contspin32.py --traj 2000 --step 2e-4 --nlf 20
GPU (CuPy) users double the step:
    ... --cupy --step 4e-4
"""
# ---------------------------------------------------------------------
import argparse, math, time, yaml
from pathlib import Path
import numpy as np
try:
    import cupy as cp
    HAVE_CUPY = True
except ImportError:
    HAVE_CUPY = False
    cp = np                        # dummy so print() still OK

# ---------- CLI -------------------------------------------------------
ap = argparse.ArgumentParser()
ap.add_argument('--L',    type=int,   default=32)
ap.add_argument('--traj', type=int,   default=2000)
ap.add_argument('--step', type=float, default=2e-4)
ap.add_argument('--nlf',  type=int,   default=20)
ap.add_argument('--save_every', type=int, default=2)
ap.add_argument('--cfgdir', default='cfg_cont')
ap.add_argument('--cupy',  action='store_true')
args = ap.parse_args()

xp = cp if (args.cupy and HAVE_CUPY) else np
if args.cupy and not HAVE_CUPY:
    print("CuPy not found – running on CPU/NumPy")

# ---------- constants -------------------------------------------------
_pkg = Path(__file__).resolve().parents[1] / 'src' / 'rtg'
K_over_J = float(yaml.safe_load(
        (_pkg / 'recursion_v2.yaml').read_text())['K_over_J'])   # 0.27
J, K = 1.0, K_over_J                                              # J = 1 units

# ---------- lattice & RNG --------------------------------------------
L  = args.L
V  = L**3
rng = xp.random.default_rng(1234)

ω = rng.uniform(0.72, 0.78, size=V)           # dimension-less ω̃
φ = rng.uniform(0.0, 2*math.pi, size=V)
theta = rng.uniform(-math.pi, math.pi, size=V)  # Continuous spin angles
s_cos = xp.cos(theta)  # Re(s/i)
s_sin = xp.sin(theta)  # Im(s/i)

# Bell pair initialisation (spins fixed anti-aligned via initial θ=0,π or similar; but since quenched, no s[0],s[1] = +1,-1)
φ[0], φ[1] = 0.0, math.pi

def nn_pairs(L: int):
    idx  = np.arange(L**3).reshape(L, L, L)
    pair = []
    for x in range(L):
        for y in range(L):
            for z in range(L):
                i = idx[x, y, z]
                pair += [(i, idx[(x+1)%L, y, z]),
                         (i, idx[x,(y+1)%L, z]),
                         (i, idx[x, y,(z+1)%L])]
    return np.asarray(pair, np.int32)

pairs        = xp.asarray(nn_pairs(L))
i_idx, j_idx = pairs.T

# ---------- energy & forces ------------------------------------------
def resonance(i, j):
    cos = xp.cos(φ[i] - φ[j])
    spin = 1 + (s_cos[i]*s_cos[j] - s_sin[i]*s_sin[j])  # 1 + cos(θi + θj)
    gauss = xp.exp(-((ω[i] - ω[j])**2))
    return 0.75 * (1 + cos) * spin * gauss

def total_E():
    dw = xp.abs(ω[i_idx] - ω[j_idx])
    return xp.sum(K * dw + J * resonance(i_idx, j_idx))

def F_ω():
    f  = xp.zeros_like(ω)
    dw = ω[i_idx] - ω[j_idx]
    # kinetic
    xp.add.at(f, i_idx,  K * xp.sign(dw))
    xp.add.at(f, j_idx, -K * xp.sign(dw))
    # resonance
    R  = resonance(i_idx, j_idx)
    dR = R * (-2 * dw)
    xp.add.at(f, i_idx,  J * dR)
    xp.add.at(f, j_idx, -J * dR)
    return f

def F_φ():
    f = xp.zeros_like(φ)
    sin   = xp.sin(φ[i_idx] - φ[j_idx])
    coeff = -0.75 * (1 + (s_cos[i_idx]*s_cos[j_idx] - s_sin[i_idx]*s_sin[j_idx])) * xp.exp(-((ω[i_idx]-ω[j_idx])**2))
    xp.add.at(f, i_idx,  J * coeff * sin)
    xp.add.at(f, j_idx, -J * coeff * sin)
    # soft pin Δφ(1,0) → π
    kappa = 5.0e5
    angle = (φ[1] - φ[0]) - math.pi
    spring = kappa * xp.sin(angle)
    f[0] += spring
    f[1] -= spring
    return f

# ---------- HMC main loop --------------------------------------------
cfgdir = Path(args.cfgdir); cfgdir.mkdir(exist_ok=True)
dt  = args.step
acc = 0
t0  = time.time()

for tr in range(1, args.traj + 1):
    pω = rng.normal(0, 1, size=V)
    pφ = rng.normal(0, 1, size=V)

    ω0, φ0, pω0, pφ0 = ω.copy(), φ.copy(), pω.copy(), pφ.copy()
    H_old = xp.sum(pω0**2 + pφ0**2) / 2 + total_E()

    # leap-frog
    pω -= 0.5*dt*F_ω();  pφ -= 0.5*dt*F_φ()
    for _ in range(args.nlf):
        ω += dt * pω
        φ += dt * pφ
        φ %= 2.0 * math.pi
        pω -= dt * F_ω()
        pφ -= dt * F_φ()
    pω -= 0.5*dt*F_ω();  pφ -= 0.5*dt*F_φ()
    pω *= -1;  pφ *= -1        # reversibility

    H_new = xp.sum(pω**2 + pφ**2) / 2 + total_E()
    dH = float(H_new - H_old)

    if rng.random() < math.exp(-dH):
        acc += 1
        if acc % args.save_every == 0:
            np.savez_compressed(cfgdir / f"cfg_{acc:05d}.npz",
                                omega=np.asarray(ω),
                                phi=np.asarray(φ),
                                theta=np.asarray(theta))  # Save theta for analysis
    else:
        ω[:], φ[:] = ω0, φ0

    if tr % 10 == 0:
        print(f"traj {tr:5d}/{args.traj}  ΔH={dH:+.3e}  acc={acc/tr:6.3f}")

t1 = time.time()
backend = 'GPU' if (args.cupy and HAVE_CUPY) else 'CPU'
print("\n---- run summary ------------------------------")
print(f"Lattice      : {L}^3   nodes = {V}")
print(f"Traj / steps : {args.traj} / {args.nlf}")
print(f"Acceptance   : {acc/args.traj:.3f}")
print(f"Configs kept : {len(list(cfgdir.glob('cfg_*.npz')))}")
print(f"Wall-time    : {t1-t0:.1f} s  ({backend})")