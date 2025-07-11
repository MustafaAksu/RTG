#!/usr/bin/env python
"""
RTG-v2 Hybrid-Monte-Carlo on a 32³ lattice (dimension-less ω̃)

  * K / J  = 0.27  (recursion_v2.yaml)
  * Resonance
        R = 0.75 (1 + cos Δφ) (1 + sᵢ sⱼ)  exp[-(Δω̃)²]
  * Bell pair (sites 0,1) pinned at Δφ = π  with a wrap-safe spring

Typical CPU run
    python hmc_proton32.py --traj 2000 --step 2e-4 --nlf 20
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
ap.add_argument('--step', type=float, default=5e-5)  # Lowered for better acc
ap.add_argument('--nlf',  type=int,   default=20)
ap.add_argument('--save_every', type=int, default=2)
ap.add_argument('--cfgdir', default='cfg')
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
s = rng.choice([+1, -1], size=V).astype(xp.int8)

# Bell pair initialisation
s[0], s[1] = +1, -1
φ[0], φ[1] = 0.0, math.pi

# Curvature: normals (flat default; [nx, ny, nz])
n = xp.tile([0.0, 0.0, 1.0], (V, 1))

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
    cos   = xp.cos(φ[i] - φ[j])
    # Spin with curvature θ_ij (from notes; flat=0 for now)
    dot = xp.einsum('ik,ik->i', n[i], n[j])  # Batched dot
    theta = xp.arccos(xp.clip(dot, -1.0, 1.0))
    exp_i_theta = xp.exp(1j * theta)
    spin_prod = s[i] * s[j] * exp_i_theta  # Complex
    spin = 1 + xp.real(spin_prod)
    gauss = xp.exp(-(ω[i] - ω[j])**2)
    return 0.75 * (1 + cos) * spin * gauss

def total_E():
    Δω = xp.abs(ω[i_idx] - ω[j_idx])
    return xp.sum(K*Δω + J*resonance(i_idx, j_idx))

def F_ω():
    f  = xp.zeros_like(ω)
    dw = ω[i_idx] - ω[j_idx]
    # kinetic |Δω|
    xp.add.at(f, i_idx,  K * xp.sign(dw))
    xp.add.at(f, j_idx, -K * xp.sign(dw))
    # resonance
    R  = resonance(i_idx, j_idx)
    dR = R * (-2 * dw)
    xp.add.at(f, i_idx,  J * dR)
    xp.add.at(f, j_idx, -J * dR)
    return f

def F_φ():
    f    = xp.zeros_like(φ)
    sin  = xp.sin(φ[i_idx] - φ[j_idx])
    coef = -0.75 * (1 + s[i_idx]*s[j_idx]) * xp.exp(-(ω[i_idx] - ω[j_idx])**2)
    xp.add.at(f, i_idx,  J * coef * sin)
    xp.add.at(f, j_idx, -J * coef * sin)

    # ---- wrap-safe Bell-pair pin ------------------------------------
    kappa = 5.0e5  # Boosted for tighter pinning
    angle = (φ[1] - φ[0]) - math.pi          # want → 0
    spring = kappa * xp.sin(angle)           # ∂/∂φ of (-cos)
    f[0] += spring
    f[1] -= spring
    return f

# ---------- HMC loop --------------------------------------------------
cfgdir = Path(args.cfgdir); cfgdir.mkdir(exist_ok=True)
dt  = args.step
acc = 0
t0  = time.time()

for tr in range(1, args.traj + 1):
    pω = rng.normal(0, 1, size=V)
    pφ = rng.normal(0, 1, size=V)

    ω0, φ0, pω0, pφ0 = ω.copy(), φ.copy(), pω.copy(), pφ.copy()
    H_old = xp.sum(pω0**2 + pφ0**2)/2 + total_E()

    # leap-frog
    pω -= 0.5*dt*F_ω();  pφ -= 0.5*dt*F_φ()
    for _ in range(args.nlf):
        ω += dt * pω
        φ += dt * pφ
        φ %= 2 * math.pi
        pω -= dt * F_ω()
        pφ -= dt * F_φ()
    pω -= 0.5*dt*F_ω();  pφ -= 0.5*dt*F_φ()
    pω *= -1;  pφ *= -1

    H_new = xp.sum(pω**2 + pφ**2)/2 + total_E()
    dH    = float(H_new - H_old)

    if rng.random() < math.exp(-dH):         # accept
        acc += 1
        if acc % args.save_every == 0:
            np.savez_compressed(cfgdir/f"cfg_{acc:05d}.npz",
                                omega=np.asarray(ω),
                                phi=np.asarray(φ),
                                spin=np.asarray(s))
    else:                                    # reject
        ω[:], φ[:] = ω0, φ0

    if tr % 10 == 0:
        avg_R = float(xp.mean(resonance(i_idx, j_idx)))  # Diagnostic
        E_tot = float(total_E())  # Equilibration monitor
        print(f"traj {tr:5d}/{args.traj}  ΔH={dH:+.3e}  acc={acc/tr:6.3f}  <R>={avg_R:.3f}  E_tot={E_tot:.3e}")

t1 = time.time()
print("\n---- run summary ------------------------------")
print(f"Lattice      : {L}^3   nodes = {V}")
print(f"Traj/steps   : {args.traj} / {args.nlf}")
print(f"Acceptance   : {acc/args.traj:.3f}")
print(f"Configs kept : {len(list(cfgdir.glob('cfg_*.npz')))}")
print(f"Wall-time    : {t1 - t0:.1f} s  "
      f"({'GPU' if args.cupy and HAVE_CUPY else 'CPU'})")