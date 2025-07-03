#!/usr/bin/env python
"""
RTG-v2 Hybrid-Monte-Carlo demo (32³ default)

* Δω⋆  = 5.8e22  s⁻¹   (constants_v2.yaml)
* a_eff = 0.0808 fm    after 6 block-spin steps
* K/J  = 0.27          (recursion_v2.yaml)
* Resonance kernel
    R = 0.75 (1+cosΔφ)(1+s_i s_j) exp[-(Δω/Δω⋆)²]

Bell pair: lattice sites 0 (+i, φ=0) and 1 (–i, φ=π) kept at Δφ = π.

Example CPU run
---------------
python scripts/hmc_proton32.py --traj 2000 --step 2e-4 --nlf 20 \
       --save_every 2 --cfgdir cfg_lock

GPU (needs CuPy):
python scripts/hmc_proton32.py --cupy --traj 4000 --step 4e-4 --nlf 20
"""
# ----------------------------------------------------------------------
import argparse, math, time, yaml
from pathlib import Path
import numpy as np
try:
    import cupy as cp
    HAVE_CUPY = True
except ImportError:
    HAVE_CUPY = False

# ---------- CLI --------------------------------------------------------
ap = argparse.ArgumentParser()
ap.add_argument('--L',    type=int,   default=32,   help='lattice size')
ap.add_argument('--traj', type=int,   default=2000, help='HMC trajectories')
ap.add_argument('--step', type=float, default=2e-4, help='leap-frog Δτ')
ap.add_argument('--nlf',  type=int,   default=20,   help='leap-frog steps/τ')
ap.add_argument('--save_every', type=int, default=2,
                help='keep every N-th accepted cfg')
ap.add_argument('--cfgdir', default='cfg',
                help='output directory for configs')
ap.add_argument('--cupy',  action='store_true', help='run on GPU via CuPy')
args = ap.parse_args()

xp = cp if (args.cupy and HAVE_CUPY) else np
if args.cupy and not HAVE_CUPY:
    print("CuPy not found – falling back to NumPy CPU")

# ---------- load constants --------------------------------------------
_pkg = Path(__file__).resolve().parents[1] / 'src' / 'rtg'
C = yaml.safe_load((_pkg / 'constants_v2.yaml').read_text())
R = yaml.safe_load((_pkg / 'recursion_v2.yaml').read_text())
Δω_star  = float(C['delta_omega_star'])
K_over_J = float(R['K_over_J'])       # already 0.27 in repo

J, K = 1.0, K_over_J                  # choose energy units with J = 1

# ---------- lattice & RNG ---------------------------------------------
L  = args.L
V  = L**3
rng = xp.random.default_rng(1234)

# fields (ω, φ, s)
ω  = rng.uniform(0.72*Δω_star, 0.78*Δω_star, size=V)
φ  = rng.uniform(0.0, 2*math.pi,             size=V)
s  = rng.choice([+1, -1], size=V).astype(xp.int8)

# Bell pair initialisation
s[0], s[1] = +1, -1
φ[0], φ[1] = 0.0, math.pi

# nearest-neighbour pairs (+x,+y,+z) – once each
def nn_pairs(L):
    idx = np.arange(L**3).reshape(L, L, L)
    pairs = []
    for x in range(L):
        for y in range(L):
            for z in range(L):
                i = idx[x,y,z]
                pairs += [(i, idx[(x+1)%L, y, z]),
                          (i, idx[x,(y+1)%L, z]),
                          (i, idx[x, y, (z+1)%L])]
    return np.asarray(pairs, np.int32)

pairs      = xp.asarray(nn_pairs(L))
i_idx, j_idx = pairs.T

# ---------- energy & forces -------------------------------------------
def resonance(i, j):
    cos = xp.cos(φ[i]-φ[j])
    spin = 1 + s[i]*s[j]
    gauss = xp.exp(-((ω[i]-ω[j])/Δω_star)**2)
    return 0.75 * (1 + cos) * spin * gauss

def total_E():
    Δω = xp.abs(ω[i_idx]-ω[j_idx])
    return xp.sum(K*Δω + J*resonance(i_idx, j_idx))

def F_ω():
    f = xp.zeros_like(ω)
    dw = ω[i_idx]-ω[j_idx]
    # kinetic |Δω|
    xp.add.at(f, i_idx,  K*xp.sign(dw))
    xp.add.at(f, j_idx, -K*xp.sign(dw))
    # resonance
    R  = resonance(i_idx, j_idx)
    dR = R * (-2*dw/Δω_star**2)
    xp.add.at(f, i_idx,  J*dR)
    xp.add.at(f, j_idx, -J*dR)
    return f

def F_φ():
    f = xp.zeros_like(φ)
    sin = xp.sin(φ[i_idx]-φ[j_idx])
    coeff = -0.75*(1+s[i_idx]*s[j_idx]) \
            * xp.exp(-((ω[i_idx]-ω[j_idx])/Δω_star)**2)
    xp.add.at(f, i_idx,  J*coeff*sin)
    xp.add.at(f, j_idx, -J*coeff*sin)
    # ---- stiff pin: keep Δφ(1,0) = π -----------------
    kappa = 5.0e4
    diff  = (φ[1]-φ[0]) - math.pi
    f[0] += kappa*diff
    f[1] -= kappa*diff
    return f

# ---------- HMC main loop ---------------------------------------------
cfgdir = Path(args.cfgdir); cfgdir.mkdir(exist_ok=True)
dt   = args.step
acc  = 0
t0   = time.time()

for tr in range(1, args.traj+1):
    # momenta
    pω = rng.normal(0, 1, size=V)
    pφ = rng.normal(0, 1, size=V)

    ω0, φ0, pω0, pφ0 = ω.copy(), φ.copy(), pω.copy(), pφ.copy()
    H_old = xp.sum(pω0**2 + pφ0**2)/2 + total_E()

    # leap-frog
    pω -= 0.5*dt*F_ω();  pφ -= 0.5*dt*F_φ()
    for _ in range(args.nlf):
        ω += dt*pω;       φ += dt*pφ
        pω -= dt*F_ω();   pφ -= dt*F_φ()
    pω -= 0.5*dt*F_ω();  pφ -= 0.5*dt*F_φ()
    pω *= -1;  pφ *= -1   # reversibility

    H_new = xp.sum(pω**2 + pφ**2)/2 + total_E()
    dH = float(H_new - H_old)

    if rng.random() < math.exp(-dH):
        acc += 1
        if (acc % args.save_every) == 0:
            np.savez_compressed(cfgdir/f"cfg_{acc:05d}.npz",
                                omega=np.asarray(ω),
                                phi=np.asarray(φ),
                                spin=np.asarray(s))
    else:                   # reject – restore fields
        ω[:], φ[:] = ω0, φ0

    if tr % 10 == 0:
        print(f"traj {tr:5d}/{args.traj}  ΔH={dH:+.3e}  acc={acc/tr:6.3f}")

t1 = time.time()
print("\n---- run summary ------------------------------")
print(f"Lattice      : {L}^3   nodes = {V}")
print(f"Traj / steps : {args.traj} / {args.nlf}")
print(f"Acceptance   : {acc/args.traj:.3f}")
print(f"Configs kept : {len(list(cfgdir.glob('cfg_*.npz')))}")
print(f"Wall-time    : {t1-t0:.1f} s  ({'GPU' if xp is cp else 'CPU'})")
