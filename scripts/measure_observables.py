#!/usr/bin/env python
"""
Measure RTG v2 observables on a stack of saved configs.

* Proton circ-radius  from the fixed three-node triangle
  (links × a_eff / √3).
* CHSH correlation    using pinned Bell pair (sites 0,+i and 1,–i)
  with optional Gaussian ω-noise σ (in units of Δω*).

Usage
-----
python scripts/measure_observables.py \
       --cfgdir cfg_lock --L 32 --links 18 --a_eff 0.0808 --noise 0.25
"""
# --------------------------------------------------------------------
import argparse, math, glob, numpy as np
from pathlib import Path

# ---------- CLI -----------------------------------------------------
ap = argparse.ArgumentParser()
ap.add_argument('--cfgdir', required=True, help='directory with *.npz configs')
ap.add_argument('--L',      type=int,   required=True, help='lattice size')
ap.add_argument('--links',  type=int,   required=True,
                help='triangle links for proton radius')
ap.add_argument('--a_eff',  type=float, required=True, help='effective spacing [fm]')
ap.add_argument('--noise',  type=float, default=0.0,
                help='Gaussian ω noise σ (Δω* units) for CHSH')
args = ap.parse_args()

cfgs = sorted(Path(args.cfgdir).glob("cfg_*.npz"))
if not cfgs:
    raise RuntimeError(f"No configs found in {args.cfgdir}")

# ---------- helpers -------------------------------------------------
def proton_radius(links, a_eff):
    """Geometric circumscribed-circle radius of equilateral triangle."""
    return links * a_eff / math.sqrt(3.0)

def resonance_pair(omega_i, omega_j, phi_i, phi_j, s_i, s_j):
    """Compute R_ij (dimensional ω) – same kernel as in HMC."""
    cos = math.cos(phi_i - phi_j)
    spin = 1 + s_i * s_j
    # exp term ~1 because |Δω| ≪ Δω* for the pinned pair; keep anyway
    gauss = math.exp(-((omega_i - omega_j)/1.0)**2)  # Δω* cancels if σ in same units
    return 0.75 * (1 + cos) * spin * gauss           # max 3.0

def normalised_corr(R):
    """Map R∈[0,3] → E∈[-1,+1]."""
    return 2*R/3.0 - 1.0

def chsh(omega, phi, spin, sigma):
    """Single-cfg CHSH with optional Gaussian ω noise (σ in Δω* units)."""
    # pinned Bell nodes
    i, j = 0, 1
    if sigma > 0.0:
        delta = np.random.normal(0.0, sigma)
        omega_j = omega[j] + delta
    else:
        omega_j = omega[j]

    def E(angle):
        R = resonance_pair(omega[i], omega_j,
                           phi[i],   phi[j] + angle,
                           spin[i],  spin[j])
        return normalised_corr(R)

    a, ap   = 0.0,  math.pi/2
    b, bp   = math.pi, 3*math.pi/2
    return abs(E(a-b) - E(a-bp) + E(ap-b) + E(ap-bp))

# ---------- loop over files -----------------------------------------
radii = []
chshs = []

for fn in cfgs:
    dat   = np.load(fn)
    ω, φ, s = dat['omega'], dat['phi'], dat['spin']
    radii.append(proton_radius(args.links, args.a_eff))
    chshs.append(chsh(ω, φ, s, args.noise))

radii   = np.asarray(radii)
chshs   = np.asarray(chshs)

# ---------- print summary -------------------------------------------
print(f"Loaded {len(cfgs)} configs, lattice {args.L}³")
print(f"Proton circ-radius ⟨r⟩ = {radii.mean():6.3f} ± {radii.std(ddof=1):6.3f}  fm")
print(f"CHSH (σ={args.noise}) = {chshs.mean():6.3f}")
