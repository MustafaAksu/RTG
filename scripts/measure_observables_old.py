#!/usr/bin/env python
"""
Measure RTG v2 observables on a stack of saved configs.

* Proton circ-radius  from the fixed three-node triangle
  (links × a_eff / √3).
* CHSH correlation    using pinned Bell pair (sites 0 and 1)
  with optional Gaussian ω-noise σ (in Δω* units).

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

def resonance_pair(omega_i, omega_j, phi_i, phi_j, s_i, s_j, theta_a=0, theta_b=0):
    effective_phi_i = phi_i + theta_a
    effective_phi_j = phi_j + theta_b
    cos = math.cos(effective_phi_i - effective_phi_j)
    s_i_c = 1j * s_i
    s_j_c = 1j * s_j
    spin_prod = s_i_c * s_j_c
    spin = 1 + np.real(spin_prod)
    gauss = math.exp(-((omega_i - omega_j)**2))
    return 0.75 * (1 + cos) * spin * gauss

def normalised_corr(R):
    """Map R∈[0,3] → E∈[-1,+1]."""
    return 2*R/3.0 - 1.0

def chsh(omega, phi, spin, sigma):
    i, j = 0, 1
    a, ap = 0.0, math.pi/2
    b, bp = math.pi/4, 7*math.pi/4  # bp equivalent to -π/4
    N_draw = 50 if sigma > 0 else 1
    E_ab, E_abp, E_apb, E_apbp = 0.0, 0.0, 0.0, 0.0
    for _ in range(N_draw):
        omega_j_noise = omega[j] + np.random.normal(0.0, sigma)
        def E(angle_a, angle_b):
            R = resonance_pair(omega[i], omega_j_noise, phi[i], phi[j], spin[i], spin[j], theta_a=angle_a, theta_b=angle_b)
            return normalised_corr(R)
        E_ab += E(a, b)
        E_abp += E(a, bp)
        E_apb += E(ap, b)
        E_apbp += E(ap, bp)
    E_ab /= N_draw; E_abp /= N_draw; E_apb /= N_draw; E_apbp /= N_draw
    sum_chsh = E_ab + E_abp + E_apb - E_apbp
    return abs(sum_chsh)

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
print(f"CHSH (σ={args.noise}) = {chshs.mean():6.3f} ± {chshs.std(ddof=1):6.3f}")