#!/usr/bin/env python
"""
measure_observables_md.py  –  RTG‑v2 binary‑spin observables
✓ works on both static  cfg_*.npz  and MD  snap_*.npz
✓ correct spin factor 1 + s_i s_j
✓ correct Gaussian exp(‑(Δω/Δω*)²)

usage example
-------------
python measure_observables_md.py \
       --cfgdir cfg_md --L 32 --links 18 --a_eff 0.0808 --noise 0.01
"""
import argparse, glob, math, numpy as np
from pathlib import Path

# ── CLI ──────────────────────────────────────────────────────────────
ap = argparse.ArgumentParser()
ap.add_argument('--cfgdir', required=True)
ap.add_argument('--L',      type=int,   required=True)
ap.add_argument('--links',  type=int,   required=True,
                help='triangle links for proton circ‑radius')
ap.add_argument('--a_eff',  type=float, required=True,
                help='effective spacing [fm]')
ap.add_argument('--noise',  type=float, default=0.0,
                help='Gaussian ω noise σ in units of Δω*')
ap.add_argument('--delta_omega_star', type=float, default=1.0,
                help='Δω* normalisation used in MD run')
args = ap.parse_args()

cfgdir = Path(args.cfgdir)
files  = sorted(cfgdir.glob("cfg_*.npz")) + sorted(cfgdir.glob("snap_*.npz"))
if not files:
    raise RuntimeError(f"No *.npz configs found in {cfgdir}")

# ── helpers ──────────────────────────────────────────────────────────
def proton_radius(links, a_eff):
    return links * a_eff / math.sqrt(3.0)

def resonance_pair(ω_i, ω_j, φ_i, φ_j, s_i, s_j, θ_a=0.0, θ_b=0.0):
    """Binary‑spin kernel R_ij (identical to MD force)."""
    cos   = math.cos((φ_i+θ_a) - (φ_j+θ_b))
    spin  = 1 + s_i * s_j                    # ±1
    gauss = math.exp(-((ω_i - ω_j)/args.delta_omega_star)**2)
    return 0.75 * (1 + cos) * spin * gauss   # 0 … 3

def E_corr(R):               # map R∈[0,3] → E∈[‑1,+1]
    return 2.0*R/3.0 - 1.0

def chsh_one_cfg(ω, φ, s, sigma):
    i, j   = 0, 1                    # Bell pair sites
    a, ap  = 0.0, math.pi/2
    b, bp  = math.pi/4, -math.pi/4
    E = np.zeros(4)

    # draw noise for ω_j if requested
    draws = 50 if sigma > 0 else 1
    for _ in range(draws):
        ω_j = ω[j] + np.random.normal(0.0, sigma*args.delta_omega_star)

        E[0] += E_corr(resonance_pair(ω[i], ω_j, φ[i], φ[j], s[i], s[j], a,  b))
        E[1] += E_corr(resonance_pair(ω[i], ω_j, φ[i], φ[j], s[i], s[j], a,  bp))
        E[2] += E_corr(resonance_pair(ω[i], ω_j, φ[i], φ[j], s[i], s[j], ap, b))
        E[3] += E_corr(resonance_pair(ω[i], ω_j, φ[i], φ[j], s[i], s[j], ap, bp))

    E /= draws
    S =  E[0] + E[1] + E[2] - E[3]
    return abs(S)

# ── loop over configs ────────────────────────────────────────────────
radii, chsh_vals = [], []
for fn in files:
    dat = np.load(fn)
    ω, φ, s = dat['omega'], dat['phi'], dat['spin']
    radii.append(proton_radius(args.links, args.a_eff))
    chsh_vals.append(chsh_one_cfg(ω, φ, s, args.noise))

radii, chsh_vals = np.asarray(radii), np.asarray(chsh_vals)

# ── results ──────────────────────────────────────────────────────────
print(f"Loaded {len(files)} configs  ({files[0].stem.split('_')[0]} format)")
print(f"Proton circ‑radius ⟨r⟩ = {radii.mean():7.3f} ± {radii.std(ddof=1):6.3f} fm")
print(f"CHSH (σ={args.noise})      = {chsh_vals.mean():7.3f} ± {chsh_vals.std(ddof=1):6.3f}")
