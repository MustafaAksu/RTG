#!/usr/bin/env python3
"""
measure_observables.py – RTG‑v2 binary‑spin observables
────────────────────────────────────────────────────────────
* Works on both static cfg_*.npz and MD snap_*.npz
* Auto‑detects Δω* when stored in the file (fallback: CLI default)
* Vectorised CHSH for speed; adaptive number of ω‑noise draws
* Optionally prints flip statistics saved by the latest MD
"""

import argparse, math, numpy as np
from pathlib import Path
from textwrap import dedent

# ── CLI ──────────────────────────────────────────────────────────────
ap = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter,
    description=dedent("""\
        Example
        -------
        python measure_observables.py \\
               --cfgdir cfg_md --L 32 --links 18 --a_eff 0.0808 --noise 0.05
        Add --csv for machine-readable output, or --csv_header to print a CSV header.
    """))
ap.add_argument('--cfgdir', required=True)
ap.add_argument('--L',      type=int,   required=True)
ap.add_argument('--links',  type=int,   required=True,
                help='triangle links for proton circ-radius')
ap.add_argument('--a_eff',  type=float, required=True,
                help='effective spacing [fm]')
ap.add_argument('--noise',  type=float, default=0.0,
                help='Gaussian ω noise σ in units of Δω*')
ap.add_argument('--delta_omega_star', type=float, default=1.0,
                help='fallback Δω* if not stored in file')
ap.add_argument('--seed',   type=int, help='random seed for CHSH noise sampling')
ap.add_argument('--csv',    action='store_true',
                help='print one CSV line (noise, CHSH, err, flips)')
ap.add_argument('--csv_header', action='store_true',
                help='print CSV header line before output')
ap.add_argument('--filter', choices=['all', 'snap', 'cfg'], default='all',
                help='which files to include in sweep')
args = ap.parse_args()

if args.seed is not None:
    np.random.seed(args.seed)

cfgdir = Path(args.cfgdir)
files = []
if args.filter in ('all', 'cfg'):
    files += sorted(cfgdir.glob("cfg_*.npz"))
if args.filter in ('all', 'snap'):
    files += sorted(cfgdir.glob("snap_*.npz"))
if not files:
    raise RuntimeError(f"No *.npz configs found in {cfgdir}")

# ── helpers: geometry & kernel ───────────────────────────────────────
def proton_radius(n_links: int, a_eff: float) -> float:
    return n_links * a_eff / math.sqrt(3.0)

# Standard CHSH angle settings
a, a_p = 0.0, math.pi/2
b, b_p = math.pi/4, -math.pi/4
angles = np.array([[a,  b ], [a,  b_p], [a_p,b ], [a_p,b_p]])

def R_kernel(ωi, ωj, φi, φj, si, sj, dω_star):
    """
    Vectorised binary‑spin kernel for one Bell pair
    Parameters:
        ωi, ωj: arrays of emitter and receiver ω
        φi, φj: arrays of emitter and receiver φ
        si, sj: ±1 (scalar or broadcasted)
        dω_star: Δω* bandwidth
    Returns:
        RTG resonance kernel for CHSH correlation
    """
    cos   = np.cos(φi - φj)
    spin  = 1 + si * sj            # ±1 → 0 or 2
    gauss = np.exp(-((ωi - ωj) / dω_star)**2)
    return 0.75 * (1 + cos) * spin * gauss

# ── sweep over configs ───────────────────────────────────────────────
radii, chsh_vals, flip_rates = [], [], []

for fn in files:
    dat = np.load(fn)
    ω, φ, s = dat['omega'], dat['phi'], dat['spin']

    if len(s) < 2 or len(φ) < 2 or len(ω) < 2:
        raise ValueError(f"{fn.name} contains fewer than 2 entries in omega/phi/spin")

    # 1. Proton radius
    radii.append(proton_radius(args.links, args.a_eff))

    # 2. Δω*
    dω_star = dat.get('delta_omega_star', args.delta_omega_star)

    # 3. CHSH
    σ = args.noise
    draws = 1 if σ == 0 else int(min(200, math.ceil(3/σ**2)))

    ω_i = np.full(4, ω[0])
    φ_i = φ[0] + angles[:, 0]
    si  = s[0]
    φ_j0 = φ[1] + angles[:, 1]
    sj   = s[1]

    acc = 0.0
    for _ in range(draws):
        ω_j = ω[1] + np.random.normal(0.0, σ * dω_star, size=4)
        R   = R_kernel(ω_i, ω_j, φ_i, φ_j0, si, sj, dω_star)
        E   = 2 * R / 3.0 - 1.0
        S   = E[0] + E[1] + E[2] - E[3]
        acc += abs(S)
    chsh_vals.append(acc / draws)

    # 4. Flip rate
    if 'flipped' in dat:
        flip_rates.append(dat['flipped'].mean())
    else:
        flip_rates.append(np.nan)

# ── output ───────────────────────────────────────────────────────────
radii = np.asarray(radii)
chsh_vals = np.asarray(chsh_vals)
flip_rates = np.asarray(flip_rates)

meanS, errS = chsh_vals.mean(), chsh_vals.std(ddof=1)
meanF       = np.nanmean(flip_rates)

if args.csv:
    if args.csv_header:
        print("noise,CHSH,err,flips")
    print(f"{args.noise:0.3f},{meanS:0.5f},{errS:0.5f},{meanF:0.5f}")
else:
    tag = files[0].stem.split('_')[0]
    print(f"Loaded {len(files)} configs  ({tag} format)")
    print(f"Proton circ‑radius ⟨r⟩ = {radii.mean():7.3f} ± {radii.std(ddof=1):6.3f}  fm")
    print(f"CHSH (σ={args.noise})      = {meanS:7.3f} ± {errS:6.3f}")
    if not np.isnan(flip_rates).all():
        print(f"Avg. flip rate per site   = {meanF:7.4f}")
