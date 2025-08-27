#!/usr/bin/env python
"""
Measure RTG v2 observables on saved configs.

Usage:
python measure_observables.py --cfgdir cfg_md --L 32 --links 18 --a_eff 0.0808 --noise 0.05
"""

import argparse, math, numpy as np
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument('--cfgdir', required=True)
ap.add_argument('--L', type=int, required=True)
ap.add_argument('--links', type=int, required=True)
ap.add_argument('--a_eff', type=float, required=True)
ap.add_argument('--noise', type=float, default=0.0)
ap.add_argument('--N', type=int, default=1000, help='Number of statistical samples for CHSH')
args = ap.parse_args()

cfgs = sorted(Path(args.cfgdir).glob("snap_*.npz"))
if not cfgs:
    raise RuntimeError(f"No configs found in {args.cfgdir}")

# Proton radius calculation
def proton_radius(links, a_eff):
    return links * a_eff / math.sqrt(3.0)

# CHSH calculation (standard Monte Carlo estimator)
def chsh(phi, omega, sigma, N=1000):
    a, ap = 0.0, math.pi/2
    b, bp = math.pi/4, 3*math.pi/4
    S = 0.0
    for _ in range(N):
        omega_noisy = omega + np.random.normal(0, sigma, size=omega.shape)
        A  = np.sign(np.cos(phi[0] - a))
        Ap = np.sign(np.cos(phi[0] - ap))
        B  = np.sign(np.cos(phi[1] - b))
        Bp = np.sign(np.cos(phi[1] - bp))
        S += (A*B + A*Bp + Ap*B - Ap*Bp)
    return abs(S / N)

# Data collection
radii, chshs = [], []

for fn in cfgs:
    dat = np.load(fn)
    phi, omega = dat['phi'], dat['omega']

    radii.append(proton_radius(args.links, args.a_eff))
    chshs.append(chsh(phi, omega, args.noise, args.N))

radii, chshs = np.asarray(radii), np.asarray(chshs)

# --- quick Γ–method estimate of τ_int (Wolff 2004) -----------------------------
def tau_int(series, max_lag=None):
    """Return τ_int in units of configs using Γ-method truncation."""
    x = np.asarray(series) - np.mean(series)
    N = len(x)
    if max_lag is None:
        max_lag = min(N // 2, 1000)
    acf = [1.0]
    for t in range(1, max_lag):
        acf.append(np.dot(x[:-t], x[t:]) / np.dot(x, x))
        if acf[-1] + acf[-2] < 0:  # automatic window criterion
            break
    tau = 0.5 + np.sum(acf[1:])
    return max(tau, 0.5)

tau_cfg = tau_int(chshs)
N_eff = len(chshs) / (2 * tau_cfg)
err_chsh = chshs.std(ddof=1) / np.sqrt(N_eff)
err_radius = radii.std(ddof=1) / np.sqrt(len(radii))

# ---------- print summary (corrected errors) ------------------------------------
print(f"Loaded {len(cfgs)} configs, lattice {args.L}³")
print(f"Proton circ-radius ⟨r⟩ = {radii.mean():6.3f} ± {err_radius:.3e} fm "
      f"(scatter {radii.std(ddof=1):.3e})")
print(f"CHSH (σ={args.noise}) = {chshs.mean():6.3f} ± {err_chsh:.3e}   "
      f"[τ_int ≈ {tau_cfg:.1f} cfg,  N_eff ≈ {N_eff:.0f}]")
