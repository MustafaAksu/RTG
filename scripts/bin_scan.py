#!/usr/bin/env python
# bin_scan.py  — estimate CHSH error by blocking                   2025-06-30
import argparse, glob, math, numpy as np, pathlib as pl, sys

# ───────────────────────────────────────────────────────── helpers ──────
def resonance_pair(ωi, ωj, φi, φj, si, sj, θa=0.0, θb=0.0):
    """Exact same formula as in measure_observables.py (with i·spin trick)."""
    cos   = math.cos((φi+θa) - (φj+θb))
    spin  = 1 - si*sj                   # because i·si · i·sj = −si sj
    gauss = math.exp(-(ωi-ωj)**2)
    return 0.75 * (1 + cos) * spin * gauss           # ∈[0,3]

def normalised_corr(R):        # maps [0,3] → [−1,+1]
    return 2*R/3.0 - 1.0

def chsh(ω, φ, s, σ):
    i, j   = 0, 1                                 # Bell pair sites
    a, ap  = 0.0, math.pi/2
    b, bp  = math.pi/4, 7*math.pi/4               # = −π/4
    draws  = 50 if σ > 0 else 1
    E = np.zeros(4)
    for _ in range(draws):
        ωj_noise = ω[j] + np.random.normal(0.0, σ)
        pairs = (
            (a,  b),  (a,  bp),
            (ap, b),  (ap, bp)
        )
        for k, (θa, θb) in enumerate(pairs):
            R  = resonance_pair(ω[i], ωj_noise, φ[i], φ[j], s[i], s[j],
                                θa, θb)
            E[k] += normalised_corr(R)
    E /= draws
    return abs(E[0] + E[1] + E[2] - E[3])
# ────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('cfgdir', help="folder with cfg_*.npz")
    ap.add_argument('--sigma', type=float, default=0.01,
                    help="Gaussian ω-noise σ (Δω*)")
    args = ap.parse_args()

    cfgs = sorted(pl.Path(args.cfgdir).glob("cfg_*.npz"))
    if not cfgs:
        sys.exit(f"[bin_scan] no cfg_*.npz in {args.cfgdir}")

    vals = []
    for fn in cfgs:
        d = np.load(fn)
        vals.append(chsh(d['omega'], d['phi'], d['spin'], args.sigma))
    vals = np.asarray(vals)

    print(f"# {len(vals)} configurations   σ={args.sigma}")
    for B in [1,2,4,8,16,32,64,128]:
        n = len(vals)//B
        if n < 2: break
        bins = vals[:n*B].reshape(n, B).mean(axis=1)
        err  = bins.std(ddof=1) / math.sqrt(n)
        print(f"bin {B:3d}   err = {err:9.4e}")

if __name__ == "__main__":
    main()
