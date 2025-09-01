#!/usr/bin/env python3
"""
Estimate a beta 'fixed point' from RTG analyze JSONs.

- Reads files like: K_<n>_same_rho1p16_a2p55_b<xx>p<yy>_<fit>.json
  where <fit> is typically 'smallt' or 'tfine'.
- Extracts (n, beta, sdim, se) from each JSON.
- Prints per-n curves and tries two estimates:
  (1) Pairwise crossings between curves (linear interpolation) if a sign change exists.
  (2) If no sign change, picks beta that minimizes variance across curves
      (piecewise-linear interpolation on a dense beta grid).

It also writes a small summary JSON and CSV.

Usage examples (PowerShell):
  python .\beta_fixed_point.py `
    --glob "K_*_same_rho1p16_a2p55_b*_smallt.json" `
    --out beta_fp_smallt

  python .\beta_fixed_point.py `
    --glob "K_*_same_rho1p16_a2p55_b*_tfine.json" `
    --out beta_fp_tfine
"""

import argparse, glob, json, math, os, re, sys
from collections import defaultdict

try:
    import numpy as np
except Exception:
    # dpnp is optional; numpy is fine for this workload
    import numpy as np

PAT_FILE = re.compile(
    r'^K_(\d+)_.*_b(\d+)p(\d+)_([A-Za-z0-9]+)\.json$'
)

def parse_record(path):
    """
    Return dict with keys: n(int), beta(float), fit(str), sdim(float), se(float).
    Robust to filename/JSON; prefers JSON 'kernel_file' for n,beta.
    """
    with open(path, 'r', encoding='utf-8') as fh:
        d = json.load(fh)

    # Prefer extracting from kernel_file (most reliable)
    kf = d.get("kernel_file", "")
    n = None
    beta = None
    m_k = re.search(r'^K_(\d+)_', os.path.basename(kf))
    if m_k:
        n = int(m_k.group(1))
    m_b = re.search(r'_b(\d+)p(\d+)\.npy$', kf)
    if m_b:
        beta = float(f"{m_b.group(1)}.{m_b.group(2)}")

    # Fall back to filename if needed
    if n is None or beta is None:
        m_f = PAT_FILE.match(os.path.basename(path))
        if m_f:
            n = int(m_f.group(1)) if n is None else n
            if beta is None:
                beta = float(f"{m_f.group(2)}.{m_f.group(3)}")

    # Fit tag from filename (e.g., smallt / tfine)
    fit = None
    m_fit = PAT_FILE.match(os.path.basename(path))
    if m_fit:
        fit = m_fit.group(4)

    sdim = float(d["spectral_dimension_mean"])
    se   = float(d["spectral_dimension_se"])
    return {"n": n, "beta": beta, "fit": fit, "sdim": sdim, "se": se, "file": path}

def load_curves(glob_pat):
    """
    Returns: dict n -> list of (beta, sdim, se, file) sorted by beta
    """
    curves = defaultdict(list)
    files = sorted(glob.glob(glob_pat))
    if not files:
        print(f"[ERROR] No files matched --glob {glob_pat}", file=sys.stderr)
        sys.exit(2)

    for f in files:
        rec = parse_record(f)
        if rec["n"] is None or rec["beta"] is None:
            print(f"[WARN] Could not parse n/beta from {f}; skipping", file=sys.stderr)
            continue
        curves[rec["n"]].append((rec["beta"], rec["sdim"], rec["se"], rec["file"]))

    # sort each by beta
    for n in list(curves.keys()):
        curves[n].sort(key=lambda t: t[0])
    return curves

def crossings(curve_a, curve_b):
    """
    Try to find linear-interpolation crossings of (sdim_a - sdim_b) vs beta.
    Returns list of beta* (floats). Only checks adjacent beta intervals.
    curve_* must be list of (beta, sdim, se, file) sorted by beta.
    Assumes both curves share (approximately) the same beta grid.
    """
    xs = []
    # Map beta -> sdim for quick lookup/interp
    def interp(curve, beta):
        # piecewise linear interpolation
        betas = [t[0] for t in curve]
        sdims = [t[1] for t in curve]
        if beta <= betas[0]:
            return sdims[0]
        if beta >= betas[-1]:
            return sdims[-1]
        for i in range(1, len(betas)):
            b0, b1 = betas[i-1], betas[i]
            if b0 <= beta <= b1:
                t = (beta - b0) / (b1 - b0 + 1e-12)
                return (1 - t) * sdims[i-1] + t * sdims[i]
        return sdims[-1]

    # Use the overlapping set of betas from the coarser curve
    base = curve_a if len(curve_a) <= len(curve_b) else curve_b
    other = curve_b if base is curve_a else curve_a

    for i in range(1, len(base)):
        b0 = base[i-1][0]
        b1 = base[i][0]
        diff0 = base[i-1][1] - interp(other, b0)
        diff1 = base[i][1]   - interp(other, b1)
        if diff0 == 0.0:
            xs.append(b0)
            continue
        if diff0 * diff1 < 0:  # sign change
            t = diff0 / (diff0 - diff1 + 1e-16)
            beta_star = b0 + t * (b1 - b0)
            xs.append(beta_star)
    return xs

def variance_min_beta(curves):
    """
    If no clean crossing exists, pick beta that minimizes variance across curves.
    We build an overlap [Bmin, Bmax] and evaluate a dense grid.
    """
    # Require at least two curves with >= 2 points to interpolate
    usable = [v for v in curves.values() if len(v) >= 2]
    if len(usable) < 2:
        return None, None

    # Overlap bracket across usable curves
    bmins, bmaxs = [], []
    for c in usable:
        bmins.append(c[0][0])
        bmaxs.append(c[-1][0])
    Bmin = max(bmins)
    Bmax = min(bmaxs)
    if not (Bmax > Bmin):
        return None, None

    # Build interpolators
    def interp(curve, beta):
        betas = [t[0] for t in curve]
        sdims = [t[1] for t in curve]
        if beta <= betas[0]:
            return sdims[0]
        if beta >= betas[-1]:
            return sdims[-1]
        for i in range(1, len(betas)):
            b0, b1 = betas[i-1], betas[i]
            if b0 <= beta <= b1:
                t = (beta - b0) / (b1 - b0 + 1e-12)
                return (1 - t) * sdims[i-1] + t * sdims[i]
        return sdims[-1]

    grid = np.linspace(Bmin, Bmax, 1001)
    best_beta, best_var = None, None
    for b in grid:
        vals = [interp(c, b) for c in usable]
        v = float(np.var(vals, ddof=0))
        if best_var is None or v < best_var:
            best_var, best_beta = v, b
    return best_beta, best_var

def slope_near_beta(curve, beta_star):
    """
    Central-difference derivative d(sdim)/d(beta) near beta_star for a single curve.
    """
    betas = np.array([t[0] for t in curve], dtype=float)
    sdims = np.array([t[1] for t in curve], dtype=float)
    # Find indices around beta_star
    idx = np.searchsorted(betas, beta_star)
    i0 = max(1, min(idx, len(betas) - 2))
    b0, b1 = betas[i0-1], betas[i0+1]
    s0, s1 = sdims[i0-1], sdims[i0+1]
    return (s1 - s0) / (b1 - b0 + 1e-12)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", required=True,
                    help="Glob for JSONs (e.g. 'K_*_same_rho1p16_a2p55_b*_smallt.json')")
    ap.add_argument("--out", default="beta_fp",
                    help="Output stem (writes <stem>.json and <stem>.csv)")
    args = ap.parse_args()

    curves = load_curves(args.glob)
    ns = sorted(curves.keys())
    if not ns:
        print("[ERROR] No usable curves were found.", file=sys.stderr)
        sys.exit(2)

    print("== Curves (sdim ± se) ==")
    for n in ns:
        print(f"n={n}")
        for beta, sdim, se, _ in curves[n]:
            print(f"  beta={beta:.2f}  sdim={sdim:.6f} ± {se:.6f}")
    print()

    # Pairwise crossings
    pairwise = []
    for i in range(len(ns) - 1):
        n1, n2 = ns[i], ns[i+1]
        xs = crossings(curves[n1], curves[n2])
        if xs:
            for x in xs:
                pairwise.append({"n1": n1, "n2": n2, "beta_star": x})
                print(f"[INFO] Crossing between n={n1} and n={n2} at beta ≈ {x:.6f}")
        else:
            print(f"[INFO] No sign-change crossing detected for n={n1} vs n={n2}")

    # Variance-min beta (uses all curves with >= 2 points)
    beta_varmin, varmin = variance_min_beta(curves)
    if beta_varmin is not None:
        print(f"\n[INFO] Variance-minimizing beta across curves: beta* ≈ {beta_varmin:.6f} (var={varmin:.3e})")
    else:
        print("\n[INFO] Could not compute variance-min beta (insufficient overlap).")

    # Slopes at beta*
    slopes = []
    if beta_varmin is not None:
        for n in ns:
            if len(curves[n]) >= 3:
                s = slope_near_beta(curves[n], beta_varmin)
                slopes.append({"n": n, "slope_dsdim_dbeta": s})
                print(f"  slope d(sdim)/d(beta) @ n={n}: {s:.6e}")

    # Save summary
    summary = {
        "glob": args.glob,
        "pairwise_crossings": pairwise,
        "beta_star_variance_min": beta_varmin,
        "variance_at_beta_star": varmin,
        "slopes_at_beta_star": slopes,
        "curves": {
            int(n): [{"beta": float(b), "sdim": float(s), "se": float(se)}
                     for (b, s, se, _) in curves[n]]
            for n in curves
        }
    }
    with open(args.out + ".json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    # Also CSV with all points
    import csv
    with open(args.out + ".csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["n", "beta", "sdim", "se"])
        for n in ns:
            for beta, sdim, se, _ in curves[n]:
                w.writerow([n, beta, sdim, se])

    print(f"\n[OK] Wrote {args.out}.json and {args.out}.csv")

if __name__ == "__main__":
    main()
