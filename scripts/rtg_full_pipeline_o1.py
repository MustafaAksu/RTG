
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
rtg_full_pipeline.py

Utilities to post-process RTG kernel/scan results and make summary tables & plots.
Designed to work with either NumPy (CPU) or, if available, dpnp (Intel GPU)
for some array-heavy steps.

Subcommands
-----------
1) scan-top
    Read a scan JSON (from `rtg_kernel_rg_v3.py scan`) and write a top-k CSV
    sorted by increasing |spectral_dim| then by mds_stress.

2) n-scaling
    Collect `report_*_tfine.json` files (from `analyze`) that include a
    `kernel_file` with "K_<n>_..." in the name. Build a table with
    [n, spectral_dimension_mean, spectral_dimension_se, mds_stress], save CSV,
    make a simple matplotlib line plot, and fit sdim vs log2(n) both "global"
    (all points) and "asymptotic" (skip the smallest n).

3) kernel-quantiles
    For large K npy files, compute approximate upper-triangle quantiles by
    random sampling (non-replacing) of index pairs (i<j). Uses np.memmap to
    avoid loading the full matrix at once. If dpnp is available, quantiles
    are computed on GPU for the sampled values; otherwise CPU is used.

4) excel
    Bundle one or more CSVs (from the above steps) into a single Excel file
    with separate sheets (requires pandas + openpyxl).

Examples
--------
python rtg_full_pipeline.py scan-top --scan scan_2048_ultrafine.json \
    --out scan_2048_ultrafine_top20.csv --topk 20

python rtg_full_pipeline.py n-scaling --glob 'K_*_same_rho1p20_a2p50_b1p60_tfine.json' \
    --csv n_scaling_summary.csv --plot n_scaling_sdim.png

python rtg_full_pipeline.py kernel-quantiles --kernel K_4096_same_rho1p20_a2p50_b1p60.npy \
    --samples 300000 --out kernel_stats.csv

python rtg_full_pipeline.py excel --xlsx rtg_summary.xlsx \
    --csv scan_2048_ultrafine_top20.csv n_scaling_summary.csv kernel_stats.csv
"""

from __future__ import annotations
import argparse, json, math, os, re, sys, random
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

# ---------- Optional GPU acceleration (dpnp) ----------
USE_DPNP = False
xp = None  # numerical backend for heavy ops on sampled arrays
#try:
#    import dpnp as xp
#    import numpy as _np_cpu
#    USE_DPNP = True
#except Exception:
import numpy as xp
_np_cpu = xp

# Memmap comes from NumPy; dpnp doesn't implement memmap
import numpy as np

# Plotting (matplotlib only; no seaborn and no custom colors)
import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt

# Optional pandas for Excel
try:
    import pandas as pd
    HAS_PANDAS = True
except Exception:
    HAS_PANDAS = False


# -------------------- helpers --------------------
def read_json(path: Path) -> Dict:
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def write_csv(rows: List[Dict], out_path: Path) -> None:
    if not rows:
        with open(out_path, 'w', encoding='utf-8') as f:
            f.write('')  # empty
        return
    # simple CSV writer without pandas
    cols = list(rows[0].keys())
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(','.join(cols) + '\n')
        for r in rows:
            vals = [r.get(c, '') for c in cols]
            line = ','.join(str(v) for v in vals)
            f.write(line + '\n')


def parse_n_from_kernel_file(kernel_file: str) -> int | None:
    # Expect something like K_4096_same_...
    m = re.search(r'K_(\d+)_', Path(kernel_file).name)
    return int(m.group(1)) if m else None


# -------------------- scan-top --------------------
def cmd_scan_top(args: argparse.Namespace) -> None:
    scan = read_json(Path(args.scan))
    rows = scan.get('scan', [])
    # Normalize keys for output
    normalized = []
    for r in rows:
        normalized.append({
            'spin_mode': r.get('spin_mode'),
            'rho_scale': r.get('rho_scale'),
            'a2': r.get('a2'),
            'beta': r.get('beta'),
            'spectral_dim': r.get('spectral_dim'),
            'mds_stress': r.get('mds_stress'),
        })
    # sort by |spectral_dim| then mds_stress
    normalized.sort(key=lambda x: (abs(x['spectral_dim']), x['mds_stress']))
    top = normalized[:args.topk]
    out = Path(args.out)
    write_csv(top, out)
    print(f"[scan-top] wrote {len(top)} rows -> {out}")


# -------------------- n-scaling --------------------
def fit_line(x: _np_cpu.ndarray, y: _np_cpu.ndarray) -> Tuple[float, float]:
    """Return (intercept, slope) for y = a + b x using least squares on CPU."""
    A = _np_cpu.vstack([_np_cpu.ones_like(x), x]).T
    (a, b), *_ = _np_cpu.linalg.lstsq(A, y, rcond=None)
    return float(a), float(b)


def cmd_n_scaling(args: argparse.Namespace) -> None:
    paths = sorted(Path('.').glob(args.glob))
    if not paths:
        print(f"[n-scaling] no files matched: {args.glob}", file=sys.stderr)
        sys.exit(1)

    rows = []
    ns, sdims, ses, stresses = [], [], [], []
    for p in paths:
        d = read_json(p)
        n = parse_n_from_kernel_file(d.get('kernel_file', '')) or None
        if n is None:
            continue
        sdim = float(d.get('spectral_dimension_mean', 'nan'))
        se   = float(d.get('spectral_dimension_se', 'nan'))
        mds  = float(d.get('mds_stress', 'nan'))
        rows.append({'n': n, 'spectral_dimension_mean': sdim,
                     'spectral_dimension_se': se, 'mds_stress': mds,
                     'file': p.name})
        ns.append(n); sdims.append(sdim); ses.append(se); stresses.append(mds)

    rows.sort(key=lambda r: r['n'])
    if args.csv:
        write_csv(rows, Path(args.csv))
        print(f"[n-scaling] wrote summary CSV -> {args.csv}")

    # Fit sdim ~ a + b * log2(n)
    x_all = _np_cpu.log2(_np_cpu.array(ns, dtype=float))
    y_all = _np_cpu.array(sdims, dtype=float)

    a_all, b_all = fit_line(x_all, y_all)

    # Asymptotic fit: skip the smallest n (if we have >=3 points)
    if len(ns) >= 3:
        x_asym = x_all[1:]
        y_asym = y_all[1:]
        a_asym, b_asym = fit_line(x_asym, y_asym)
    else:
        a_asym, b_asym = float('nan'), float('nan')

    # Save a brief text report with the fits
    if args.fit_txt:
        with open(args.fit_txt, 'w', encoding='utf-8') as f:
            f.write(f"Global fit:   sdim ≈ a + b * log2(n) with a={a_all:.6f}, b={b_all:.6f}\n")
            f.write(f"Asymptotic:   sdim ≈ a + b * log2(n) with a={a_asym:.6f}, b={b_asym:.6f}\n")
            if args.predict_n:
                log2n = math.log2(args.predict_n)
                y_g = a_all + b_all * log2n
                y_a = a_asym + b_asym * log2n
                f.write(f"Prediction at n={args.predict_n}: global={y_g:.6f}, asympt={y_a:.6f}\n")
        print(f"[n-scaling] wrote fit report -> {args.fit_txt}")

    # Plot n vs sdim
    if args.plot:
        fig = plt.figure(figsize=(8,6))
        ax = fig.add_subplot(111)
        ax.plot(ns, sdims, marker='o')
        ax.set_title('n-scaling sdim')
        ax.set_xlabel('n')
        ax.set_ylabel('spectral_dimension_mean')
        fig.tight_layout()
        fig.savefig(args.plot, dpi=120)
        plt.close(fig)
        print(f"[n-scaling] wrote plot -> {args.plot}")


# -------------------- kernel-quantiles --------------------
def sample_upper_triangle(n: int, n_samples: int, rng: random.Random) -> _np_cpu.ndarray:
    """
    Draw random (i,j) pairs with i<j uniformly at random (approximate).
    Returns an array of shape (n_samples, 2) with int32 indices.
    """
    out = _np_cpu.empty((n_samples, 2), dtype=_np_cpu.int32)
    for k in range(n_samples):
        i = rng.randrange(n - 1)
        j = rng.randrange(i + 1, n)
        out[k, 0] = i
        out[k, 1] = j
    return out


def compute_kernel_quantiles(npy_path: Path, n_samples: int, q_list: List[float],
                             seed: int = 0, exclude_diag: bool = True) -> Dict[str, float]:
    """
    Approximate quantiles of the upper triangle (excluding diagonal) of K using random sampling.
    Uses numpy.memmap to avoid loading the entire matrix; quantiles on sampled values are
    computed with dpnp if available, otherwise numpy.

    Returns a dict of summary stats.
    """
    mm = np.load(str(npy_path), mmap_mode='r')  # shape (n,n)
    if mm.ndim != 2 or mm.shape[0] != mm.shape[1]:
        raise ValueError(f"Expected square matrix in {npy_path.name}, got {mm.shape}")

    n = int(mm.shape[0])
    rng = random.Random(seed)
    n_samples = int(min(n_samples, n * (n - 1) // 2))  # cannot exceed #unique i<j pairs

    idx = sample_upper_triangle(n, n_samples, rng)
    rows = idx[:, 0]
    cols = idx[:, 1]
    # Vectorized gather (CPU) from memmap; this *does* allocate the sampled array
    vals_cpu = mm[rows, cols].astype(_np_cpu.float64, copy=False)

    # Compute stats on GPU if dpnp is available
    if USE_DPNP:
        vals = xp.asarray(vals_cpu)  # transfer to GPU
        q = xp.quantile(vals, xp.asarray(q_list))
        q = _np_cpu.asarray(q)  # back to CPU
        mean_v = float(xp.mean(vals).get() if hasattr(xp, 'mean') else _np_cpu.mean(vals_cpu))
        max_v  = float(xp.max(vals).get() if hasattr(xp, 'max') else _np_cpu.max(vals_cpu))
        min_v  = float(xp.min(vals).get() if hasattr(xp, 'min') else _np_cpu.min(vals_cpu))
    else:
        q = _np_cpu.quantile(vals_cpu, _np_cpu.asarray(q_list))
        mean_v = float(_np_cpu.mean(vals_cpu))
        max_v  = float(_np_cpu.max(vals_cpu))
        min_v  = float(_np_cpu.min(vals_cpu))

    out = {
        'n': n,
        'file': npy_path.name,
        'mean': mean_v,
        'min': min_v,
        'max': max_v,
    }
    for frac, val in zip(q_list, q):
        key = f"q{str(frac).replace('.','_')}"
        out[key] = float(val)
    return out


def cmd_kernel_quantiles(args: argparse.Namespace) -> None:
    paths = [Path(p) for p in args.kernel]
    q_list = [float(q) for q in args.q.split(',')]
    rows = []
    for p in paths:
        stats = compute_kernel_quantiles(p, n_samples=args.samples,
                                         q_list=q_list, seed=args.seed)
        rows.append(stats)
        print(f"[kernel-quantiles] {p.name}: n={stats['n']} mean={stats['mean']:.6g} "
              f"q0.99={stats.get('q0_99', float('nan')):.6g} q0.999={stats.get('q0_999', float('nan')):.6g}")
    if args.out:
        write_csv(rows, Path(args.out))
        print(f"[kernel-quantiles] wrote CSV -> {args.out}")


# -------------------- excel bundling --------------------
def cmd_excel(args: argparse.Namespace) -> None:
    if not HAS_PANDAS:
        print("[excel] pandas not available; cannot write xlsx. Install pandas+openpyxl.", file=sys.stderr)
        sys.exit(2)

    xlsx_path = Path(args.xlsx)
    with pd.ExcelWriter(xlsx_path, engine='openpyxl') as writer:
        for csv_path in args.csv:
            p = Path(csv_path)
            if not p.exists():
                print(f"[excel] missing {p}, skipping")
                continue
            df = pd.read_csv(p)
            sheet = p.stem[:30]
            df.to_excel(writer, index=False, sheet_name=sheet)
    print(f"[excel] wrote -> {xlsx_path}")


# -------------------- main --------------------
def main(argv=None):
    parser = argparse.ArgumentParser(description="RTG post-processing utilities (NumPy/dpnp).")
    sub = parser.add_subparsers(dest='cmd', required=True)

    # scan-top
    p0 = sub.add_parser('scan-top', help='Top-k rows from a scan JSON')
    p0.add_argument('--scan', required=True, help='scan_*.json')
    p0.add_argument('--out', required=True, help='output CSV path')
    p0.add_argument('--topk', type=int, default=20, help='how many rows to keep (default 20)')
    p0.set_defaults(func=cmd_scan_top)

    # n-scaling
    p1 = sub.add_parser('n-scaling', help='Aggregate sdim vs n from report_*_tfine.json files')
    p1.add_argument('--glob', required=True, help="glob for tfine JSON reports, e.g. 'K_*_tfine.json'")
    p1.add_argument('--csv', default='n_scaling_summary.csv', help='summary CSV path')
    p1.add_argument('--plot', default='n_scaling_sdim.png', help='output PNG for the sdim vs n plot')
    p1.add_argument('--fit-txt', default='n_scaling_fit.txt', help='text file with fit coefficients & prediction')
    p1.add_argument('--predict-n', type=int, default=16384, help='n to predict from the fits (default 16384)')
    p1.set_defaults(func=cmd_n_scaling)

    # kernel-quantiles
    p2 = sub.add_parser('kernel-quantiles', help='Approximate quantiles of upper-triangle K_ij')
    p2.add_argument('--kernel', nargs='+', required=True, help='one or more K_*.npy files')
    p2.add_argument('--samples', type=int, default=300000, help='number of (i<j) pairs to sample (default 300k)')
    p2.add_argument('--q', default='0.5,0.9,0.99,0.999,0.9999', help='comma-separated quantiles')
    p2.add_argument('--seed', type=int, default=0, help='PRNG seed (default 0)')
    p2.add_argument('--out', default='kernel_stats.csv', help='output CSV path')
    p2.set_defaults(func=cmd_kernel_quantiles)

    # excel
    p3 = sub.add_parser('excel', help='Bundle CSVs into a single Excel workbook')
    p3.add_argument('--xlsx', default='rtg_summary.xlsx', help='output Excel path (default rtg_summary.xlsx)')
    p3.add_argument('--csv', nargs='+', required=True, help='CSV files to bundle as sheets')
    p3.set_defaults(func=cmd_excel)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == '__main__':
    raise SystemExit(main())
