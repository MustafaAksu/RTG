#!/usr/bin/env python3
"""
rtg_pipeline.py — Orchestrate RTG kernel scans/analyses and build summary tables.

Highlights
- Reproducible wrappers around rtg_kernel_rg_v3.py for build/analyze/scan/rg.
- Robust JSON loaders + summary makers (scan top-K, seed sweeps, beta sweeps, n-scaling).
- Writes a single Excel file with multiple sheets (requires pandas + openpyxl or xlsxwriter).
- Optional dpnp acceleration for small array ops (export USE_DPNP=1).

Examples (PowerShell / bash alike)
----------------------------------
# Top-20 rows (by |spectral_dim| then mds_stress) from a grid scan:
python rtg_pipeline.py make-summary ^
  --scan scan_2048_ultrafine.json ^
  --topk 20 ^
  --excel rtg_summary.xlsx ^
  --csv scan_2048_ultrafine_top20.csv

# Seeds sweep:
python rtg_pipeline.py seeds-sweep ^
  --kernel K_2048_same_rho1p16_a2p55_b1p40.npy ^
  --seeds 41,42,43,44,45,73,97 ^
  --t-grid 0.25,0.2,0.15,0.1,0.07,0.05,0.035,0.025,0.02,0.015,0.01,0.007,0.005 ^
  --bootstrap 12 --mds-dim 4 --max-nodes-eig 1200 ^
  --outdir .

# Beta sweep at (rho_scale=1.16, a2=0.55):
python rtg_pipeline.py beta-sweep ^
  --attrs attrs_2048.npz --a2 0.55 --rho-scale 1.16 ^
  --betas 1.35,1.40,1.45,1.50,1.55,1.60,1.65 ^
  --t-grid 0.25,0.2,0.15,0.1,0.07,0.05,0.035,0.025,0.02,0.015,0.01,0.007,0.005 ^
  --bootstrap 16 --seed 71 --mds-dim 4 --max-nodes-eig 1200 ^
  --outdir .

# N-scaling at fixed (rho_scale=1.20, a2=0.50, beta=1.60):
python rtg_pipeline.py n-scaling ^
  --sizes 1024,2048,4096,8192 --seed 71 ^
  --t-grid 0.25,0.2,0.15,0.1,0.07,0.05,0.035,0.025,0.02,0.015,0.01,0.007,0.005

Notes
-----
- All wrappers assume rtg_kernel_rg_v3.py is on your PATH or in the CWD.
- To use dpnp for minor vector ops, set USE_DPNP=1 in the environment before running.
- Plots are created by rtg_kernel_rg_v3.py if --plots is passed to the wrappers.
"""

from __future__ import annotations
import os, re, sys, json, glob, math, subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional

# ---- Optional dpnp backend for simple array ops --------------------------------
USE_DPNP = os.environ.get("USE_DPNP", "0") == "1"
xp = None
if USE_DPNP:
    try:
        import dpnp as xp  # type: ignore
    except Exception as e:
        xp = None
        print(f"[warn] USE_DPNP=1 but dpnp import failed: {e} — falling back to numpy", file=sys.stderr)

if xp is None:
    import numpy as xp  # type: ignore

# ---- Utility ------------------------------------------------------------------

def parse_t_grid(s: str) -> List[float]:
    return [float(tok) for tok in s.split(",") if tok.strip()]

def pretty(obj: Any) -> str:
    return json.dumps(obj, indent=2, sort_keys=False)

def run_cmd(args: List[str], cwd: Optional[str] = None) -> Tuple[int, str, str]:
    p = subprocess.run(args, cwd=cwd, capture_output=True, text=True)
    return p.returncode, p.stdout, p.stderr

def py_exe() -> str:
    return sys.executable

def rtg_script_path() -> str:
    # Rely on PATH/CWD resolution
    return "rtg_kernel_rg_v3.py"

# ---- Wrappers around rtg_kernel_rg_v3.py --------------------------------------

def build_kernel(attrs: str, spin_mode: str, a2: float, beta: float, rho_scale: float,
                 out: str, plots: bool = True) -> None:
    args = [
        py_exe(), rtg_script_path(), "build-kernel",
        "--attrs", attrs,
        "--spin-mode", spin_mode,
        "--a2", str(a2),
        "--beta", str(beta),
        "--rho-scale", str(rho_scale),
        "--out", out
    ]
    if plots:
        args.append("--plots")
    code, out_s, err_s = run_cmd(args)
    sys.stdout.write(out_s)
    if code != 0:
        sys.stderr.write(err_s)
        raise RuntimeError(f"build-kernel failed ({code})")

def analyze_kernel(kernel: str, report: str, t_grid: str, bootstrap: int, seed: int,
                   mds_dim: int, max_nodes_eig: int, plots: bool = False) -> None:
    args = [
        py_exe(), rtg_script_path(), "analyze",
        "--kernel", kernel,
        "--report", report,
        "--t-grid", t_grid,
        "--bootstrap", str(bootstrap),
        "--seed", str(seed),
        "--mds-dim", str(mds_dim),
        "--max-nodes-eig", str(max_nodes_eig),
    ]
    if plots:
        args.append("--plots")
    code, out_s, err_s = run_cmd(args)
    sys.stdout.write(out_s)
    if code != 0:
        sys.stderr.write(err_s)
        raise RuntimeError(f"analyze failed ({code})")

def scan_grid(attrs: str, report: str, rho_scale_grid: str, a2_grid: str, beta_grid: str,
              spin_modes: str, t_grid: str, mds_dim: int, max_nodes_eig: int) -> None:
    args = [
        py_exe(), rtg_script_path(), "scan",
        "--attrs", attrs,
        "--report", report,
        "--rho-scale-grid", rho_scale_grid,
        "--a2-grid", a2_grid,
        "--beta-grid", beta_grid,
        "--spin-modes", spin_modes,
        "--t-grid", t_grid,
        "--mds-dim", str(mds_dim),
        "--max-nodes-eig", str(max_nodes_eig),
    ]
    code, out_s, err_s = run_cmd(args)
    sys.stdout.write(out_s)
    if code != 0:
        sys.stderr.write(err_s)
        raise RuntimeError(f"scan failed ({code})")

def rg_flow(attrs: str, steps: int, report: str, radius_scale: float, rho_scale_grid: str,
            beta_grid: str, t_grid: str, mds_dim: int, max_nodes_eig: int, seed: int) -> None:
    args = [
        py_exe(), rtg_script_path(), "rg",
        "--attrs", attrs,
        "--steps", str(steps),
        "--report", report,
        "--radius-scale", str(radius_scale),
        "--rho-scale-grid", rho_scale_grid,
        "--beta-grid", beta_grid,
        "--t-grid", t_grid,
        "--mds-dim", str(mds_dim),
        "--max-nodes-eig", str(max_nodes_eig),
        "--seed", str(seed),
    ]
    code, out_s, err_s = run_cmd(args)
    sys.stdout.write(out_s)
    if code != 0:
        sys.stderr.write(err_s)
        raise RuntimeError(f"rg failed ({code})")

# ---- Parsers & Aggregators ----------------------------------------------------

_KERNEL_RE = re.compile(
    r"^K_(?P<n>\d+)_(?P<spin>[A-Za-z\-]+)_rho(?P<rho_i>\d+)p(?P<rho_f>\d+)_a2p(?P<a2>\d+)_b(?P<beta>(?:\d+p\d+)|IGNORED)\.npy$"
)

def parse_kernel_meta(kernel_file: str) -> Dict[str, Any]:
    """Extract n, spin_mode, rho_scale, a2, beta from a kernel filename."""
    bn = os.path.basename(kernel_file)
    m = _KERNEL_RE.match(bn)
    if not m:
        return {}
    n = int(m.group("n"))
    spin_mode = m.group("spin")
    rho_scale = float(f"{m.group('rho_i')}.{m.group('rho_f')}")
    a2 = float(f"0.{m.group('a2')}")
    beta_str = m.group("beta")
    beta = None
    if beta_str != "IGNORED":
        # e.g., "1p60" -> 1.60
        parts = beta_str.split("p")
        beta = float(f"{parts[0]}.{parts[1]}")
    return dict(n=n, spin_mode=spin_mode, rho_scale=rho_scale, a2=a2, beta=beta)

def load_json(fp: str) -> Dict[str, Any]:
    with open(fp, "r", encoding="utf-8") as f:
        return json.load(f)

def df_from_scan(scan_json: str):
    import pandas as pd
    d = load_json(scan_json)
    rows = d.get("scan", [])
    if not rows:
        return pd.DataFrame()
    # sort by |spectral_dim|, then mds_stress (ascending)
    abs_vals = [abs(r["spectral_dim"]) for r in rows]
    # Use xp for demonstration (dpnp/numpy) but pandas does the heavy lifting

    order = sorted(range(len(rows)), key=lambda i: (abs_vals[i], rows[i]["mds_stress"]))
    sorted_rows = [rows[i] for i in order]
    return pd.DataFrame(sorted_rows)

def df_from_reports(glob_pattern: str) -> "pd.DataFrame":
    import pandas as pd
    data = []
    for fp in glob.glob(glob_pattern):
        try:
            r = load_json(fp)
        except Exception:
            continue
        kf = r.get("kernel_file", "")
        meta = parse_kernel_meta(kf) if kf else {}
        row = dict(
            file=os.path.basename(fp),
            kernel_file=kf,
            spectral_dimension_mean=r.get("spectral_dimension_mean"),
            spectral_dimension_se=r.get("spectral_dimension_se"),
            mds_stress=r.get("mds_stress"),
        )
        hm = r.get("heat_meta", {})
        row.update(dict(
            slope=hm.get("slope"), intercept=hm.get("intercept"),
            t_grid=",".join(str(x) for x in hm.get("t_grid", [])),
        ))
        row.update(meta)
        data.append(row)
    df = pd.DataFrame(data)
    # Try to infer n from kernel_file if missing
    if "n" not in df.columns or df["n"].isna().any():
        def _n_from_kf(kf: str):
            m = re.match(r"^K_(\d+)_", os.path.basename(kf or ""))
            return int(m.group(1)) if m else None
        df["n"] = df.get("kernel_file", "").apply(_n_from_kf) if "kernel_file" in df else None
    return df.sort_values(["n", "spin_mode", "rho_scale", "a2", "beta"], na_position="last")

def df_n_scaling(all_reports_df: "pd.DataFrame", target: Dict[str, Any]) -> "pd.DataFrame":
    import pandas as pd
    # target e.g.: dict(spin_mode="same", rho_scale=1.20, a2=0.50, beta=1.60)
    filt = (all_reports_df["spin_mode"] == target["spin_mode"]) & \
           (all_reports_df["rho_scale"] == target["rho_scale"]) & \
           (all_reports_df["a2"] == target["a2"]) & \
           (all_reports_df["beta"] == target["beta"])
    cols = ["n", "spectral_dimension_mean", "spectral_dimension_se", "mds_stress", "slope", "intercept", "file"]
    return all_reports_df.loc[filt, cols].sort_values("n")

def df_beta_sweep(all_reports_df: "pd.DataFrame", target: Dict[str, Any]) -> "pd.DataFrame":
    import pandas as pd
    # target e.g.: dict(n=2048, spin_mode="same", rho_scale=1.16, a2=0.55)
    filt = (all_reports_df["n"] == target["n"]) & \
           (all_reports_df["spin_mode"] == target["spin_mode"]) & \
           (all_reports_df["rho_scale"] == target["rho_scale"]) & \
           (all_reports_df["a2"] == target["a2"]) & \
           (~all_reports_df["beta"].isna())
    cols = ["beta", "spectral_dimension_mean", "spectral_dimension_se", "mds_stress", "slope", "intercept", "file"]
    return all_reports_df.loc[filt, cols].sort_values("beta")

# ---- Plot helpers -------------------------------------------------------------

def plot_series(x, y, xlabel, ylabel, title, out_png):
    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(x, y, marker="o")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)

# ---- CLI ----------------------------------------------------------------------

def _add_common_analyze_args(sp, include_seed: bool = True):
    sp.add_argument("--t-grid", required=True, help="comma-separated floats")
    sp.add_argument("--bootstrap", type=int, default=12)
    if include_seed:
        sp.add_argument("--seed", type=int, default=71)
    sp.add_argument("--mds-dim", type=int, default=4)
    sp.add_argument("--max-nodes-eig", type=int, default=1200)

def main():
    import argparse
    p = argparse.ArgumentParser(description="RTG orchestration & summary tools")
    sub = p.add_subparsers(dest="cmd", required=True)

    # build
    sp = sub.add_parser("run-build", help="build-kernel wrapper")
    sp.add_argument("--attrs", required=True)
    sp.add_argument("--spin-mode", required=True)
    sp.add_argument("--a2", type=float, required=True)
    sp.add_argument("--beta", type=float, required=True)
    sp.add_argument("--rho-scale", type=float, required=True)
    sp.add_argument("--out", required=True)
    sp.add_argument("--no-plots", action="store_true")

    # analyze
    sp = sub.add_parser("run-analyze", help="analyze wrapper")
    sp.add_argument("--kernel", required=True)
    sp.add_argument("--report", required=True)
    _add_common_analyze_args(sp)
    sp.add_argument("--no-plots", action="store_true")

    # scan
    sp = sub.add_parser("run-scan", help="grid scan wrapper")
    sp.add_argument("--attrs", required=True)
    sp.add_argument("--report", required=True)
    sp.add_argument("--rho-scale-grid", required=True)
    sp.add_argument("--a2-grid", required=True)
    sp.add_argument("--beta-grid", required=True)
    sp.add_argument("--spin-modes", required=True)
    sp.add_argument("--t-grid", required=True)
    sp.add_argument("--mds-dim", type=int, default=4)
    sp.add_argument("--max-nodes-eig", type=int, default=1200)

    # rg
    sp = sub.add_parser("run-rg", help="RG flow wrapper")
    sp.add_argument("--attrs", required=True)
    sp.add_argument("--steps", type=int, required=True)
    sp.add_argument("--report", required=True)
    sp.add_argument("--radius-scale", type=float, default=1.0)
    sp.add_argument("--rho-scale-grid", required=True)
    sp.add_argument("--beta-grid", required=True)
    sp.add_argument("--t-grid", required=True)
    sp.add_argument("--mds-dim", type=int, default=4)
    sp.add_argument("--max-nodes-eig", type=int, default=1200)
    sp.add_argument("--seed", type=int, default=13)

    # seeds sweep
    sp = sub.add_parser("seeds-sweep", help="repeat analyze for multiple seeds")
    sp.add_argument("--kernel", required=True)
    sp.add_argument("--seeds", required=True, help="comma-separated ints")
    sp.add_argument("--outdir", default=".")
    _add_common_analyze_args(sp)

    # beta sweep
    sp = sub.add_parser("beta-sweep", help="build+analyze across betas")
    sp.add_argument("--attrs", required=True)
    sp.add_argument("--a2", type=float, required=True)
    sp.add_argument("--rho-scale", type=float, required=True)
    sp.add_argument("--betas", required=True, help="comma-separated floats")
    sp.add_argument("--outdir", default=".")
    _add_common_analyze_args(sp)

    # n scaling (synthesize attrs + analyze)
    sp = sub.add_parser("n-scaling", help="synthesize attrs for sizes and analyze at fixed params")
    sp.add_argument("--sizes", required=True, help="comma-separated ints, e.g. 1024,2048,4096,8192")
    sp.add_argument("--a2", type=float, default=0.5)
    sp.add_argument("--rho-scale", type=float, default=1.20)
    sp.add_argument("--beta", type=float, default=1.60)
    sp.add_argument("--seed", type=int, default=71)
    _add_common_analyze_args(sp, include_seed=False)

    # make summary
    sp = sub.add_parser("make-summary", help="build a single Excel/CSV summary from JSON outputs")
    sp.add_argument("--scan", default="scan_2048_ultrafine.json")
    sp.add_argument("--topk", type=int, default=20)
    sp.add_argument("--excel", default="rtg_summary.xlsx")
    sp.add_argument("--csv", default="scan_2048_ultrafine_top20.csv")
    sp.add_argument("--reports-glob", default="report_*json")

    args = p.parse_args()

    if args.cmd == "run-build":
        build_kernel(args.attrs, args.spin_mode, args.a2, args.beta, args.rho_scale, args.out, plots=not args.no_plots)
        return

    if args.cmd == "run-analyze":
        analyze_kernel(args.kernel, args.report, args.t_grid, args.bootstrap, args.seed, args.mds_dim, args.max_nodes_eig, plots=not args.no_plots)
        return

    if args.cmd == "run-scan":
        scan_grid(args.attrs, args.report, args.rho_scale_grid, args.a2_grid, args.beta_grid, args.spin_modes, args.t_grid, args.mds_dim, args.max_nodes_eig)
        return

    if args.cmd == "run-rg":
        rg_flow(args.attrs, args.steps, args.report, args.radius_scale, args.rho_scale_grid, args.beta_grid, args.t_grid, args.mds_dim, args.max_nodes_eig, args.seed)
        return

    if args.cmd == "seeds-sweep":
        seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
        base = Path(args.outdir)
        base.mkdir(parents=True, exist_ok=True)
        for s in seeds:
            rep = base / f"report_seed{s}.json"
            analyze_kernel(args.kernel, str(rep), args.t_grid, args.bootstrap, s, args.mds_dim, args.max_nodes_eig, plots=False)
        # Summarize
        import pandas as pd
        rows = []
        for fp in glob.glob(str(base / "report_seed*.json")):
            r = load_json(fp)
            rows.append(dict(
                file=os.path.basename(fp),
                spectral_dimension_mean=r.get("spectral_dimension_mean"),
                spectral_dimension_se=r.get("spectral_dimension_se"),
                mds_stress=r.get("mds_stress"),
            ))
        if rows:
            df = pd.DataFrame(rows).sort_values("spectral_dimension_mean")
            df.to_csv(base / "seeds_sweep_summary.csv", index=False)
            print(f"[ok] seeds sweep summary -> {base / 'seeds_sweep_summary.csv'}")
        return

    if args.cmd == "beta-sweep":
        betas = [float(s) for s in args.betas.split(",") if s.strip()]
        base = Path(args.outdir); base.mkdir(parents=True, exist_ok=True)
        for b in betas:
            bp = f"{b:.2f}".replace(".", "p")
            kname = base / f"K_2048_same_rho{args.rho_scale:.2f}".replace(".", "p") + Path(f"_a2{args.a2:.2f}".replace(".", "p") + f"_b{bp}.npy")
            # safer build: construct directly
            kname = base / f"K_2048_same_rho{str(args.rho_scale).replace('.','p')}_a2{str(args.a2).replace('.','p')}_b{bp}.npy"
            build_kernel(args.attrs, "same", args.a2, b, args.rho_scale, str(kname), plots=True)
            rep = str(kname).replace(".npy", "_tfine.json")
            analyze_kernel(str(kname), rep, args.t_grid, args.bootstrap, args.seed, args.mds_dim, args.max_nodes_eig, plots=True)
        # Summarize
        import pandas as pd
        df = df_from_reports(str(base / "K_2048_same_rho*_a2*_b*_tfine.json").replace("\\", "/"))
        if not df.empty:
            out_csv = base / "beta_sweep_summary.csv"
            df.to_csv(out_csv, index=False)
            try:
                plot_df = df.sort_values("beta")
                plot_series(plot_df["beta"], plot_df["spectral_dimension_mean"],
                            xlabel="beta", ylabel="spectral_dimension_mean",
                            title="β-sweep sdim", out_png=str(base / "beta_sweep_sdim.png"))
            except Exception as e:
                print(f"[warn] plot failed: {e}")
            print(f"[ok] beta sweep summary -> {out_csv}")
        else:
            print("[warn] no summary rows found; check patterns.")
        return

    if args.cmd == "n-scaling":
        sizes = [int(s) for s in args.sizes.split(",") if s.strip()]
        for n in sizes:
            attrs = f"attrs_{n}.npz"
            # synth attrs if missing
            if not Path(attrs).exists():
                code, out_s, err_s = run_cmd([py_exe(), rtg_script_path(), "synth", "--n", str(n), "--seed", "1", "--out", attrs])
                sys.stdout.write(out_s)
                if code != 0:
                    sys.stderr.write(err_s); raise RuntimeError(f"synth failed for n={n}")
            kname = f"K_{n}_same_rho{str(args.rho_scale).replace('.','p')}_a2{str(args.a2).replace('.','p')}_b{str(args.beta).replace('.','p')}.npy"
            build_kernel(attrs, "same", args.a2, args.beta, args.rho_scale, kname, plots=True)
            rep = kname.replace(".npy", "_tfine.json")
            analyze_kernel(kname, rep, args.t_grid, args.bootstrap, args.seed, args.mds_dim, args.max_nodes_eig, plots=True)
        # Summarize
        import pandas as pd
        df = df_from_reports("K_*_same_rho*_a2*_b*_tfine.json")
        if not df.empty:
            target = dict(spin_mode="same", rho_scale=round(args.rho_scale,2), a2=round(args.a2,2), beta=round(args.beta,2))
            ns = df_n_scaling(df, target)
            out_csv = "n_scaling_summary.csv"
            ns.to_csv(out_csv, index=False)
            try:
                plot_series(ns["n"], ns["spectral_dimension_mean"],
                            xlabel="n", ylabel="spectral_dimension_mean",
                            title="n-scaling sdim", out_png="n_scaling_sdim.png")
            except Exception as e:
                print(f"[warn] plot failed: {e}")
            print(f"[ok] n-scaling summary -> {out_csv}")
        else:
            print("[warn] no reports found matching n-scaling pattern.")
        return

    if args.cmd == "make-summary":
        import pandas as pd
        # scan top-K
        top_df = pd.DataFrame()
        if Path(args.scan).exists():
            df = df_from_scan(args.scan)
            top_df = df.head(args.topk).copy()
            top_df.to_csv(args.csv, index=False)
            print(f"[ok] wrote top-{args.topk} scan rows to {args.csv}")
        else:
            print(f"[warn] scan json not found: {args.scan}")

        # all reports (very flexible; we parse kernel_file for meta)
        all_reports = df_from_reports(args.reports_glob)

        # Specific convenience slices if present
        beta_df = pd.DataFrame()
        nscale_df = pd.DataFrame()
        if not all_reports.empty:
            # β sweep at (n=2048, spin=same, rho_scale=1.16, a2=0.55)
            try:
                beta_df = df_beta_sweep(
                    all_reports, dict(n=2048, spin_mode="same", rho_scale=1.16, a2=0.55)
                )
            except Exception:
                pass
            # n scaling at (spin=same, rho=1.20, a2=0.50, beta=1.60)
            try:
                nscale_df = df_n_scaling(
                    all_reports, dict(spin_mode="same", rho_scale=1.20, a2=0.50, beta=1.60)
                )
            except Exception:
                pass

        # Write Excel
        with pd.ExcelWriter(args.excel, engine="openpyxl") as xw:
            if not top_df.empty:
                top_df.to_excel(xw, sheet_name="scan_top", index=False)
            if not all_reports.empty:
                all_reports.to_excel(xw, sheet_name="all_reports", index=False)
            if not beta_df.empty:
                beta_df.to_excel(xw, sheet_name="beta_sweep", index=False)
            if not nscale_df.empty:
                nscale_df.to_excel(xw, sheet_name="n_scaling", index=False)
        print(f"[ok] Excel summary -> {args.excel}")
        return

if __name__ == "__main__":
    main()
