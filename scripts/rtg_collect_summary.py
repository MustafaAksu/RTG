#!/usr/bin/env python3
"""
RTG kernel analysis — single master table + per-topic sheets.

What it does
------------
- Loads whatever it finds (by sensible defaults) among:
  * scan_2048_ultrafine.json
  * report_*_seed*.json                 (seed stability set)
  * *_rho1p16_a2p55_b*_tfine.json      (β-sweep at rho_scale=1.16, a2=0.55)
  * K_*_same_rho1p20_a2p50_b1p60_tfine.json  (n-scaling set)
  * K_*_opp_*_tfine.json OR report_opp_*.json (opp / phase-encode tests)
- Builds:
  * one "master" long table (column `section` identifies the subset)
  * tidy per-section tables on separate Excel sheets
- Saves: an Excel workbook and CSVs.

GPU acceleration
----------------
If `dpnp` is present (Intel GPU / oneAPI), numeric reductions use it; otherwise
we fall back to NumPy. Pandas is used for I/O/frames.

Usage
-----
python rtg_collect_summary.py --root . --out rtg_summary.xlsx

Optional flags:
  --topk 20                 # top-k for the scan table (by |sdim| then mds)
  --make-plots              # also produce quick .png plots
  --strict                  # error if a section has no files (default: skip)
  --patterns-json <k=v...>  # override any glob (see PATTERNS below)

Examples
--------
python rtg_collect_summary.py --root . --out results/rtg_summary.xlsx
python rtg_collect_summary.py --root C:/GIT/rtg-resonance-2025/RTG/scripts

Author: you + GPT-5 Pro
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

# Prefer dpnp when available; fall back to numpy.
#try:
#    import dpnp as _np
#    _BACKEND = "dpnp"
#except Exception:
import numpy as _np  # type: ignore
_BACKEND = "numpy"

import pandas as pd

# ------------------------------- Patterns ------------------------------------

PATTERNS: Dict[str, str] = {
    # Ultra-fine scan
    "scan": "scan_2048_ultrafine.json",

    # Seed stability at (rho_scale=1.16, a2=0.55, beta=1.40)
    "seed": "report_2048_rho1p16_a2p55_b1p40_seed*.json",

    # β sweep at fixed (rho_scale=1.16, a2=0.55); produced as *_tfine.json
    "beta": "*rho1p16_a2p55_b*_tfine.json",

    # n-scaling at fixed (rho_scale=1.20, a2=0.50, beta=1.60)
    "nscale": "K_*_same_rho1p20_a2p50_b1p60_tfine.json",

    # Opp / phase-encode tests (either report_* or K_* naming)
    "opp": "*(opp|phase-encode-opp)*_tfine.json",
}

# ------------------------------- Utilities -----------------------------------

def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)

def _asfloat(x: Any) -> Optional[float]:
    try:
        return float(x)
    except Exception:
        return None

def parse_beta_from_kernel_name(s: str) -> Optional[float]:
    """
    Parse beta from things like "..._b1p60.npy" or "..._b1p60_tfine.json".
    """
    m = re.search(r"_b(\d+)p(\d+)", s)
    if not m:
        return None
    return float(f"{m.group(1)}.{m.group(2)}")

def parse_n_from_kernel_name(s: str) -> Optional[int]:
    m = re.search(r"^K_(\d+)_", s)
    if not m:
        return None
    return int(m.group(1))

def to_host(a: Any) -> Any:
    """Convert dpnp arrays back to host numpy when needed for pandas."""
    try:
        # dpnp-compatible:
        return a.asnumpy()  # type: ignore[attr-defined]
    except Exception:
        return a

def robust_mean_std(values: Iterable[float]) -> Tuple[float, float]:
    arr = _np.asarray(list(values), dtype=float)
    if arr.size == 0:
        return float("nan"), float("nan")
    mu = float(_np.mean(arr))
    sd = float(_np.std(arr, ddof=1)) if arr.size > 1 else 0.0
    return mu, sd

# ------------------------------ Collectors -----------------------------------

def collect_scan_top(root: Path, topk: int) -> pd.DataFrame:
    path = root / PATTERNS["scan"]
    if not path.exists():
        return pd.DataFrame()

    data = load_json(path)
    rows = []
    for rec in data.get("scan", []):
        rows.append({
            "spin_mode": rec.get("spin_mode"),
            "rho_scale": rec.get("rho_scale"),
            "a2": rec.get("a2"),
            "beta": rec.get("beta"),
            "spectral_dim": rec.get("spectral_dim"),
            "mds_stress": rec.get("mds_stress"),
        })
    df = pd.DataFrame(rows)
    if df.empty:
        return df

    df["abs_sdim"] = df["spectral_dim"].abs()
    df = df.sort_values(["abs_sdim", "mds_stress"], ascending=[True, True]).head(topk)
    df.insert(0, "section", "scan_top")
    return df.drop(columns=["abs_sdim"])


def collect_seed_stability(root: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    files = sorted(root.glob(PATTERNS["seed"]))
    rows: List[Dict[str, Any]] = []
    for fp in files:
        j = load_json(fp)
        # seed from filename:
        seed = None
        m = re.search(r"seed(\d+)\.json$", fp.name)
        if m:
            seed = int(m.group(1))
        rows.append({
            "file": fp.name,
            "seed": seed,
            "sdim": _asfloat(j.get("spectral_dimension_mean")),
            "se": _asfloat(j.get("spectral_dimension_se")),
            "mds": _asfloat(j.get("mds_stress")),
        })
    df = pd.DataFrame(rows)
    if df.empty:
        return df, df

    mu, sd = robust_mean_std(df["sdim"].dropna())
    summary = pd.DataFrame([{
        "n_files": len(df),
        "sdim_mean": mu,
        "sdim_std": sd,
        "sdim_se_from_std": (sd / math.sqrt(len(df))) if len(df) > 0 else float("nan"),
        "mds_mean": df["mds"].mean(),
    }])
    df.insert(0, "section", "seed_stability")
    return df, summary


def collect_beta_sweep(root: Path) -> pd.DataFrame:
    files = sorted(root.glob(PATTERNS["beta"]))
    rows: List[Dict[str, Any]] = []
    for fp in files:
        j = load_json(fp)
        kernel_file = j.get("kernel_file", fp.name)
        beta = j.get("beta") or parse_beta_from_kernel_name(kernel_file) or parse_beta_from_kernel_name(fp.name)
        rows.append({
            "beta": beta,
            "sdim": _asfloat(j.get("spectral_dimension_mean")),
            "se": _asfloat(j.get("spectral_dimension_se")),
            "mds": _asfloat(j.get("mds_stress")),
            "file": fp.name,
        })
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df = df.sort_values("beta")
    df.insert(0, "section", "beta_sweep")
    return df


def collect_n_scaling(root: Path) -> pd.DataFrame:
    files = sorted(root.glob(PATTERNS["nscale"]))
    rows: List[Dict[str, Any]] = []
    for fp in files:
        j = load_json(fp)
        kernel_file = j.get("kernel_file", fp.name)
        n = parse_n_from_kernel_name(kernel_file) or parse_n_from_kernel_name(fp.name)
        rows.append({
            "n": n,
            "sdim": _asfloat(j.get("spectral_dimension_mean")),
            "se": _asfloat(j.get("spectral_dimension_se")),
            "mds": _asfloat(j.get("mds_stress")),
            "file": fp.name,
        })
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df = df.sort_values("n")
    df.insert(0, "section", "n_scaling")
    return df


def collect_opp_phase(root: Path) -> pd.DataFrame:
    # Find any *_opp_*_tfine.json or report_opp_*.json
    files = sorted([fp for fp in root.glob("**/*_tfine.json") if re.search(r"(opp|phase-encode-opp)", fp.name, re.I)])
    files += sorted([fp for fp in root.glob("report_opp_*") if fp.suffix.lower() == ".json"])

    seen = set()
    rows: List[Dict[str, Any]] = []
    for fp in files:
        if fp in seen:
            continue
        seen.add(fp)
        j = load_json(fp)
        kernel_file = str(j.get("kernel_file", fp.name))
        # Parse rho_scale and a2 if present in the filename pattern
        m_rho = re.search(r"rho(\d+)p(\d+)", kernel_file)
        rho_scale = float(f"{m_rho.group(1)}.{m_rho.group(2)}") if m_rho else None
        m_a2 = re.search(r"_a2(?:0)?(\d+)p(\d+)", kernel_file) or re.search(r"_a20?(\d+)p(\d+)", kernel_file)
        a2 = float(f"{m_a2.group(1)}.{m_a2.group(2)}") if m_a2 else None

        rows.append({
            "rho_scale": rho_scale,
            "a2": a2,
            "sdim": _asfloat(j.get("spectral_dimension_mean")),
            "se": _asfloat(j.get("spectral_dimension_se")),
            "mds": _asfloat(j.get("mds_stress")),
            "file": fp.name,
        })

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df.insert(0, "section", "opp_phase")
    return df


# ------------------------------ Orchestration --------------------------------

@dataclass
class Tables:
    master: pd.DataFrame
    scan_top: pd.DataFrame
    seed_rows: pd.DataFrame
    seed_summary: pd.DataFrame
    beta_sweep: pd.DataFrame
    n_scaling: pd.DataFrame
    opp_phase: pd.DataFrame

def build_tables(root: Path, topk: int, strict: bool) -> Tables:
    scan_top = collect_scan_top(root, topk)
    seed_rows, seed_summary = collect_seed_stability(root)
    beta_sweep = collect_beta_sweep(root)
    n_scaling = collect_n_scaling(root)
    opp_phase = collect_opp_phase(root)

    pieces = []
    for name, df in [
        ("scan_top", scan_top),
        ("seed_stability", seed_rows),
        ("beta_sweep", beta_sweep),
        ("n_scaling", n_scaling),
        ("opp_phase", opp_phase),
    ]:
        if df is None or df.empty:
            msg = f"[warn] no rows for section '{name}' (pattern: {PATTERNS.get(name, '-')})"
            if strict:
                raise FileNotFoundError(msg)
            else:
                print(msg, file=sys.stderr)
                continue
        pieces.append(df)

    master = pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame()
    return Tables(master, scan_top, seed_rows, seed_summary, beta_sweep, n_scaling, opp_phase)


def save_outputs(tables: Tables, out: Path, make_plots: bool) -> List[Path]:
    out.parent.mkdir(parents=True, exist_ok=True)
    saved: List[Path] = []

    # Excel with separate sheets
    with pd.ExcelWriter(out, engine="openpyxl") as xl:
        if not tables.master.empty:
            tables.master.to_excel(xl, sheet_name="master", index=False)
        if not tables.scan_top.empty:
            tables.scan_top.to_excel(xl, sheet_name="scan_top", index=False)
        if not tables.seed_rows.empty:
            tables.seed_rows.to_excel(xl, sheet_name="seed_rows", index=False)
        if not tables.seed_summary.empty:
            tables.seed_summary.to_excel(xl, sheet_name="seed_summary", index=False)
        if not tables.beta_sweep.empty:
            tables.beta_sweep.to_excel(xl, sheet_name="beta_sweep", index=False)
        if not tables.n_scaling.empty:
            tables.n_scaling.to_excel(xl, sheet_name="n_scaling", index=False)
        if not tables.opp_phase.empty:
            tables.opp_phase.to_excel(xl, sheet_name="opp_phase", index=False)

    saved.append(out)

    # Also drop CSVs (handy for quick diffs)
    base = out.with_suffix("")
    csv_map = {
        "master": tables.master,
        "scan_top": tables.scan_top,
        "seed_rows": tables.seed_rows,
        "seed_summary": tables.seed_summary,
        "beta_sweep": tables.beta_sweep,
        "n_scaling": tables.n_scaling,
        "opp_phase": tables.opp_phase,
    }
    for name, df in csv_map.items():
        if df is not None and not df.empty:
            p = Path(f"{base}_{name}.csv")
            df.to_csv(p, index=False)
            saved.append(p)

    # Optional quick plots
    if make_plots:
        import matplotlib.pyplot as plt

        if not tables.n_scaling.empty:
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.plot(tables.n_scaling["n"], tables.n_scaling["sdim"], marker="o")
            ax.set_xlabel("n")
            ax.set_ylabel("spectral_dimension_mean")
            ax.set_title("n vs spectral dimension (rho=1.20, a2=0.50, beta=1.60)")
            p = base.with_name(base.name + "_n_vs_sdim.png").with_suffix(".png")
            fig.savefig(p, bbox_inches="tight", dpi=150)
            plt.close(fig)
            saved.append(p)

        if not tables.beta_sweep.empty:
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.plot(tables.beta_sweep["beta"], tables.beta_sweep["sdim"], marker="o")
            ax.set_xlabel("beta")
            ax.set_ylabel("spectral_dimension_mean")
            ax.set_title("beta sweep (rho=1.16, a2=0.55)")
            p = base.with_name(base.name + "_beta_vs_sdim.png").with_suffix(".png")
            fig.savefig(p, bbox_inches="tight", dpi=150)
            plt.close(fig)
            saved.append(p)

    return saved


def parse_kv_overrides(items: List[str]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for kv in items:
        if "=" not in kv:
            raise ValueError(f"Bad pattern override '{kv}', expected key=glob")
        k, v = kv.split("=", 1)
        k = k.strip()
        v = v.strip()
        if not k or not v:
            raise ValueError(f"Bad pattern override '{kv}'")
        out[k] = v
    return out


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="RTG: build one master table + tidy sheets.")
    ap.add_argument("--root", type=Path, default=Path("."), help="Folder with JSON outputs.")
    ap.add_argument("--out", type=Path, default=Path("rtg_summary.xlsx"), help="Excel file to write.")
    ap.add_argument("--topk", type=int, default=20, help="Top-k rows for scan_2048_ultrafine.json.")
    ap.add_argument("--make-plots", action="store_true", help="Also save quick diagnostic plots (.png).")
    ap.add_argument("--strict", action="store_true", help="Error if a section is missing.")
    ap.add_argument("--patterns-json", nargs="*", default=[], help="Override any pattern: key=glob")
    args = ap.parse_args(argv)

    # Apply pattern overrides if passed
    if args.patterns_json:
        overrides = parse_kv_overrides(args.patterns_json)
        for k, v in overrides.items():
            if k not in PATTERNS:
                print(f"[warn] unknown pattern key '{k}' — ignoring", file=sys.stderr)
                continue
            PATTERNS[k] = v

    print(f"[info] backend: {_BACKEND}")
    print(f"[info] scanning root: {args.root.resolve()}")
    for k, v in PATTERNS.items():
        print(f"       {k:8s} -> {v}")

    # Build tables
    tables = build_tables(args.root, args.topk, strict=args.strict)

    if tables.master.empty:
        print("[warn] master table is empty. Check patterns and file presence.", file=sys.stderr)
    else:
        print(f"[info] master rows: {len(tables.master)}")

    # Save outputs
    saved = save_outputs(tables, args.out, args.make_plots)
    print("[info] wrote:")
    for p in saved:
        print(f"  - {p}")

    # Console summary for seed stability (if present)
    if not tables.seed_summary.empty:
        s = tables.seed_summary.iloc[0].to_dict()
        print("\n[seed stability] "
              f"n_files={int(s['n_files'])}, "
              f"sdim_mean={s['sdim_mean']:.9f}, "
              f"sdim_std={s['sdim_std']:.9f}, "
              f"se_from_std={s['sdim_se_from_std']:.9f}, "
              f"mds_mean={s['mds_mean']:.9f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
