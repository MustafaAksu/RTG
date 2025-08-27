#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RTG full pipeline utilities

Subcommands:
  - collate:   Summarize *_tfine.json or *_smallt.json into CSV/JSON tables.
  - fss-beta:  Finite-size scaling vs beta at fixed (rho_scale, a2); find crossings.
  - fss-rho:   Finite-size scaling vs rho_scale at fixed (beta, a2); find crossings.
  - mds-scan:  Summarize a scan_*.json (grid search) by |sdim| & stress.
  - curvature: Forman curvature on a weighted kernel (K_*.npy). Outputs CSV + JSON.
  - homology:  0D/1D graph homology & triangles across thresholds. CSV + JSON.
  - plot:      Minimal plotting helpers for FSS curves and kernel histograms.

Notes
-----
* This file does not replace `rtg_kernel_rg_v3.py`; it complements it.
* dpnp acceleration: we use dpnp when available for simple array ops; networkx ops
  run on CPU and we convert arrays with xp.asnumpy when needed.
* No external OT library is required; we implement Forman curvature (not Ollivier).
* All outputs are simple, transparent files you can version-control.
"""

import argparse, json, re, sys, os, glob, math, csv, textwrap
from collections import defaultdict, namedtuple
from dataclasses import dataclass, asdict

# ---------- Array backend (dpnp -> numpy fallback) ----------
try:
    import dpnp as xp    # Intel GPU-accelerated drop-in for NumPy
    _USING_DPNP = True
except Exception:
    import numpy as xp
    _USING_DPNP = False

import numpy as _np      # always keep CPU NumPy available for networkx interop

# ---------- Optional plotting ----------
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _HAS_MPL = True
except Exception:
    _HAS_MPL = False

# ---------- Graph utilities ----------
try:
    import networkx as nx
    _HAS_NX = True
except Exception:
    _HAS_NX = False

# ----------------- Utilities -----------------

def read_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def write_json(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True)

def write_csv(rows, path, header=None):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if header:
            w.writerow(header)
        for r in rows:
            w.writerow(r)

def ensure_dir(path):
    d = os.path.dirname(os.path.abspath(path))
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)

def np_from_xp(a):
    # Convert dpnp or numpy array to host numpy array
    try:
        import dpctl
        return _np.asarray(a) if isinstance(a, _np.ndarray) else _np.asarray(a.get())
    except Exception:
        try:
            return a.asnumpy()
        except Exception:
            return _np.asarray(a)

def kernel_load(path):
    """Load a kernel matrix from .npy file into host numpy array."""
    K = _np.load(path)
    if hasattr(K, "asnumpy"):  # dpnp ndarray
        K = K.asnumpy()
    return _np.asarray(K, dtype=float)

# -------- Filename parsing helpers for K_* pattern --------

K_BETA_RE = re.compile(r'^K_(?P<n>\d+)_.*_rho(?P<rho>\d+)p(?P<rho2>\d+).*_a2p(?P<a2>\d+).*_b(?P<b1>\d+)p(?P<b2>\d+).*\.npy$')
JSON_TFINE_RE = re.compile(r'^K_(?P<n>\d+)_.*_rho(?P<rho>\d+)p(?P<rho2>\d+).*_a2p(?P<a2>\d+).*_b(?P<b1>\d+)p(?P<b2>\d+)_t(fine|smallt)\.json$')
SCAN_FILE_RE = re.compile(r'^scan_.*\.json$')

def parse_beta_from_json_name(path):
    fn = os.path.basename(path)
    m = JSON_TFINE_RE.match(fn)
    if not m:
        return None
    b = float(f"{m.group('b1')}.{m.group('b2')}")
    n = int(m.group('n'))
    rho = float(f"{m.group('rho')}.{m.group('rho2')}")
    a2 = float(m.group('a2'))/100.0
    return n, rho, a2, b

# --------------- collate -----------------

def cmd_collate(args):
    """
    Gather spectral dimension & stress from *_tfine.json or *_smallt.json files.
    Output: CSV & JSON with columns: n, rho_scale, a2, beta, sdim, se, mds.
    """
    files = sorted(glob.glob(args.glob))
    rows = []
    for f in files:
        d = read_json(f)
        meta = parse_beta_from_json_name(f)
        if meta is None:
            # allow generic report_* names if they contain kernel_file with same pattern
            kpath = d.get("kernel_file", "")
            m2 = JSON_TFINE_RE.match(os.path.basename(kpath).replace(".npy","_tfine.json"))
            if not m2:
                continue
            n = int(m2.group('n'))
            rho = float(f"{m2.group('rho')}.{m2.group('rho2')}")
            a2 = float(m2.group('a2'))/100.0
            b  = float(f"{m2.group('b1')}.{m2.group('b2')}")
        else:
            n, rho, a2, b = meta

        sdim = float(d.get("spectral_dimension_mean", _np.nan))
        se   = float(d.get("spectral_dimension_se", _np.nan))
        mds  = float(d.get("mds_stress", _np.nan))
        rows.append([n, rho, a2, b, sdim, se, mds, os.path.basename(f)])

    rows.sort(key=lambda r: (r[0], r[1], r[2], r[3]))
    header = ["n","rho_scale","a2","beta","sdim","sdim_se","mds_stress","file"]
    ensure_dir(args.out_csv)
    write_csv([header] + rows, args.out_csv)
    out_json = args.out_csv.rsplit(".",1)[0] + ".json"
    write_json({"rows": [
        dict(zip(header, r)) for r in rows
    ]}, out_json)
    print(f"[collate] wrote {args.out_csv} and {out_json} ({len(rows)} rows)")

# --------------- FSS tools -----------------

def _group_by_n(rows):
    byn = defaultdict(list)
    for r in rows:
        n, rho, a2, beta, sdim, se, mds, fn = r
        byn[n].append((beta, sdim, se, rho, a2, fn))
    for k in byn:
        byn[k].sort(key=lambda t: t[0])
    return byn

def _group_by_n_beta(rows, target_beta, tol=1e-9):
    byn = defaultdict(list)
    for r in rows:
        n, rho, a2, beta, sdim, se, mds, fn = r
        if abs(beta - target_beta) <= tol:
            byn[n].append((rho, sdim, se, beta, a2, fn))
    for k in byn:
        byn[k].sort(key=lambda t: t[0])
    return byn

def _load_collated(csv_path):
    # returns rows [[n,rho,a2,beta,sdim,se,mds,file], ...]
    rows = []
    with open(csv_path, "r", encoding="utf-8") as f:
        rdr = csv.DictReader(f)
        for row in rdr:
            rows.append([int(row["n"]), float(row["rho_scale"]), float(row["a2"]), float(row["beta"]),
                         float(row["sdim"]), float(row["sdim_se"]), float(row["mds_stress"]), row["file"]])
    return rows

def _crossings(curveA, curveB):
    """Linear-interpolation crossings between two curves (beta,sdim)."""
    X = []
    A = sorted([(b, y) for (b,y,_,_,_,_) in curveA])
    B = sorted([(b, y) for (b,y,_,_,_,_) in curveB])
    K = min(len(A), len(B))
    for i in range(1, K):
        b0, a0 = A[i-1]; b1, a1 = A[i]
        c0, d0 = B[i-1]; c1, d1 = B[i]
        if abs((b1-b0) - (c1-c0)) > 1e-12:
            # grids do not align; skip this segment
            continue
        f0 = a0 - d0
        f1 = a1 - d1
        if f0 == 0 and f1 == 0:
            # perfectly overlapping in this segment; record the mid as crossing
            X.append((b0+b1)/2.0)
        elif f0 == 0:
            X.append(b0)
        elif f1 == 0:
            X.append(b1)
        elif f0 * f1 < 0:
            # sign change -> crossing between b0 and b1
            t = -f0 / (f1 - f0)
            X.append(b0 + t * (b1 - b0))
    return X

def cmd_fss_beta(args):
    """
    Finite-size scaling vs beta:
      * Filter collated CSV to fixed rho_scale & a2.
      * Group curves by n: sdim(beta) for each n.
      * Compute pairwise crossings; summarize β* cluster.
      * Optional plot.
    """
    rows = _load_collated(args.collated_csv)
    rows = [r for r in rows if abs(r[1]-args.rho_scale) <= 1e-12 and abs(r[2]-args.a2) <= 1e-12]
    byn = _group_by_n(rows)
    ns = sorted(byn.keys())
    out = {"rho_scale": args.rho_scale, "a2": args.a2, "n_list": ns, "curves": {}, "crossings": []}
    for n in ns:
        out["curves"][str(n)] = [{"beta": b, "sdim": y, "sdim_se": se} for (b,y,se,_,_,_) in byn[n]]

    # pairwise crossings
    for i in range(len(ns)-1):
        n1, n2 = ns[i], ns[i+1]
        xs = _crossings(byn[n1], byn[n2])
        out["crossings"].append({"n_pair": [n1, n2], "beta_star": xs})
    # Flatten all crossings for a crude consensus
    flat = [x for pair in out["crossings"] for x in pair["beta_star"]]
    if flat:
        out["beta_star_mean"] = float(_np.mean(flat))
        out["beta_star_std"] = float(_np.std(flat))
    ensure_dir(args.out_json)
    write_json(out, args.out_json)
    print(f"[fss-beta] wrote {args.out_json}")

    if args.plot and _HAS_MPL:
        fig, ax = plt.subplots(figsize=(6.5,4.2))
        for n in ns:
            xs = [b for (b,_,_,_,_,_) in byn[n]]
            ys = [y for (_,y,_,_,_,_) in byn[n]]
            ax.plot(xs, ys, marker='o', label=f"n={n}")
        if flat:
            ax.axvline(out["beta_star_mean"], linestyle="--")
        ax.set_xlabel(r"$\beta$")
        ax.set_ylabel("spectral dimension (sdim)")
        ax.legend()
        plt.tight_layout()
        png = args.out_json.rsplit(".",1)[0] + ".png"
        plt.savefig(png, dpi=160)
        print(f"[fss-beta] plot saved -> {png}")

def cmd_fss_rho(args):
    """
    Finite-size scaling vs rho_scale at fixed beta & a2.
    """
    rows = _load_collated(args.collated_csv)
    rows = [r for r in rows if abs(r[3]-args.beta) <= 1e-12 and abs(r[2]-args.a2) <= 1e-12]
    # regroup as sdim(rho) curves per n
    byn = defaultdict(list)
    for r in rows:
        n, rho, a2, beta, sdim, se, mds, fn = r
        byn[n].append((rho, sdim, se, beta, a2, fn))
    for k in byn:
        byn[k].sort(key=lambda t: t[0])
    ns = sorted(byn.keys())
    out = {"beta": args.beta, "a2": args.a2, "n_list": ns, "curves": {}, "crossings": []}
    for n in ns:
        xs = byn[n]
        out["curves"][str(n)] = [{"rho_scale": r, "sdim": y, "sdim_se": se} for (r,y,se,_,_,_) in xs]

    # pairwise crossings in rho
    def xrho(A, B):
        # both as (rho, y)
        X = []
        A = sorted([(r,y) for (r,y,_,_,_,_) in A]); 
        B = sorted([(r,y) for (r,y,_,_,_,_) in B])
        K = min(len(A), len(B))
        for i in range(1, K):
            r0, a0 = A[i-1]; r1, a1 = A[i]
            s0, d0 = B[i-1]; s1, d1 = B[i]
            if abs((r1-r0) - (s1-s0)) > 1e-12:
                continue
            f0 = a0 - d0; f1 = a1 - d1
            if f0 == 0 and f1 == 0:
                X.append((r0+r1)/2.0)
            elif f0 == 0:
                X.append(r0)
            elif f1 == 0:
                X.append(r1)
            elif f0 * f1 < 0:
                t = -f0 / (f1 - f0)
                X.append(r0 + t * (r1 - r0))
        return X

    for i in range(len(ns)-1):
        n1, n2 = ns[i], ns[i+1]
        xs = xrho(byn[n1], byn[n2])
        out["crossings"].append({"n_pair": [n1, n2], "rho_star": xs})
    flat = [x for pair in out["crossings"] for x in pair["rho_star"]]
    if flat:
        out["rho_star_mean"] = float(_np.mean(flat))
        out["rho_star_std"]  = float(_np.std(flat))
    ensure_dir(args.out_json)
    write_json(out, args.out_json)
    print(f"[fss-rho] wrote {args.out_json}")

    if args.plot and _HAS_MPL:
        fig, ax = plt.subplots(figsize=(6.5,4.2))
        for n in ns:
            xs = [r for (r,_,_,_,_,_) in byn[n]]
            ys = [y for (_,y,_,_,_,_) in byn[n]]
            ax.plot(xs, ys, marker='o', label=f"n={n}")
        if flat:
            ax.axvline(out["rho_star_mean"], linestyle="--")
        ax.set_xlabel(r"$\rho$ scale")
        ax.set_ylabel("spectral dimension (sdim)")
        ax.legend()
        plt.tight_layout()
        png = args.out_json.rsplit(".",1)[0] + ".png"
        plt.savefig(png, dpi=160)
        print(f"[fss-rho] plot saved -> {png}")

# --------------- mds-scan -----------------

def cmd_mds_scan(args):
    """
    Summarize scan_*.json (grid-search outputs from `rtg_kernel_rg_v3.py scan`).
    Produces a CSV of top-K entries by |sdim| (closest to 0) and then by lowest stress.
    """
    # find scan json
    if os.path.isdir(args.path):
        candidates = [p for p in glob.glob(os.path.join(args.path, "scan_*.json"))]
    else:
        candidates = [args.path] if os.path.exists(args.path) else []
    if not candidates:
        print("[mds-scan] no scan_*.json found")
        return
    rows = []
    for p in candidates:
        d = read_json(p)
        scan = d.get("scan", [])
        for e in scan:
            rows.append([e.get("spin_mode"), float(e.get("rho_scale")), float(e.get("a2")),
                         float(e.get("beta")), float(e.get("spectral_dim")), float(e.get("mds_stress")),
                         os.path.basename(p)])
    # rank by |sdim| ascending, then stress
    rows.sort(key=lambda r: (abs(r[4]), r[5]))
    header = ["spin_mode","rho_scale","a2","beta","spectral_dim","mds_stress","source"]
    K = args.top if args.top>0 else len(rows)
    sel = rows[:K]
    ensure_dir(args.out_csv)
    write_csv([header] + sel, args.out_csv)
    print(f"[mds-scan] wrote {args.out_csv} (top {K} rows of {len(rows)})")

# --------------- curvature (Forman) -----------------

def forman_curvature_edges(K, tau=0.0):
    """
    Forman curvature for weighted graphs given weight matrix K (numpy array).
    Returns:
      edges: list of (i,j,w, kappa_ij)
      nodes: list of (i, RicF_i, deg_w, deg_c)
    """
    assert K.ndim == 2 and K.shape[0] == K.shape[1], "K must be square"
    n = K.shape[0]
    # threshold
    W = _np.array(K, dtype=float)
    W[_np.isnan(W)] = 0.0
    # consider i<j
    deg_w = W.sum(axis=1)  # weighted degree
    deg_c = (W >= tau).sum(axis=1) - 1  # count excluding diagonal if self-weights exist

    edges = []
    # Precompute neighbor lists
    nbrs = [ _np.where(W[i] >= tau)[0] for i in range(n) ]
    for i in range(n):
        for j in nbrs[i]:
            if j <= i: 
                continue
            w_ij = W[i,j]
            if w_ij < tau or w_ij <= 0: 
                continue
            # Forman edge curvature (node weights -> weighted degrees deg_w)
            # F(i,j) = deg_w(i)/w_ij + deg_w(j)/w_ij 
            #          - sum_{k in N(i)\{j}} deg_w(i)/sqrt(w_ij*w_ik)
            #          - sum_{k in N(j)\{i}} deg_w(j)/sqrt(w_ij*w_jk)
            a = deg_w[i] / w_ij + deg_w[j] / w_ij
            si = 0.0
            for k in nbrs[i]:
                if k == j or W[i,k] < tau or W[i,k] <= 0: 
                    continue
                si += deg_w[i] / math.sqrt(max(w_ij*W[i,k], 1e-30))
            sj = 0.0
            for k in nbrs[j]:
                if k == i or W[j,k] < tau or W[j,k] <= 0:
                    continue
                sj += deg_w[j] / math.sqrt(max(w_ij*W[j,k], 1e-30))
            kappa = a - si - sj
            edges.append((i, j, w_ij, kappa))

    # Node-level Ricci (sum of adjacent edge curvatures)
    RicF = _np.zeros(n, dtype=float)
    for (i,j,w,kappa) in edges:
        RicF[i] += kappa
        RicF[j] += kappa
    nodes = [(i, float(RicF[i]), float(deg_w[i]), int(deg_c[i])) for i in range(n)]
    return edges, nodes

def cmd_curvature(args):
    if not os.path.exists(args.kernel):
        print(f"[curvature] kernel not found: {args.kernel}", file=sys.stderr)
        sys.exit(2)
    K = kernel_load(args.kernel)
    edges, nodes = forman_curvature_edges(K, tau=args.tau)
    # Write CSVs
    base = os.path.splitext(os.path.basename(args.kernel))[0]
    e_csv = args.out_prefix + "_curv_edges.csv"
    n_csv = args.out_prefix + "_curv_nodes.csv"
    write_csv([["i","j","w_ij","forman_edge"]]+edges, e_csv)
    write_csv([["i","RicF","deg_w","deg_c"]]+nodes, n_csv)
    # Summary JSON
    e_vals = _np.array([e[3] for e in edges], dtype=float) if edges else _np.array([],dtype=float)
    n_vals = _np.array([n[1] for n in nodes], dtype=float) if nodes else _np.array([],dtype=float)
    summary = {
        "kernel": os.path.basename(args.kernel),
        "tau": args.tau,
        "n_nodes": int(K.shape[0]),
        "n_edges": int(len(edges)),
        "edge_curv_mean": float(e_vals.mean()) if e_vals.size else None,
        "edge_curv_std": float(e_vals.std()) if e_vals.size else None,
        "node_RicF_mean": float(n_vals.mean()) if n_vals.size else None,
        "node_RicF_std": float(n_vals.std()) if n_vals.size else None,
        "edges_csv": os.path.basename(e_csv),
        "nodes_csv": os.path.basename(n_csv),
    }
    out_json = args.out_prefix + "_curv_summary.json"
    write_json(summary, out_json)
    print(f"[curvature] wrote {e_csv}, {n_csv}, {out_json}")

# --------------- homology (graph-level β0, β1, triangles) -----------------

def graph_from_kernel(K, tau):
    if not _HAS_NX:
        raise RuntimeError("networkx not available")
    W = _np.array(K, dtype=float)
    n = W.shape[0]
    G = nx.Graph()
    G.add_nodes_from(range(n))
    for i in range(n):
        # start j at i+1 to avoid duplicates
        for j in range(i+1, n):
            w = W[i,j]
            if w >= tau:
                G.add_edge(i, j, weight=float(w))
    return G

def betti_graph(G):
    # β0 is number of connected components
    beta0 = nx.number_connected_components(G)
    # β1 (first Betti of 1-skeleton) = m - n + β0
    n = G.number_of_nodes()
    m = G.number_of_edges()
    beta1 = m - n + beta0
    try:
        tri_counts = sum(nx.triangles(G).values()) // 3
    except Exception:
        tri_counts = None
    euler = n - m + (tri_counts if tri_counts is not None else 0)
    return beta0, beta1, tri_counts, euler

def cmd_homology(args):
    if not _HAS_NX:
        print("[homology] networkx not available. Please install networkx.", file=sys.stderr)
        sys.exit(2)
    if not os.path.exists(args.kernel):
        print(f"[homology] kernel not found: {args.kernel}", file=sys.stderr)
        sys.exit(2)
    K = kernel_load(args.kernel)
    # levels
    levels = []
    if args.levels:
        # parse as comma-separated floats
        for tok in args.levels.split(","):
            tok = tok.strip()
            if not tok:
                continue
            levels.append(float(tok))
    else:
        # auto: 32 linearly spaced thresholds between small positive and max(K)
        Kpos = K[K>0]
        if Kpos.size == 0:
            levels = [0.0]
        else:
            wmin = float(_np.percentile(Kpos, 5.0))
            wmax = float(Kpos.max())
            levels = list(_np.linspace(wmin, wmax, 32))

    rows = []
    for tau in levels:
        G = graph_from_kernel(K, tau)
        beta0, beta1, tris, euler = betti_graph(G)
        rows.append([tau, G.number_of_nodes(), G.number_of_edges(), beta0, beta1, tris, euler])
    header = ["tau","n","m","beta0","beta1","triangles","euler_2skeleton"]
    ensure_dir(args.out_csv)
    write_csv([header]+rows, args.out_csv)
    # summary
    summary = {
        "kernel": os.path.basename(args.kernel),
        "levels": levels,
        "rows_csv": os.path.basename(args.out_csv),
        "n_levels": len(levels),
        "notes": "β1 here is the cyclomatic number of the graph (1-skeleton). Triangles counted via networkx; euler uses 2-skeleton only."
    }
    out_json = args.out_csv.rsplit(".",1)[0] + ".json"
    write_json(summary, out_json)
    print(f"[homology] wrote {args.out_csv} and {out_json} ({len(rows)} levels)")

# --------------- plotting helpers -----------------

def cmd_plot(args):
    if not _HAS_MPL:
        print("[plot] matplotlib not available. Skipping.")
        return
    if args.kind == "kernel-hist":
        K = kernel_load(args.kernel)
        W = K[_np.triu_indices_from(K, k=1)]
        W = W[W>0]
        fig, ax = plt.subplots(figsize=(6,4))
        ax.hist(W, bins=50)
        ax.set_xlabel("Kernel weight")
        ax.set_ylabel("count")
        ax.set_title(os.path.basename(args.kernel))
        plt.tight_layout()
        out = args.out or (os.path.splitext(args.kernel)[0] + "_Khist.png")
        plt.savefig(out, dpi=160)
        print(f"[plot] saved -> {out}")
    elif args.kind == "fss-beta":
        J = read_json(args.fss_json)
        curves = J.get("curves", {})
        fig, ax = plt.subplots(figsize=(6.5,4.2))
        for n_str, pts in curves.items():
            xs = [p["beta"] for p in pts]
            ys = [p["sdim"] for p in pts]
            ax.plot(xs, ys, marker='o', label=f"n={n_str}")
        if "beta_star_mean" in J:
            ax.axvline(J["beta_star_mean"], linestyle="--")
        ax.set_xlabel(r"$\beta$")
        ax.set_ylabel("spectral dimension (sdim)")
        ax.legend()
        plt.tight_layout()
        out = args.out or (os.path.splitext(args.fss_json)[0] + ".png")
        plt.savefig(out, dpi=160)
        print(f"[plot] saved -> {out}")
    elif args.kind == "fss-rho":
        J = read_json(args.fss_json)
        curves = J.get("curves", {})
        fig, ax = plt.subplots(figsize=(6.5,4.2))
        for n_str, pts in curves.items():
            xs = [p["rho_scale"] for p in pts]
            ys = [p["sdim"] for p in pts]
            ax.plot(xs, ys, marker='o', label=f"n={n_str}")
        if "rho_star_mean" in J:
            ax.axvline(J["rho_star_mean"], linestyle="--")
        ax.set_xlabel(r"$\rho$ scale")
        ax.set_ylabel("spectral dimension (sdim)")
        ax.legend()
        plt.tight_layout()
        out = args.out or (os.path.splitext(args.fss_json)[0] + ".png")
        plt.savefig(out, dpi=160)
        print(f"[plot] saved -> {out}")
    else:
        print(f"[plot] unknown kind: {args.kind}")

# --------------- CLI setup -----------------

def main():
    ap = argparse.ArgumentParser(
        prog="rtg_full_pipeline.py",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=textwrap.dedent("""
        RTG full pipeline helpers (FSS, scans, curvature, homology).

        Examples
        --------
        # 1) Collate JSON reports into a single table
        python rtg_full_pipeline.py collate --glob "K_*_tfine.json" --out-csv rtg_collated.csv

        # 2) FSS vs beta at fixed rho_scale & a2 (with plot)
        python rtg_full_pipeline.py fss-beta --collated-csv rtg_collated.csv \
            --rho-scale 1.16 --a2 0.55 --out-json fss_beta_rho1p16_a2p55.json --plot

        # 3) FSS vs rho_scale at fixed beta & a2 (with plot)
        python rtg_full_pipeline.py fss-rho --collated-csv rtg_collated.csv \
            --beta 1.50 --a2 0.55 --out-json fss_rho_b1p50_a2p55.json --plot

        # 4) Summarize a grid scan file by |sdim|, stress
        python rtg_full_pipeline.py mds-scan --path scan_2048_ultrafine.json --top 20 --out-csv scan_top20.csv

        # 5) Forman curvature from a kernel
        python rtg_full_pipeline.py curvature --kernel K_4096_same_rho1p16_a2p55_b1p50.npy \
            --tau 0.0 --out-prefix curv_4096_rho1p16_b1p50

        # 6) Graph homology across thresholds
        python rtg_full_pipeline.py homology --kernel K_4096_same_rho1p16_a2p55_b1p50.npy \
            --levels 0.2,0.5,1.0,1.5,2.0 --out-csv homology_4096_levels.csv

        # 7) Plot helper (FSS or kernel histogram)
        python rtg_full_pipeline.py plot --kind kernel-hist --kernel K_4096_same_rho1p16_a2p55_b1p50.npy
        python rtg_full_pipeline.py plot --kind fss-beta --fss-json fss_beta_rho1p16_a2p55.json
        """))
    sub = ap.add_subparsers(dest="cmd", required=True)

    # collate
    ap_col = sub.add_parser("collate", help="Collate *_tfine.json/_smallt.json into CSV/JSON")
    ap_col.add_argument("--glob", required=True, help="Glob for report json files (e.g., 'K_*_tfine.json')")
    ap_col.add_argument("--out-csv", required=True, help="Output CSV path")
    ap_col.set_defaults(func=cmd_collate)

    # fss-beta
    ap_fb = sub.add_parser("fss-beta", help="Finite-size scaling vs beta at fixed rho_scale & a2")
    ap_fb.add_argument("--collated-csv", required=True, help="CSV produced by 'collate'")
    ap_fb.add_argument("--rho-scale", type=float, required=True)
    ap_fb.add_argument("--a2", type=float, required=True)
    ap_fb.add_argument("--out-json", required=True)
    ap_fb.add_argument("--plot", action="store_true")
    ap_fb.set_defaults(func=cmd_fss_beta)

    # fss-rho
    ap_fr = sub.add_parser("fss-rho", help="Finite-size scaling vs rho_scale at fixed beta & a2")
    ap_fr.add_argument("--collated-csv", required=True, help="CSV produced by 'collate'")
    ap_fr.add_argument("--beta", type=float, required=True)
    ap_fr.add_argument("--a2", type=float, required=True)
    ap_fr.add_argument("--out-json", required=True)
    ap_fr.add_argument("--plot", action="store_true")
    ap_fr.set_defaults(func=cmd_fss_rho)

    # mds-scan
    ap_ms = sub.add_parser("mds-scan", help="Summarize scan_*.json files by |sdim| then stress")
    ap_ms.add_argument("--path", required=True, help="scan_*.json file OR a folder containing them")
    ap_ms.add_argument("--top", type=int, default=20, help="Top-K rows to keep (default 20; <=0 means all)")
    ap_ms.add_argument("--out-csv", required=True, help="Output CSV path")
    ap_ms.set_defaults(func=cmd_mds_scan)

    # curvature (Forman)
    ap_cv = sub.add_parser("curvature", help="Forman curvature on a weighted kernel matrix (.npy)")
    ap_cv.add_argument("--kernel", required=True, help="K_*.npy kernel matrix")
    ap_cv.add_argument("--tau", type=float, default=0.0, help="Weight threshold; keep edges with w>=tau")
    ap_cv.add_argument("--out-prefix", required=True, help="Prefix for outputs (CSV + JSON)")
    ap_cv.set_defaults(func=cmd_curvature)

    # homology
    ap_hm = sub.add_parser("homology", help="Graph-level homology (β0, β1) & triangles across thresholds")
    ap_hm.add_argument("--kernel", required=True, help="K_*.npy kernel matrix")
    ap_hm.add_argument("--levels", default=None, help="Comma-separated thresholds (e.g., '0.2,0.5,1.0'); default=auto 32 levels")
    ap_hm.add_argument("--out-csv", required=True, help="Output CSV path for per-level summary")
    ap_hm.set_defaults(func=cmd_homology)

    # plot helper
    ap_pl = sub.add_parser("plot", help="Plotting helpers")
    ap_pl.add_argument("--kind", required=True, choices=["kernel-hist","fss-beta","fss-rho"])
    ap_pl.add_argument("--kernel", default=None)
    ap_pl.add_argument("--fss-json", default=None)
    ap_pl.add_argument("--out", default=None)
    ap_pl.set_defaults(func=cmd_plot)

    args = ap.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()
