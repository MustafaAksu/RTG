#!/usr/bin/env python3
"""
rtg_full_pipeline.py

One-stop CLI for:
  - fss-beta       : find β* by variance minimization (and try sign-change crossings)
  - fss-rho        : collate ρ sweeps at fixed β
  - sdim-plateau   : instantaneous d_s(t) from analyze JSON
  - curvature      : Forman & (approximate) Ollivier on a thresholded sparse graph from K.npy
  - homology       : graph Betti0/Betti1 across thresholds (optionally with triangle fill stats)
  - fit-sdim-vs-n  : weighted fit of sdim(n) to d_inf + c*n^{-omega}
  - export         : merge common JSON fields to CSV/XLSX
  - chsh           : simple CHSH S-value from attrs + kernel edges (optional sanity check)

It expects:
  - analyze JSON files produced by your rtg_kernel_rg_v3.py (for sdim, slope, stress, etc.)
  - kernel matrices K_*.npy (float weights, symmetric)

It will try to use dpnp (Intel GPU) if available; otherwise falls back to numpy.
"""

import argparse, json, glob, re, os, csv, math, statistics, warnings

# ---- NumPy / dpnp selection -------------------------------------------------
USE_DPNP = False
"""
try:
    import dpnp as np  # type: ignore
    USE_DPNP = True
except Exception:
    import numpy as np  # type: ignore
    USE_DPNP = False
"""
import numpy as np  # type: ignore

def _to_numpy(x):
    """Convert dpnp arrays to numpy for file I/O and Py-only ops."""
    if USE_DPNP and hasattr(x, "get"):
        return x.get()
    return x

# ---- Utilities --------------------------------------------------------------
def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_json(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)

def save_csv(rows, header, path):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if header: w.writerow(header)
        for r in rows: w.writerow(r)

def parse_kernel_fields_from_jsonfile(fname):
    # Tries to parse n and beta from filename like:
    #   K_4096_same_rho1p16_a2p55_b1p50_smallt.json
    m = re.match(r"^K_(\d+)_.*_b(\d+)p(\d+)_.*\.json$", os.path.basename(fname))
    n = int(m.group(1)) if m else None
    beta = float(f"{m.group(2)}.{m.group(3)}") if m else None
    return n, beta

def parse_rho_from_jsonfile(fname):
    m = re.match(r"^K_(\d+)_.*_rho(\d+)p(\d+)_.*\.json$", os.path.basename(fname))
    n = int(m.group(1)) if m else None
    rho = float(f"{m.group(2)}.{m.group(3)}") if m else None
    return n, rho

def finite_diff_loglog(t, P):
    """Central-difference slope d log P / d log t at each t."""
    t = np.asarray(t); P = np.asarray(P)
    lt = np.log(t); lP = np.log(P)
    n = t.shape[0]
    slope = np.empty(n, dtype=float)
    # endpoints: one-sided, interior: central
    slope[0]  = (lP[1] - lP[0]) / (lt[1] - lt[0])
    slope[-1] = (lP[-1] - lP[-2]) / (lt[-1] - lt[-2])
    if n > 2:
        slope[1:-1] = (lP[2:] - lP[:-2]) / (lt[2:] - lt[:-2])
    return _to_numpy(slope)

# ---- Subcommands ------------------------------------------------------------

def cmd_fss_beta(args):
    files = glob.glob(args.glob)
    curves = {}  # n -> list of (beta, sdim, se)
    betas_all = set()
    for f in files:
        n, beta = parse_kernel_fields_from_jsonfile(f)
        if n is None or beta is None:
            continue
        d = load_json(f)
        sdim = d.get("spectral_dimension_mean")
        se   = d.get("spectral_dimension_se")
        if sdim is None: continue
        curves.setdefault(n, []).append((beta, sdim, se))
        betas_all.add(beta)

    # clean/sort
    for n in list(curves.keys()):
        curves[n].sort(key=lambda x: x[0])

    # print
    print("== Curves (sdim ± se) ==")
    for n in sorted(curves.keys()):
        print(f"n={n}")
        for beta, sdim, se in curves[n]:
            if se is None: se = float("nan")
            print(f"  beta={beta:.2f}  sdim={sdim:.6f} ± {se:.6f}")

    # try sign-change crossings (pairwise, same index method)
    def crossings(a, b):
        X=[]
        A=sorted(a); B=sorted(b)
        m = min(len(A), len(B))
        for i in range(1, m):
            b0, a0, _ = A[i-1]; b1, a1, _ = A[i]
            c0, d0, _ = B[i-1]; c1, d1, _ = B[i]
            if abs((b1-b0)-(c1-c0)) > 1e-9:
                continue
            diff0 = a0 - d0
            diff1 = a1 - d1
            if diff0 == 0.0:
                X.append(b0)
            elif diff0*diff1 < 0:
                t = -diff0 / (diff1 - diff0 + 1e-12)
                X.append(b0 + t*(b1-b0))
        return X

    ns = sorted(curves.keys())
    xinfo = []
    for i in range(len(ns)-1):
        n1, n2 = ns[i], ns[i+1]
        xs = crossings(curves[n1], curves[n2])
        if xs:
            print(f"[INFO] Crossing between n={n1} and n={n2} at β ≈ {xs}")
        else:
            print(f"[INFO] No sign-change crossing detected for n={n1} vs n={n2}")
        xinfo.append({"pair":[n1,n2],"crossings":xs})

    # variance minimization over betas common to all curves
    beta_candidates = sorted(set.intersection(*[set(b for b,_,_ in curves[n]) for n in ns]))
    best = None
    rows = []
    for b in beta_candidates:
        vals=[]
        for n in ns:
            for bb, s, _ in curves[n]:
                if abs(bb-b) < 1e-9:
                    vals.append(s)
                    break
        if len(vals) == len(ns):
            var = statistics.pvariance(vals) if len(vals) > 1 else 0.0
            rows.append([b, var] + vals)
            if best is None or var < best[1]:
                best = (b, var)

    if best:
        bstar, var = best
        print(f"\n[INFO] Variance-minimizing beta across curves: beta* ≈ {bstar:.6f} (var={var:.3e})")
        # rough slope around beta* for each n
        for n in ns:
            vv = curves[n]
            # local slope via least squares on (beta, sdim)
            xb = np.asarray([v[0] for v in vv], dtype=float)
            y  = np.asarray([v[1] for v in vv], dtype=float)
            # center at bstar
            X = np.vstack([xb - bstar, np.ones_like(xb)]).T
            beta_slope, _ = np.linalg.lstsq(X, y, rcond=None)[0]
            print(f"  slope d(sdim)/d(beta) @ n={n}: {beta_slope:.9e}")
    else:
        bstar, var = None, None

    out = args.out or "beta_fp"
    save_csv(rows=[["beta","var"] + [f"sdim_n{n}" for n in ns]] + rows,
             header=None, path=f"{out}.csv")
    result = {
        "curves": {str(n): [{"beta":b,"sdim":s,"se":se} for (b,s,se) in curves[n]] for n in curves},
        "crossings": xinfo,
        "beta_star_varmin": bstar,
        "beta_star_var": var
    }
    save_json(result, f"{out}.json")
    print(f"\n[OK] Wrote {out}.json and {out}.csv")

def cmd_fss_rho(args):
    files = glob.glob(args.glob)
    rows = [["n","rho_scale","sdim","se"]]
    byn = {}
    for f in files:
        n, rho = parse_rho_from_jsonfile(f)
        if n is None or rho is None: continue
        d = load_json(f)
        sdim = d.get("spectral_dimension_mean")
        se   = d.get("spectral_dimension_se")
        rows.append([n, rho, sdim, se])
        byn.setdefault(n, []).append((rho, sdim, se))
    for n,v in byn.items():
        v.sort()
        print(f"n={n}")
        for rho, sdim, se in v:
            print(f"  rho={rho:.2f}  sdim={sdim:.6f} ± {se:.6f}")
    out = args.out or "rho_fss"
    save_csv(rows, header=None, path=f"{out}.csv")
    save_json({"data": rows[1:]}, f"{out}.json")
    print(f"[OK] Wrote {out}.json and {out}.csv")

def cmd_sdim_plateau(args):
    d = load_json(args.file)
    hm = d.get("heat_meta", {})
    t  = hm.get("t_grid")
    P  = hm.get("heat_trace")
    if not t or not P:
        raise SystemExit("heat_meta.t_grid / heat_meta.heat_trace missing")
    slope = finite_diff_loglog(np.array(t, dtype=float), np.array(P, dtype=float))
    sdim_inst = -2.0 * slope
    rows = [["t","sdim_inst"]]+[[f"{float(tt):g}", f"{float(ss):.9f}"] for tt,ss in zip(t, _to_numpy(sdim_inst))]
    out = args.out or (os.path.splitext(os.path.basename(args.file))[0] + "_plateau.csv")
    save_csv(rows, header=None, path=out)
    print("t, sdim_inst")
    for tt, ss in zip(t, _to_numpy(sdim_inst)):
        print(f"{tt}, {ss:.9f}")
    print(f"[OK] Wrote {out}")

def kernel_to_sparse_edges(K, tau=None, quantile=None, kmax=None):
    """
    Build a sparse edge list from dense K (symmetric nonnegative).
    - If tau is set: keep weights >= tau.
    - Else if quantile is set in (0,1): choose tau at that global quantile of nonzero weights.
    - Else if kmax is set: keep top-k neighbors per node.
    Returns: (edges list of (i,j,w)), deg array of weighted degrees, neighbors dict
    """
    n = K.shape[0]
    # consider only upper triangle
    tri = np.triu(K, 1)
    w = tri[tri > 0]
    if w.size == 0:
        return [], np.zeros(n), {i:set() for i in range(n)}
    if quantile is not None and tau is None and kmax is None:
        q = float(quantile)
        if q <= 0 or q >= 1:
            raise ValueError("quantile must be in (0,1)")
        tau = float(np.quantile(_to_numpy(w), q))
    edges=[]
    if kmax is not None:
        # top-k neighbors per node
        k = int(kmax)
        for i in range(n):
            row = K[i].copy()
            row[i] = 0.0
            idx = np.argsort(-row)[:k]
            for j in _to_numpy(idx):
                if j <= i: continue
                wij = float(row[j])
                if wij > 0:
                    edges.append((i,j,wij))
    else:
        # threshold
        assert tau is not None
        I,J = np.where(tri >= tau)
        for i,j in zip(_to_numpy(I), _to_numpy(J)):
            edges.append((int(i), int(j), float(K[i,j])))

    deg = np.zeros(n, dtype=float)
    neigh = {i:set() for i in range(n)}
    for i,j,wv in edges:
        deg[i]+=wv; deg[j]+=wv
        neigh[i].add(j); neigh[j].add(i)
    return edges, deg, neigh

def cmd_curvature(args):
    K = np.load(args.kernel)
    mode = args.mode
    if (args.quantile is None) and (args.tau is None) and (args.kmax is None):
        # default: very sparse top-0.2% global edges
        args.quantile = 0.998
    edges, deg, neigh = kernel_to_sparse_edges(K,
                                               tau=args.tau,
                                               quantile=args.quantile,
                                               kmax=args.kmax)
    n = K.shape[0]
    print(f"[curvature] n={n}, edges_kept={len(edges)}")
    edge_rows=[]
    node_val = np.zeros(n, dtype=float)

    if mode in ("forman","both"):
        # Weighted Forman (simplified Sreejith+ generalization)
        for (i,j,w) in edges:
            wi = deg[i] if deg[i] > 0 else 1.0
            wj = deg[j] if deg[j] > 0 else 1.0
            s1 = 0.0
            for k in neigh[i]:
                if k == j: continue
                wik = K[i,k]
                if wik > 0:
                    s1 += 1.0/math.sqrt(max(w*wik, 1e-30))
            s2 = 0.0
            for k in neigh[j]:
                if k == i: continue
                wjk = K[j,k]
                if wjk > 0:
                    s2 += 1.0/math.sqrt(max(w*wjk, 1e-30))
            kF = (wi/w + wj/w) - (s1 + s2)
            edge_rows.append([i,j,w,kF,"forman"])
            node_val[i]+=kF; node_val[j]+=kF

    if mode in ("ollivier","both"):
        # Approximate Ollivier-Ricci: use neighbor-weight distributions and overlap mass
        # κ(i,j) ≈ sum_k min(μ_i(k), μ_j(k))  (ground distance ≈ 1 on overlap, ≈2 elsewhere)
        # This is a fast proxy; for exact W1 one would solve a transport LP per edge.
        for (i,j,w) in edges:
            Ni = list(neigh[i]); Nj = list(neigh[j])
            if len(Ni)==0 or len(Nj)==0:
                kOR = 0.0
            else:
                wi_sum = sum(K[i,k] for k in Ni)
                wj_sum = sum(K[j,k] for k in Nj)
                mu_i = {k: (K[i,k]/wi_sum if wi_sum>0 else 0.0) for k in Ni}
                mu_j = {k: (K[j,k]/wj_sum if wj_sum>0 else 0.0) for k in Nj}
                common = set(mu_i.keys()).intersection(mu_j.keys())
                overlap = sum(min(mu_i[k], mu_j[k]) for k in common)
                # κ ≈ overlap (since d(i,j)=1 here)
                kOR = overlap
            edge_rows.append([i,j,w,kOR,"ollivier"])

    base = args.out or os.path.splitext(os.path.basename(args.kernel))[0]
    save_csv([["i","j","w","kappa","type"]]+edge_rows, header=None, path=f"{base}_curv_edges.csv")

    # node summaries (only Forman accumulated exactly above)
    if mode in ("forman","both"):
        node_rows=[["node","kappa_forman_node","deg","deg_w"]]
        for i in range(n):
            node_rows.append([i, node_val[i], len(neigh[i]), deg[i]])
        save_csv(node_rows, header=None, path=f"{base}_curv_nodes_forman.csv")

    # global summary
    summary = {"n": n, "edges": len(edges), "mode": mode,
               "quantile": args.quantile, "tau": args.tau, "kmax": args.kmax}
    save_json(summary, f"{base}_curv_summary.json")
    print(f"[OK] Curvature written to {base}_curv_edges.csv (+ nodes, summary)")

def connected_components(n, neigh):
    seen=[False]*n
    sizes=[]
    for s in range(n):
        if seen[s]: continue
        # BFS
        q=[s]; seen[s]=True; csize=0
        while q:
            u=q.pop()
            csize+=1
            for v in neigh[u]:
                if not seen[v]:
                    seen[v]=True; q.append(v)
        sizes.append(csize)
    return sizes

def count_triangles_approx(K, edges, neigh, max_per_node=200):
    # approximate triangle count by sampling neighbors per node
    tri=0
    for u in neigh:
        Nu = list(neigh[u])
        if len(Nu) > max_per_node:
            Nu = Nu[:max_per_node]
        Nu_set = set(Nu)
        for i, v in enumerate(Nu):
            for w in Nu[i+1:]:
                if w in neigh[v]:
                    tri += 1
    return tri

def cmd_homology(args):
    K = np.load(args.kernel)
    n = K.shape[0]
    # thresholds
    thr = []
    if args.tau_list:
        thr = [float(x) for x in args.tau_list.split(",")]
    elif args.quantiles:
        qs = [float(x) for x in args.quantiles.split(",")]
        w = np.triu(K,1)
        w = w[w>0]
        for q in qs:
            thr.append(float(np.quantile(_to_numpy(w), q)))
    else:
        # defaults: aggressive sparsification
        qs = [0.999, 0.998, 0.997, 0.995, 0.990]
        w = np.triu(K,1); w = w[w>0]
        for q in qs:
            thr.append(float(np.quantile(_to_numpy(w), q)))

    base = args.out or os.path.splitext(os.path.basename(args.kernel))[0]
    levels_rows=[["tau","n","m","cc","beta0","beta1","tri_approx"]]
    for tau in thr:
        edges, deg, neigh = kernel_to_sparse_edges(K, tau=tau)
        m = len(edges)
        cc_sizes = connected_components(n, neigh)
        cc = len(cc_sizes)
        beta0 = cc
        # For a simple graph (no 2-simplices filled), β1 = m - n + cc
        beta1 = m - n + cc
        tri = count_triangles_approx(K, edges, neigh, max_per_node=args.max_tri_sample)
        levels_rows.append([tau, n, m, cc, beta0, beta1, tri])
        print(f"tau={tau:.6g}  m={m}  cc={cc}  beta1={beta1}  tri≈{tri}")

    save_csv(levels_rows, header=None, path=f"{base}_homology_levels.csv")
    save_json({"levels": levels_rows[1:]}, f"{base}_homology_summary.json")
    print(f"[OK] Homology written to {base}_homology_levels.csv (+ summary)")

def cmd_fit_sdim_vs_n(args):
    # Accept multiple JSON paths or a glob
    files = args.files or []
    for g in args.glob or []:
        files.extend(glob.glob(g))
    # fallback: use *_CURRENT_smallt.json if present
    if not files:
        files = glob.glob("K_*_CURRENT_smallt.json")
    pts = []
    for f in files:
        # try to read n from filename 'K_(\d+)_...json'
        m = re.match(r"^K_(\d+)_.*\.json$", os.path.basename(f))
        n = int(m.group(1)) if m else None
        d = load_json(f)
        sdim = d.get("spectral_dimension_mean")
        se   = d.get("spectral_dimension_se") or 1e-6
        if n is not None and (sdim is not None):
            pts.append((n, sdim, se, f))
    # sort and (optionally) deduplicate same n
    pts.sort()
    # weighted LS for sdim(n) = d_inf + c * n^{-omega}
    # Take logs on n
    xs = np.array([math.log(p[0]) for p in pts], dtype=float)
    y  = np.array([p[1] for p in pts], dtype=float)
    w  = np.array([1.0/((p[2] or 1e-6)**2) for p in pts], dtype=float)
    # fit parameters: d_inf, c, omega
    # Nonlinear LS (simple grid over omega then linear for (d_inf, c))
    best=None
    for omega in [i/1000.0 for i in range(10, 801)]:  # 0.01 .. 0.8
        z = np.exp(-omega*xs)
        A = np.vstack([np.ones_like(z), z]).T
        W = np.diag(w)
        # solve weighted normal equations
        Aw = W @ A; yw = W @ y
        theta = np.linalg.lstsq(Aw, yw, rcond=None)[0]
        d_inf, c = theta[0], theta[1]
        yhat = A @ theta
        sse = float(((w*(y - yhat)**2)).sum())
        if (best is None) or (sse < best[0]):
            best = (sse, omega, d_inf, c)
    sse, omega, d_inf, c = best
    print("Best power-law fit (weighted):")
    print(f"  omega={omega:.3f},  d_inf={d_inf:.6f},  c={c:.6e},  weighted_SSE={sse:.6e}")
    print("Points used:")
    for n, sdim, se, f in pts:
        print(f"  n={n:5d}  sdim={sdim:.9f} ± {se:.9f}  <- {os.path.basename(f)}")
    out = args.out or "fit_sdim_vs_n"
    save_json({"omega":omega,"d_inf":d_inf,"c":c,"sse":sse,
               "points":[{"n":n,"sdim":sdim,"se":se,"file":f} for n,sdim,se,f in pts]}, f"{out}.json")

def cmd_export(args):
    files = []
    if args.glob:
        for g in args.glob:
            files.extend(glob.glob(g))
    files = sorted(set(files))
    rows=[["file","n","beta","rho_scale","a2","spin_mode","sdim","sdim_se","mds_stress","slope","intercept"]]
    for f in files:
        d = load_json(f)
        n=None; beta=None; rho=None; a2=None; spin=None
        # try to extract from JSON payload (kernel_file)
        kf = d.get("kernel_file","")
        mm = re.search(r"_b(\d+)p(\d+)\.npy$", kf)
        if mm: beta = float(f"{mm.group(1)}.{mm.group(2)}")
        mm = re.search(r"_rho(\d+)p(\d+)_", kf)
        if mm: rho = float(f"{mm.group(1)}.{mm.group(2)}")
        mm = re.search(r"_a2p(\d+)", kf)
        if mm: a2 = float(f"{mm.group(1)}")/100.0
        mm = re.match(r"^K_(\d+)_", os.path.basename(kf))
        if mm: n = int(mm.group(1))
        spin = re.search(r"_same_", kf) and "same" or (re.search(r"_opp_", kf) and "opp" or None)
        rows.append([
            os.path.basename(f),
            n, beta, rho, a2, spin,
            d.get("spectral_dimension_mean"),
            d.get("spectral_dimension_se"),
            d.get("mds_stress"),
            d.get("heat_meta",{}).get("slope"),
            d.get("heat_meta",{}).get("intercept"),
        ])
    out = args.out or "rtg_summary_master.csv"
    save_csv(rows, header=None, path=out)
    print(f"[OK] Wrote {out}")
    if args.xlsx:
        try:
            import pandas as pd
            df = pd.DataFrame(rows[1:], columns=rows[0])
            df.to_excel(args.xlsx, index=False)
            print(f"[OK] Wrote {args.xlsx}")
        except Exception as e:
            print(f"[WARN] Could not write XLSX ({e}). CSV is available at {out}.")

def cmd_chsh(args):
    """
    Very simple CHSH using attrs phases and a thresholded graph from K.
    Settings default (degrees): A0=0, A1=45, B0=22.5, B1=-22.5
    Outcome s(v,theta) = sign(cos(phi(v)-theta)).
    Correlation E(thetaA,thetaB) averaged over edges kept.
    """
    attrs = np.load(args.attrs)
    if "phi" not in attrs:
        raise SystemExit("attrs must contain 'phi' (angles in radians).")
    phi = attrs["phi"]  # assume radians
    K   = np.load(args.kernel)
    # sparsify
    edges, deg, neigh = kernel_to_sparse_edges(K, tau=args.tau, quantile=args.quantile, kmax=args.kmax)
    if not edges:
        raise SystemExit("No edges kept for CHSH; relax sparsification.")
    def sgn(x): return 1.0 if x>=0 else -1.0
    def outcome(theta):
        # vectorized
        return np.sign(np.cos(phi - theta))
    # settings
    deg2rad = math.pi/180.0
    A0 = (args.A0 or 0.0) * deg2rad
    A1 = (args.A1 or 45.0) * deg2rad
    B0 = (args.B0 or 22.5) * deg2rad
    B1 = (args.B1 or -22.5) * deg2rad
    sA0 = outcome(A0); sA1 = outcome(A1); sB0 = outcome(B0); sB1 = outcome(B1)
    def E(sX, sY):
        # average over edge pairs
        acc = 0.0
        for (i,j,w) in edges[:args.max_edges]:
            acc += float(sX[i])*float(sY[j])
        return acc / min(len(edges), args.max_edges)
    E00 = E(sA0, sB0)
    E01 = E(sA0, sB1)
    E10 = E(sA1, sB0)
    E11 = E(sA1, sB1)
    S = abs(E00 + E01 + E10 - E11)
    out = args.out or os.path.splitext(os.path.basename(args.kernel))[0] + "_chsh.json"
    save_json({"E00":E00,"E01":E01,"E10":E10,"E11":E11,"S":S,
               "n": int(K.shape[0]),
               "kept_edges": int(min(len(edges), args.max_edges)),
               "settings":{"A0":A0, "A1":A1, "B0":B0, "B1":B1}}, out)
    print(f"[OK] CHSH S={S:.6f}  (wrote {out})")


# --- EFT small-t fit ---------------------------------------------------------
def _eft_fit_one(json_file, tmin=None, tmax=None):
    import json, numpy as np
    d = json.load(open(json_file))
    t = np.array(d["heat_meta"]["t_grid"], dtype=float)
    P = np.array(d["heat_meta"]["heat_trace"], dtype=float)
    # small-t window selection
    if tmin is not None: mask = (t >= tmin)
    else: mask = np.ones_like(t, dtype=bool)
    if tmax is not None: mask &= (t <= tmax)
    t = t[mask]; P = P[mask]
    y = np.log(np.clip(P - 1.0, 1e-15, None))  # log(P-1)
    X = np.column_stack([np.ones_like(t), np.log(t), t])  # [1, ln t, t]
    # Weighted least squares (weights ~ 1 since SE on P small and uniform here)
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    c0, c1, c2 = beta  # y = c0 + c1 ln t + c2 t
    d_eff = 2.0 * c1
    m2_eff = -c2
    A = float(np.exp(c0))
    return {
        "json_file": json_file,
        "n": int(d.get("n", -1)),
        "d_eff": float(d_eff),
        "m2_eff": float(m2_eff),
        "A": A,
        "tmin": tmin, "tmax": tmax
    }

def cmd_eft(args):
    import glob, csv, json, os
    files = []
    for g in args.glob:
        files.extend(sorted(glob.glob(g)))
    if not files:
        print("[ERR] No files matched --glob")
        return 2
    rows = []
    for f in files:
        try:
            r = _eft_fit_one(f, tmin=args.tmin, tmax=args.tmax)
            rows.append(r)
        except Exception as e:
            print(f"[WARN] {f}: {e}")
    # Group by n; report median
    byn = {}
    for r in rows:
        byn.setdefault(r["n"], []).append(r)
    out = {"window":{"tmin":args.tmin,"tmax":args.tmax},"fits":[]}
    for n, L in sorted(byn.items()):
        d_med  = sorted(x["d_eff"] for x in L)[len(L)//2]
        m2_med = sorted(x["m2_eff"] for x in L)[len(L)//2]
        A_med  = sorted(x["A"] for x in L)[len(L)//2]
        out["fits"].append({"n":n,"d_eff_med":d_med,"m2_eff_med":m2_med,"A_med":A_med,"count":len(L)})
    # Write JSON
    jname = (args.out if args.out.endswith(".json") else args.out + ".json")
    with open(jname, "w") as f:
        json.dump(out, f, indent=2)
    # Write CSV
    cname = (args.out if args.out.endswith(".csv") else args.out + ".csv")
    with open(cname, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["n","d_eff_med","m2_eff_med","A_med","count","tmin","tmax"])
        for r in out["fits"]:
            w.writerow([r["n"], r["d_eff_med"], r["m2_eff_med"], r["A_med"], r["count"], args.tmin, args.tmax])
    print(f"[OK] EFT written to {jname} and {cname}")
    return 0

# ---- Argparse wiring --------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description="RTG full pipeline CLI")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("fss-beta", help="β fixed point from JSONs")
    s.add_argument("--glob", required=True, help="e.g. 'K_*_b*_smallt.json'")
    s.add_argument("--out", default=None)
    s.set_defaults(func=cmd_fss_beta)

    s = sub.add_parser("fss-rho", help="ρ sweep collation from JSONs")
    s.add_argument("--glob", required=True, help="e.g. 'K_*_rho*_smallt.json'")
    s.add_argument("--out", default=None)
    s.set_defaults(func=cmd_fss_rho)

    s = sub.add_parser("sdim-plateau", help="Instantaneous d_s(t) from one analyze JSON")
    s.add_argument("file")
    s.add_argument("--out", default=None)
    s.set_defaults(func=cmd_sdim_plateau)

    s = sub.add_parser("curvature", help="Forman/Ollivier curvature on sparse graph from K.npy")
    s.add_argument("--kernel", required=True)
    s.add_argument("--mode", choices=["forman","ollivier","both"], default="both")
    s.add_argument("--tau", type=float, default=None, help="absolute threshold")
    s.add_argument("--quantile", type=float, default=None, help="global weight quantile (0-1)")
    s.add_argument("--kmax", type=int, default=None, help="top-k neighbors per node")
    s.add_argument("--out", default=None)
    s.set_defaults(func=cmd_curvature)

    s = sub.add_parser("homology", help="Graph Betti0/Betti1 over thresholds")
    s.add_argument("--kernel", required=True)
    s.add_argument("--tau-list", default=None, help="comma sep absolute thresholds")
    s.add_argument("--quantiles", default=None, help="comma sep quantiles (0-1)")
    s.add_argument("--max-tri-sample", type=int, default=200)
    s.add_argument("--out", default=None)
    s.set_defaults(func=cmd_homology)

    s = sub.add_parser("fit-sdim-vs-n", help="Fit sdim(n) = d_inf + c*n^{-omega}")
    s.add_argument("--glob", nargs="*", default=None, help="one or more globs for JSON inputs")
    s.add_argument("files", nargs="*", help="explicit JSON files")
    s.add_argument("--out", default=None)
    s.set_defaults(func=cmd_fit_sdim_vs_n)

    s = sub.add_parser("export", help="Export table from analyze JSONs")
    s.add_argument("--glob", nargs="+", required=True, help="e.g. K_*_smallt.json K_*_tfine.json")
    s.add_argument("--out", default=None)
    s.add_argument("--xlsx", default=None)
    s.set_defaults(func=cmd_export)

    s = sub.add_parser("chsh", help="Simple CHSH on attrs+kernel")
    s.add_argument("--attrs", required=True, help="attrs_XXXX.npz with phi")
    s.add_argument("--kernel", required=True, help="K_XXXX_*.npy")
    s.add_argument("--tau", type=float, default=None)
    s.add_argument("--quantile", type=float, default=0.9995)
    s.add_argument("--kmax", type=int, default=None)
    s.add_argument("--A0", type=float, default=0.0)
    s.add_argument("--A1", type=float, default=45.0)
    s.add_argument("--B0", type=float, default=22.5)
    s.add_argument("--B1", type=float, default=-22.5)
    s.add_argument("--max-edges", type=int, default=200000)
    s.add_argument("--out", default=None)
    s.set_defaults(func=cmd_chsh)

    s = sub.add_parser("eft", help="Small-t EFT fit: P(t)-1 ≈ A t^{d/2} e^{-m^2 t} using existing *_smallt.json files")
    s.add_argument("--glob", nargs="+", required=True, help="glob(s) for *_smallt.json, e.g. 'K_*_smallt.json'")
    s.add_argument("--tmin", type=float, default=None, help="lower t bound (optional)")
    s.add_argument("--tmax", type=float, default=None, help="upper t bound (optional)")
    s.add_argument("--out",  required=True, help="output basename (.json & .csv)")
    s.set_defaults(func=cmd_eft)

    args = p.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()
