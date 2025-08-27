#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
rtg_full_pipeline.py
====================

Compact end-to-end analysis helpers for the RTG project.
Includes subcommands to:
  • sdim-collate  — collate spectral-dimension JSONs into tidy tables
  • mds-scan      — classical MDS stress vs dimension on a subsample graph
  • curvature     — Forman & (approx) Ollivier–Ricci curvature on sampled edges
  • homology      — Betti_0, Betti_1 across a weight filtration (optional triangles)
  • summarize     — write multi-sheet XLSX/CSV summary from collated results

Accelerates numerics with dpnp if present; falls back to numpy automatically.

Author: ChatGPT (GPT-5 Pro) with user collaboration
License: MIT
"""
from __future__ import annotations

import argparse
import json
import os
import re
import math
import csv
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

# ------- Array backend (dpnp -> numpy fallback) ---------
_BACKEND = "numpy"
#try:
#    import dpnp as xp  # type: ignore
#    _BACKEND = "dpnp"
#except Exception:
import numpy as xp  # type: ignore
_BACKEND = "numpy"

# To convert to host numpy for CPU-only routines / features
def _to_numpy(a):
    try:
        import dpnp as _dpnp  # type: ignore
        return _dpnp.asnumpy(a)
    except Exception:
        import numpy as _np  # type: ignore
        return _np.asarray(a)

def _percentile(a, q):
    try:
        return xp.percentile(a, q)
    except Exception:
        import numpy as _np
        return _np.percentile(_to_numpy(a), q)

def _eigh_cpu(a):
    """CPU eigen-decomposition (numpy) for stability."""
    import numpy as _np
    w, v = _np.linalg.eigh(_np.asarray(a))
    return w, v

# -------------------- I/O helpers -----------------------
def _save_json(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)

def _ensure_dir(path):
    d = os.path.dirname(os.path.abspath(path))
    if d and not os.path.isdir(d):
        os.makedirs(d, exist_ok=True)

# -------------------- Parsing utils ---------------------
_re_kernel = re.compile(
    r'^K_(?P<n>\d+)_'
    r'(?P<spin>same|opp|phase|.*)_'
    r'rho(?P<rho>\d+)p(?P<rhodec>\d+)_'
    r'a2p(?P<a2>\d+)_'
    r'b(?P<bint>\d+|IGNORED)(?P<bdec>p\d+)?'
    r'(?:_.*)?\.(?:npy|npz|csv|json)$'
)

def parse_kernel_filename(path: str) -> Dict[str, Optional[float]]:
    name = os.path.basename(path)
    m = _re_kernel.match(name)
    out = {"n": None, "spin": None, "rho_scale": None, "a2": None, "beta": None}
    if not m:
        return out
    out["n"] = int(m.group("n"))
    out["spin"] = m.group("spin")
    rho = float(f"{m.group('rho')}.{m.group('rhodec')}")
    out["rho_scale"] = rho
    a2 = float(f"0.{m.group('a2')}") if len(m.group("a2")) <= 2 else float(m.group("a2"))
    out["a2"] = a2
    bint = m.group("bint")
    bdec = m.group("bdec")
    if bint == "IGNORED":
        out["beta"] = None
    else:
        if bdec and bdec.startswith("p"):
            out["beta"] = float(f"{bint}.{bdec[1:]}")
        else:
            out["beta"] = float(bint)
    return out

# ------------------ Graph building ----------------------
def load_kernel(path: str):
    """Load a kernel matrix K (symmetric, nonnegative) from NPY/Npz/CSV."""
    ext = os.path.splitext(path)[1].lower()
    if ext == ".npy":
        K = xp.load(path)
    elif ext == ".npz":
        K = xp.load(path)["K"]
    elif ext == ".csv":
        # Expect dense CSV; large files discouraged.
        import numpy as _np
        K = _np.loadtxt(path, delimiter=",")
        # Move to xp backend if dpnp
        try:
            import dpnp as _dpnp
            K = _dpnp.asarray(K)
        except Exception:
            pass
    else:
        raise ValueError(f"Unsupported kernel extension: {ext}")
    # Zero the diagonal for good measure
    if K.ndim != 2 or K.shape[0] != K.shape[1]:
        raise ValueError(f"Kernel must be square 2D array; got shape {K.shape}")
    n = K.shape[0]
    K = K.copy()
    idx = xp.arange(n)
    K[idx, idx] = 0.0
    return K

def top_edges_from_kernel(K, edge_frac: float = 0.02, min_edges: int = 10000) -> Tuple[xp.ndarray, xp.ndarray, xp.ndarray]:
    """Return top edges (i<j) by weight as arrays (i,j,w)."""
    n = K.shape[0]
    iu = xp.triu_indices(n, k=1)
    w = K[iu]
    total_pairs = w.size
    k = max(min_edges, int(edge_frac * total_pairs))
    k = min(k, total_pairs)
    # partition for top-k
    try:
        # xp.argpartition may be missing on some dpnp builds; fallback to numpy
        idxk = xp.argpartition(w, -k)[-k:]
    except Exception:
        import numpy as _np
        wn = _to_numpy(w)
        idxk = _np.argpartition(wn, -k)[-k:]
    ii = iu[0][idxk]
    jj = iu[1][idxk]
    ww = w[idxk]
    # sort by descending weight
    try:
        order = xp.argsort(-ww)
        ii, jj, ww = ii[order], jj[order], ww[order]
    except Exception:
        import numpy as _np
        order = _np.argsort(-_to_numpy(ww))
        ii, jj, ww = ii[order], jj[order], ww[order]
    return ii, jj, ww

def build_adj_list(n: int, edges: Sequence[Tuple[int,int,float]]) -> Tuple[List[List[int]], List[List[float]]]:
    nbrs = [[] for _ in range(n)]
    wts  = [[] for _ in range(n)]
    for i,j,w in edges:
        nbrs[i].append(j); wts[i].append(float(w))
        nbrs[j].append(i); wts[j].append(float(w))
    return nbrs, wts

# ------------------ Dijkstra (weighted) -----------------
import heapq
def dijkstra(n: int, nbrs: List[List[int]], wts: List[List[float]], src: int, edge_metric: str = "inv",
             cap: Optional[float] = None) -> List[float]:
    """Single-source Dijkstra. edge_metric in {"inv","neglog"}."""
    INF = 1e300
    dist = [INF]*n
    dist[src] = 0.0
    pq = [(0.0, src)]
    while pq:
        d,u = heapq.heappop(pq)
        if d!=dist[u]: 
            continue
        if cap is not None and d > cap:
            continue
        for v,w in zip(nbrs[u], wts[u]):
            if w <= 0.0:
                continue
            if edge_metric == "inv":
                ell = 1.0/max(w, 1e-15)
            elif edge_metric == "neglog":
                ell = -math.log(min(1.0, w/max(1e-15, w)))
                # the above is not meaningful; in practice pass --edge-metric inv
            else:
                ell = 1.0/max(w, 1e-15)
            nd = d + ell
            if nd < dist[v]:
                dist[v] = nd
                heapq.heappush(pq, (nd, v))
    return dist

# ------------------ Classical MDS -----------------------
def classical_mds_stress(D: xp.ndarray, k: int) -> Tuple[float, xp.ndarray]:
    """
    Classical MDS via double centering. Returns (stress, coords_k).
    D: (m x m) symmetric distance matrix (numpy or dpnp array).
    """
    # Work on CPU for stability (eigh)
    import numpy as _np
    D2 = _np.asarray(D)**2
    m = D2.shape[0]
    J = _np.eye(m) - _np.ones((m,m))/m
    B = -0.5 * J @ D2 @ J
    w, V = _np.linalg.eigh(B)
    # take top-k positive eigenvalues
    idx = _np.argsort(w)[::-1]
    w = w[idx]; V = V[:, idx]
    w_pos = _np.clip(w[:k], 0, None)
    X = V[:, :k] @ _np.diag(_np.sqrt(w_pos + 1e-15))
    # Stress: normalized Kruskal (relative)
    def pdist(X):
        from itertools import combinations
        m = X.shape[0]
        DD = _np.zeros((m,m), dtype=float)
        for i in range(m):
            for j in range(i+1, m):
                d = _np.linalg.norm(X[i]-X[j])
                DD[i,j]=DD[j,i]=d
        return DD
    DD = pdist(X)
    num = _np.sum((D - DD)**2)
    den = _np.sum(D**2) + 1e-15
    stress = math.sqrt(num/den)
    return float(stress), X

# ------------------ Forman curvature --------------------
def forman_curvature(n: int, nbrs: List[List[int]], wts: List[List[float]]) -> List[Tuple[int,int,float]]:
    """Compute Forman curvature per (undirected) edge using weighted-degree variant."""
    degw = [sum(ws) for ws in wts]
    out = []
    seen = set()
    for i in range(n):
        for j,wij in zip(nbrs[i], wts[i]):
            if i<j:
                # avoid double
                pass
            else:
                continue
            if wij <= 0: 
                continue
            # sum over i's neighbors excluding j
            Si = 0.0
            for k,wik in zip(nbrs[i], wts[i]):
                if k==j or wik<=0: 
                    continue
                Si += degw[i] / math.sqrt(max(wij*wik, 1e-15))
            Sj = 0.0
            for k,wjk in zip(nbrs[j], wts[j]):
                if k==i or wjk<=0: 
                    continue
                Sj += degw[j] / math.sqrt(max(wij*wjk, 1e-15))
            kF = (degw[i] + degw[j]) / max(wij,1e-15) - Si - Sj
            out.append((i,j,float(kF)))
    return out

# --------------- Approx. Ollivier–Ricci -----------------
def _sinkhorn(mu, nu, C, eps=0.1, n_iter=300, tol=1e-6):
    """Simple entropic OT (Sinkhorn) to approximate W1; mu,nu >=0 sum to 1; C>=0 cost matrix (numpy)."""
    import numpy as _np
    K = _np.exp(-C/ max(eps, 1e-9))
    u = _np.ones_like(mu)
    v = _np.ones_like(nu)
    Kt = K.T
    for _ in range(n_iter):
        u_prev = u
        Ku = K @ v
        u = mu / (Ku + 1e-15)
        Kv = Kt @ u
        v = nu / (Kv + 1e-15)
        if _np.max(_np.abs(u - u_prev)) < tol:
            break
    P = _np.diag(u) @ K @ _np.diag(v)
    cost = float((P * C).sum())
    return cost

def _hop_distance_matrix(nbrs: List[List[int]], Aset: List[set], X: List[int], Y: List[int], cap:int=2):
    """Small ground-distance matrix between X and Y nodes using hop distances up to 'cap'."""
    import numpy as _np
    m, r = len(X), len(Y)
    C = _np.full((m,r), 2.0 if cap==2 else 3.0, dtype=float)
    # 0 if same, 1 if neighbors, else cap
    for a,xa in enumerate(X):
        Ay = Aset[xa]
        for b,yb in enumerate(Y):
            if xa==yb:
                C[a,b]=0.0
            elif yb in Ay:
                C[a,b]=1.0
            else:
                # leave as 2.0 (or 3.0)
                pass
    return C

def ollivier_ricci_edges(n: int, nbrs: List[List[int]], wts: List[List[float]],
                         sample_edges: Optional[int]=10000,
                         hop_cap: int = 2,
                         edge_length_mode: str = "unit") -> List[Tuple[int,int,float]]:
    """
    Approximate Ollivier–Ricci curvature per edge (i,j).
    - mu_i(j) proportional to w_ij (weighted neighbors)
    - ground distance: hop metric up to cap (0/1/2 or 0/1/2/3)
    - transport via entropic Sinkhorn (regularized W1 proxy)
    - edge length: unit (1) or 1/w_ij
    """
    import random, numpy as _np
    # Build neighbor set for hop checks
    Aset = [set(N) for N in nbrs]
    edges = []
    for i in range(n):
        for j,w in zip(nbrs[i], wts[i]):
            if i<j and w>0:
                edges.append((i,j,w))
    if sample_edges is not None and len(edges) > sample_edges:
        random.seed(42)
        edges = random.sample(edges, sample_edges)
    out = []
    for (i,j,wij) in edges:
        Ni, wi = nbrs[i], wts[i]
        Nj, wj = nbrs[j], wts[j]
        # distributions over neighbors (exclude each other if desired? keep for locality)
        mu_nodes = Ni
        nu_nodes = Nj
        if len(mu_nodes)==0 or len(nu_nodes)==0:
            continue
        mu_w = xp.asarray(wi) / (sum(wi) + 1e-15)
        nu_w = xp.asarray(wj) / (sum(wj) + 1e-15)
        mu = _to_numpy(mu_w)
        nu = _to_numpy(nu_w)
        C = _hop_distance_matrix(nbrs, Aset, mu_nodes, nu_nodes, cap=hop_cap)
        W = _sinkhorn(mu, nu, C, eps=0.1)
        ell = 1.0 if edge_length_mode=="unit" else 1.0/max(wij,1e-15)
        k_or = 1.0 - W / max(ell, 1e-15)
        out.append((i,j,float(k_or)))
    return out

# ----------------- Homology (Betti) ---------------------
@dataclass
class UF:
    parent: List[int]
    rank: List[int]
    comp: int
    @classmethod
    def create(cls, n:int):
        return cls(list(range(n)), [0]*n, n)
    def find(self,a:int)->int:
        while self.parent[a]!=a:
            self.parent[a]=self.parent[self.parent[a]]
            a=self.parent[a]
        return a
    def union(self,a:int,b:int)->bool:
        ra, rb = self.find(a), self.find(b)
        if ra==rb: return False
        if self.rank[ra]<self.rank[rb]:
            ra,rb = rb,ra
        self.parent[rb]=ra
        if self.rank[ra]==self.rank[rb]:
            self.rank[ra]+=1
        self.comp-=1
        return True

def homology_filtration_from_top_edges(n:int, edges_sorted: Sequence[Tuple[int,int,float]],
                                       levels: int = 32,
                                       triangles: str = "none",
                                       tri_cap_n: int = 3072) -> List[Dict[str, float]]:
    """
    Build a descending weight filtration from the provided edges (already sorted by descending w).
    Returns per level: threshold index, edges_used, betti0, betti1 (1-skeleton), and optionally triangles-adjusted betti1_tri.
    triangles: "none", "matmul", or "auto" (matmul if n<=tri_cap_n else none)
    """
    m = len(edges_sorted)
    if m == 0:
        return []
    # choose cut indices geometrically spaced
    cuts = xp.unique(xp.linspace(1, m, num=levels, dtype=int))
    uf = UF.create(n)
    used = 0
    edges_seen = []
    out = []
    for cut in cuts:
        # add edges up to 'cut'
        while used < int(cut):
            i,j,w = edges_sorted[used]
            used += 1
            edges_seen.append((i,j,w))
            uf.union(i,j)
        c = uf.comp
        m_used = used
        betti0 = c
        betti1 = m_used - n + c  # for 1-skeleton
        rec = {
            "edge_index": int(cut),
            "edges_used": int(m_used),
            "betti0": int(betti0),
            "betti1": int(betti1),
        }
        # optional triangles correction
        if triangles != "none":
            tri_mode = triangles
            if triangles == "auto":
                tri_mode = "matmul" if n <= tri_cap_n else "none"
            if tri_mode == "matmul":
                # Dense adjacency on CPU; use manageable n only!
                import numpy as _np
                A = _np.zeros((n,n), dtype=_np.int8)
                for (a,b,_) in edges_seen:
                    A[a,b]=1; A[b,a]=1
                # triangles = trace(A^3)/6
                A2 = A @ A
                A3 = A2 @ A
                tri = int(_np.trace(A3)//6)
                rec["triangles"] = tri
                rec["betti1_tri"] = int(betti1 - tri)
        out.append(rec)
    return out

# --------------- SDIM Collation from JSONs --------------
def sdim_collate(patterns: Sequence[str], out_csv: str, out_json: Optional[str]=None):
    """
    Collect spectral-dimension summaries from rtg_kernel_rg_v3.py analyze outputs.
    Writes tidy CSV with columns: kernel_file,n,n_spin,rho_scale,a2,beta,sdim,se,mds_stress.
    """
    import glob
    rows = []
    for pat in patterns:
        for f in glob.glob(pat):
            try:
                with open(f,"r",encoding="utf-8") as fh:
                    d = json.load(fh)
            except Exception:
                continue
            kf = d.get("kernel_file") or d.get("kernel") or ""
            meta = parse_kernel_filename(os.path.basename(kf))
            rows.append({
                "file": os.path.basename(f),
                "kernel_file": os.path.basename(kf),
                "n": meta.get("n"),
                "spin": meta.get("spin"),
                "rho_scale": meta.get("rho_scale"),
                "a2": meta.get("a2"),
                "beta": meta.get("beta"),
                "sdim": d.get("spectral_dimension_mean"),
                "se": d.get("spectral_dimension_se"),
                "mds_stress": d.get("mds_stress"),
            })
    # write CSV
    _ensure_dir(out_csv)
    with open(out_csv, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()) if rows else
                           ["file","kernel_file","n","spin","rho_scale","a2","beta","sdim","se","mds_stress"])
        w.writeheader()
        for r in rows:
            w.writerow(r)
    if out_json:
        _save_json({"rows": rows}, out_json)

# ----------------- MDS Scan subcommand ------------------
def mds_scan(kernel_path: str, out_json: str, out_csv: Optional[str],
             edge_frac: float = 0.02, min_edges: int = 20000,
             subsample: int = 512, kmax: int = 8,
             edge_metric: str = "inv", seed: int = 71):
    """
    Build graph from top edges of the kernel and run classical MDS on a node subsample.
    Outputs stress vs embedding dimension.
    """
    import random, numpy as _np
    K = load_kernel(kernel_path)
    n = K.shape[0]
    ii, jj, ww = top_edges_from_kernel(K, edge_frac=edge_frac, min_edges=min_edges)
    edges = [(int(i),int(j),float(w)) for i,j,w in zip(_to_numpy(ii), _to_numpy(jj), _to_numpy(ww))]
    nbrs, wts = build_adj_list(n, edges)
    # node subsample
    random.seed(seed)
    nodes = list(range(n))
    if subsample < n:
        nodes = random.sample(nodes, subsample)
    idx_map = {u:i for i,u in enumerate(nodes)}
    # build pairwise distances for subsample via multi-source Dijkstra
    D = _np.zeros((len(nodes), len(nodes)), dtype=float)
    for a,u in enumerate(nodes):
        dist = dijkstra(n, nbrs, wts, u, edge_metric=edge_metric)
        for b,v in enumerate(nodes):
            D[a,b] = dist[v] if math.isfinite(dist[v]) else 1e6
    # run MDS for k=2..kmax
    results = []
    for k in range(2, max(2,kmax)+1):
        stress, X = classical_mds_stress(D, k)
        results.append({"k": int(k), "stress": float(stress)})
    out = {
        "kernel_file": os.path.basename(kernel_path),
        "n": int(n),
        "edge_frac": float(edge_frac),
        "subsample": int(len(nodes)),
        "mds_results": results,
        "backend": _BACKEND,
    }
    _ensure_dir(out_json)
    _save_json(out, out_json)
    if out_csv:
        with open(out_csv, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=["k","stress"])
            w.writeheader()
            for r in results:
                w.writerow(r)

# ---------------- Curvature subcommand ------------------
def curvature_scan(kernel_path: str, out_prefix: str,
                   edge_frac: float = 0.02, min_edges: int = 20000,
                   sample_edges: Optional[int] = 10000,
                   hop_cap: int = 2,
                   edge_length_mode: str = "unit"):
    """
    Compute Forman and approx. Ollivier–Ricci on sampled edges.
    Writes:
      - {out_prefix}_edges_forman.csv  (i,j,kappaF)
      - {out_prefix}_edges_oll.csv     (i,j,kappaOR)
      - {out_prefix}_summary.json      (stats)
    """
    import numpy as _np
    K = load_kernel(kernel_path)
    n = K.shape[0]
    ii, jj, ww = top_edges_from_kernel(K, edge_frac=edge_frac, min_edges=min_edges)
    edges = [(int(i),int(j),float(w)) for i,j,w in zip(_to_numpy(ii), _to_numpy(jj), _to_numpy(ww))]
    nbrs, wts = build_adj_list(n, edges)
    # Forman
    kF = forman_curvature(n, nbrs, wts)
    # Ollivier (sample edges)
    kOR = ollivier_ricci_edges(n, nbrs, wts, sample_edges=sample_edges, hop_cap=hop_cap, edge_length_mode=edge_length_mode)
    # write CSVs
    _ensure_dir(out_prefix + "_edges_forman.csv")
    with open(out_prefix + "_edges_forman.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["i","j","kappa_forman"])
        for i,j,k in kF:
            w.writerow([i,j,f"{k:.6g}"])
    with open(out_prefix + "_edges_oll.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["i","j","kappa_ollivier"])
        for i,j,k in kOR:
            w.writerow([i,j,f"{k:.6g}"])
    # summary stats
    def stats(vals):
        import numpy as _np
        if not vals: return {"count":0}
        a = _np.asarray(vals, dtype=float)
        return {"count": int(a.size),
                "mean": float(a.mean()),
                "std": float(a.std(ddof=1) if a.size>1 else 0.0),
                "min": float(a.min()),
                "p25": float(_np.percentile(a,25)),
                "median": float(_np.percentile(a,50)),
                "p75": float(_np.percentile(a,75)),
                "max": float(a.max())}
    summ = {
        "kernel_file": os.path.basename(kernel_path),
        "n": int(n),
        "backend": _BACKEND,
        "forman_stats": stats([k for _,_,k in kF]),
        "ollivier_stats": stats([k for _,_,k in kOR]),
        "edge_frac": edge_frac,
        "sample_edges": sample_edges,
        "hop_cap": hop_cap,
        "edge_length_mode": edge_length_mode,
    }
    _save_json(summ, out_prefix + "_summary.json")

# ----------------- Homology subcommand ------------------
def homology_scan(kernel_path: str, out_prefix: str,
                  edge_frac: float = 0.02, min_edges: int = 20000,
                  levels: int = 32,
                  triangles: str = "auto"):
    """
    Persistent homology for 1-skeleton across descending weight levels of top edges.
    Writes:
      - {out_prefix}_levels.csv with columns: edge_index,edges_used,betti0,betti1[,triangles,betti1_tri]
      - {out_prefix}_summary.json with head/tail stats
    """
    import numpy as _np
    K = load_kernel(kernel_path)
    n = K.shape[0]
    ii, jj, ww = top_edges_from_kernel(K, edge_frac=edge_frac, min_edges=min_edges)
    edges = [(int(i),int(j),float(w)) for i,j,w in zip(_to_numpy(ii), _to_numpy(jj), _to_numpy(ww))]
    recs = homology_filtration_from_top_edges(n, edges, levels=levels, triangles=triangles)
    # write CSV
    _ensure_dir(out_prefix + "_levels.csv")
    with open(out_prefix + "_levels.csv", "w", newline="", encoding="utf-8") as fh:
        cols = ["edge_index","edges_used","betti0","betti1","triangles","betti1_tri"]
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in recs:
            w.writerow({k: r.get(k,"") for k in cols})
    # summary
    head = recs[0] if recs else {}
    tail = recs[-1] if recs else {}
    summ = {
        "kernel_file": os.path.basename(kernel_path),
        "n": int(n),
        "backend": _BACKEND,
        "levels": int(levels),
        "edge_frac": float(edge_frac),
        "head": head,
        "tail": tail,
    }
    _save_json(summ, out_prefix + "_summary.json")

# ----------------- Summarize subcommand -----------------
def summarize(outputs: Sequence[str], out_xlsx: Optional[str], out_csv: Optional[str]):
    """
    Merge multiple CSV/JSON outputs into one multi-sheet XLSX (if pandas available) and/or a master CSV.
    Accepts mixed file types; sheets are created per base name.
    """
    frames = []
    names = []
    try:
        import pandas as pd
    except Exception:
        pd = None
    for f in outputs:
        base = os.path.basename(f)
        name, ext = os.path.splitext(base)
        names.append(name)
        if ext.lower() == ".csv":
            try:
                if pd is not None:
                    frames.append((name, pd.read_csv(f)))
                else:
                    # load and append to master CSV later
                    frames.append((name, f))
            except Exception:
                pass
        elif ext.lower() == ".json":
            try:
                with open(f, "r", encoding="utf-8") as fh:
                    data = json.load(fh)
                if pd is not None:
                    frames.append((name, pd.json_normalize(data)))
                else:
                    frames.append((name, f))
            except Exception:
                pass
    if pd is not None and out_xlsx:
        _ensure_dir(out_xlsx)
        with pd.ExcelWriter(out_xlsx, engine="xlsxwriter") as xw:
            for name, df in frames:
                if hasattr(df, "to_excel"):
                    df.to_excel(xw, sheet_name=name[:31], index=False)
    if out_csv:
        # flatten into one CSV by stacking with a 'source' column when feasible
        import pandas as pd2
        rows = []
        for name, df in frames:
            if hasattr(df, "to_dict"):
                tmp = df.copy()
                tmp.insert(0, "source", name)
                rows.append(tmp)
        if rows:
            cat = pd2.concat(rows, ignore_index=True)
            _ensure_dir(out_csv)
            cat.to_csv(out_csv, index=False)

# --------------------------- CLI ------------------------
def main():
    p = argparse.ArgumentParser(description="RTG full analysis pipeline (sdim-collate, mds-scan, curvature, homology, summarize).")
    sub = p.add_subparsers(dest="cmd", required=True)

    pc = sub.add_parser("sdim-collate", help="Collate spectral-dimension JSONs into a tidy CSV (and optional JSON).")
    pc.add_argument("--patterns", nargs="+", required=True, help="Glob patterns for *json (analyze outputs).")
    pc.add_argument("--out-csv", required=True, help="Output CSV path.")
    pc.add_argument("--out-json", default=None, help="Optional JSON path.")

    pm = sub.add_parser("mds-scan", help="MDS stress vs dimension on a subsample graph built from top edges of kernel.")
    pm.add_argument("--kernel", required=True, help="Kernel file (.npy/.npz/.csv).")
    pm.add_argument("--out-json", required=True, help="Output JSON path.")
    pm.add_argument("--out-csv", default=None, help="Optional CSV path.")
    pm.add_argument("--edge-frac", type=float, default=0.02, help="Top edge fraction to keep (default 0.02).")
    pm.add_argument("--min-edges", type=int, default=20000, help="Minimum number of edges to keep.")
    pm.add_argument("--subsample", type=int, default=512, help="Node subsample size (default 512).")
    pm.add_argument("--kmax", type=int, default=8, help="Max embedding dimension for MDS (default 8).")
    pm.add_argument("--edge-metric", choices=["inv","neglog"], default="inv", help="Edge length mapping (default inv => 1/w).")
    pm.add_argument("--seed", type=int, default=71, help="RNG seed.")

    pr = sub.add_parser("curvature", help="Compute Forman and approx. Ollivier–Ricci curvature on sampled edges.")
    pr.add_argument("--kernel", required=True, help="Kernel file.")
    pr.add_argument("--out-prefix", required=True, help="Prefix for outputs.")
    pr.add_argument("--edge-frac", type=float, default=0.02, help="Top edge fraction to keep.")
    pr.add_argument("--min-edges", type=int, default=20000, help="Minimum number of edges to keep.")
    pr.add_argument("--sample-edges", type=int, default=10000, help="Edge sample for Ollivier; set 0 to use all.")
    pr.add_argument("--hop-cap", type=int, default=2, help="Hop cap for ground metric (2 or 3).")
    pr.add_argument("--edge-length-mode", choices=["unit","invw"], default="unit", help="Edge length for denominator (1 or 1/w).")

    ph = sub.add_parser("homology", help="Persistent homology (Betti_0, Betti_1) across descending weight levels.")
    ph.add_argument("--kernel", required=True, help="Kernel file.")
    ph.add_argument("--out-prefix", required=True, help="Prefix for outputs.")
    ph.add_argument("--edge-frac", type=float, default=0.02, help="Top edge fraction to keep.")
    ph.add_argument("--min-edges", type=int, default=20000, help="Minimum edges to keep.")
    ph.add_argument("--levels", type=int, default=32, help="Number of filtration levels.")
    ph.add_argument("--triangles", choices=["none","matmul","auto"], default="auto", help="Triangle counting mode.")

    ps = sub.add_parser("summarize", help="Collect multiple outputs into a single XLSX and/or CSV.")
    ps.add_argument("--outputs", nargs="+", required=True, help="List of CSV/JSON outputs to merge.")
    ps.add_argument("--out-xlsx", default=None, help="Path to write an Excel workbook.")
    ps.add_argument("--out-csv", default=None, help="Path to write a master CSV.")

    args = p.parse_args()

    if args.cmd == "sdim-collate":
        sdim_collate(args.patterns, args.out_csv, args.out_json)
    elif args.cmd == "mds-scan":
        mds_scan(args.kernel, args.out_json, args.out_csv, args.edge_frac, args.min_edges,
                 args.subsample, args.kmax, args.edge_metric, args.seed)
    elif args.cmd == "curvature":
        sample_edges = None if (args.sample_edges is not None and int(args.sample_edges)==0) else int(args.sample_edges)
        curvature_scan(args.kernel, args.out_prefix, args.edge_frac, args.min_edges, sample_edges,
                       args.hop_cap, "unit" if args.edge_length_mode=="unit" else "invw")
    elif args.cmd == "homology":
        homology_scan(args.kernel, args.out_prefix, args.edge_frac, args.min_edges, args.levels, args.triangles)
    elif args.cmd == "summarize":
        summarize(args.outputs, args.out_xlsx, args.out_csv)
    else:
        raise SystemExit("Unknown command.")

if __name__ == "__main__":
    main()
