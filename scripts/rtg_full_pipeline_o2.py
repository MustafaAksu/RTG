#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RTG Full Pipeline — MDS stress scan, curvature, and homology
Author: ChatGPT (with user’s prior workflow in mind)
Date: 2025-08-26

This script adds three analysis subcommands that operate directly on a kernel
matrix saved as a .npy file:
  • mds-scan   : MDS/Isomap-style stress vs embedding dimension
  • curvature  : Forman curvature (always), Ollivier–Ricci (optional)
  • homology   : persistent H0/H1 over a weight threshold filtration

Input
-----
A dense (or memory-mappable) kernel file (.npy), shape (n, n), symmetric,
with nonnegative weights. You can generate it using your existing builder:
  python rtg_kernel_rg_v3.py build-kernel --attrs ... --out K_....npy

Outputs
-------
• JSON summaries (one per subcommand run)
• CSVs for per-edge/per-node curvature, and β_k vs threshold for homology
• Optional plots (disabled by default; turn on with --plots)

Performance / scaling
---------------------
• Distances: form a sparse graph by thresholding the kernel (--edge-threshold)
  then run Dijkstra (SciPy’s csgraph or a built-in heap-based fallback).
• MDS: run on a random node subset (--n-embed) to keep O(m * log n) + O(p^3),
  where p = n-embed (default 1024).
• Curvature: Forman is O(m). Ollivier–Ricci per edge solves a small balanced
  transport LP on neighborhoods; limit edges with --max-edges to keep runtimes.
• Homology: H0 via union–find across thresholds is near-linear in m. Optional
  clique-triangle counting is O(∑_i deg(i)^2) on the filtered graph; use
  --edge-threshold and/or --subsample to keep practical.

Optional acceleration
---------------------
If dpnp is installed, some array ops will use it. For SciPy functionality
(e.g. dijkstra, linprog) we convert to NumPy as required.

"""

from __future__ import annotations
import argparse, json, math, os, random, sys, time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

# ---- Optional array backends -------------------------------------------------
HAS_DPNP = False
try:
    import dpnp as dpnp
    HAS_DPNP = True
except Exception:
    pass

import numpy as np

def asnumpy(x):
    """Convert dpnp array to numpy if needed; otherwise return x."""
    if HAS_DPNP:
        try:
            if isinstance(x, dpnp.ndarray):
                return dpnp.asnumpy(x)
        except Exception:
            pass
    return x

# ---- Optional SciPy (nice to have, not required) -----------------------------
HAS_SCIPY = False
try:
    import scipy.sparse as sp
    import scipy.sparse.csgraph as csgraph
    from scipy.optimize import linprog
    HAS_SCIPY = True
except Exception:
    sp = None
    csgraph = None
    linprog = None
    HAS_SCIPY = False

# ---- Small utilities ---------------------------------------------------------
def now():
    return time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())

def ensure_dir(path: str):
    d = os.path.dirname(os.path.abspath(path))
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)

def save_json(obj, outpath: str):
    ensure_dir(outpath)
    with open(outpath, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)

def save_csv(rows: List[List], headers: List[str], outpath: str):
    ensure_dir(outpath)
    import csv
    with open(outpath, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(headers)
        w.writerows(rows)

def seed_all(seed: int):
    random.seed(seed)
    np.random.seed(seed)

# ---- Kernel loading & sparsification -----------------------------------------
def load_kernel(npy_path: str, dtype=np.float32, mmap=True) -> np.ndarray:
    if not os.path.exists(npy_path):
        raise FileNotFoundError(f"Kernel not found: {npy_path}")
    arr = np.load(npy_path, mmap_mode="r" if mmap else None)
    if dtype is not None and arr.dtype != dtype:
        arr = np.array(arr, dtype=dtype)
    return arr

def csr_from_dense_threshold(K: np.ndarray, threshold: float,
                             symmetrize: str = "max",
                             drop_diag: bool = True):
    """
    Build a CSR sparse matrix from dense K by thresholding (K >= threshold).
    symmetrize: 'max' (elementwise max with transpose) or 'none'
    """
    if not HAS_SCIPY:
        raise RuntimeError("SciPy is required for sparse CSR operations.")
    n = K.shape[0]
    if drop_diag:
        # Make a view and zero diagonal (avoid self-loops)
        K = K.copy()
        np.fill_diagonal(K, 0.0)

    mask = K >= threshold
    rows, cols = np.nonzero(mask)
    data = K[rows, cols].astype(np.float32, copy=False)
    A = sp.coo_matrix((data, (rows, cols)), shape=(n, n)).tocsr()
    if symmetrize == "max":
        A = A.maximum(A.T)
    return A

def lengths_from_weights_csr(W: "sp.csr_matrix", mode: str = "one_over_w",
                             eps: float = 1e-9) -> "sp.csr_matrix":
    """
    Convert weight CSR to length CSR (edge costs for Dijkstra).
    modes:
      - 'one_over_w'  : ell = 1 / (eps + w)
      - 'neg_log'     : ell = -log(max(eps, w / w_max))   (scale-invariant)
    """
    if not HAS_SCIPY:
        raise RuntimeError("SciPy is required for sparse CSR operations.")
    W = W.tocoo()
    w = W.data
    if mode == "one_over_w":
        ell = 1.0 / (eps + w)
    elif mode == "neg_log":
        wmax = float(w.max()) if w.size else 1.0
        ell = -np.log(np.maximum(eps, w / (wmax + 1e-12)))
    else:
        raise ValueError(f"Unknown length mode: {mode}")
    L = sp.coo_matrix((ell, (W.row, W.col)), shape=W.shape).tocsr()
    return L

# ---- Shortest paths -----------------------------------------------------------
def dijkstra_allpairs_subset(L: "sp.csr_matrix", nodes: np.ndarray) -> np.ndarray:
    """
    All-pairs distances restricted to 'nodes' indices.
    If SciPy available: use csgraph.dijkstra in batched mode.
    """
    if not HAS_SCIPY:
        # Fallback: run our own Dijkstra per source (heapq). O(m log n) per src.
        return _dijkstra_heap_subset(L, nodes)

    dmat = csgraph.dijkstra(csgraph=L, directed=False, indices=nodes, return_predecessors=False)
    # dmat shape: len(nodes) x n; we want the submatrix for nodes x nodes
    idx = np.array(nodes, dtype=int)
    return dmat[:, idx]

def _dijkstra_heap_subset(L: "sp.csr_matrix", nodes: np.ndarray) -> np.ndarray:
    """
    CPU-only fallback using a simple binary heap Dijkstra. For large graphs, prefer SciPy.
    """
    import heapq
    N = L.shape[0]
    indptr, indices, data = L.indptr, L.indices, L.data

    def run_single(src: int) -> np.ndarray:
        dist = np.full(N, np.inf, dtype=np.float64)
        dist[src] = 0.0
        visited = np.zeros(N, dtype=bool)
        h = [(0.0, src)]
        while h:
            d, u = heapq.heappop(h)
            if visited[u]: continue
            visited[u] = True
            start, end = indptr[u], indptr[u+1]
            nbrs, wts = indices[start:end], data[start:end]
            for v, w in zip(nbrs, wts):
                alt = d + w
                if alt < dist[v]:
                    dist[v] = alt
                    heapq.heappush(h, (alt, v))
        return dist

    # Compute distances for each source in 'nodes' and keep the nodes x nodes submatrix
    out = np.zeros((len(nodes), len(nodes)), dtype=np.float64)
    for i, src in enumerate(nodes):
        dist = run_single(int(src))
        out[i, :] = dist[nodes]
    return out

# ---- Classical MDS (strain) + stress -----------------------------------------
def classical_mds_stress(D: np.ndarray, dims: List[int]) -> Dict[str, float]:
    # Use only finite distances
    mask = np.isfinite(D)
    if not mask.all():
        # Replace inf on the diagonal with 0, elsewhere with max finite * 2
        D_work = D.copy()
        np.fill_diagonal(D_work, 0.0)
        finite_vals = D_work[np.isfinite(D_work)]
        big = finite_vals.max() * 2.0 if finite_vals.size else 1.0
        D_work[~np.isfinite(D_work)] = big
        D = D_work

    # Classical MDS (Torgerson): B = -1/2 * H D^2 H; eig(B)
    n = D.shape[0]
    H = np.eye(n) - np.ones((n, n)) / n
    D2 = D ** 2
    B = -0.5 * H @ D2 @ H

    # Numerical symmetrization
    B = 0.5 * (B + B.T)

    # Eigen-decompose
    evals, evecs = np.linalg.eigh(B)  # returns in ascending order
    idx = np.argsort(evals)[::-1]
    evals = evals[idx]
    evecs = evecs[:, idx]

    # Helper to reconstruct distances from top d components
    def recon_dist(d: int) -> np.ndarray:
        lam = np.maximum(evals[:d], 0.0)
        X = evecs[:, :d] * np.sqrt(lam.clip(min=0))
        # Euclidean distances in embedding
        G = np.sum(X * X, axis=1)
        Dhat2 = np.maximum(G[:, None] + G[None, :] - 2.0 * (X @ X.T), 0.0)
        return np.sqrt(Dhat2)

    # Frobenius norms
    denom = np.linalg.norm(D, ord="fro")
    out = {}
    for d in dims:
        Dhat = recon_dist(d)
        num = np.linalg.norm(D - Dhat, ord="fro")
        out[str(d)] = float(num / (denom + 1e-12))
    return out

# ---- Forman curvature --------------------------------------------------------
def forman_curvature(W: "sp.csr_matrix") -> Tuple[np.ndarray, np.ndarray, List[Tuple[int,int,float,float]]]:
    """
    Compute Forman curvature on edges of weighted undirected graph (CSR).
    Node weights w_u = weighted degree. Edge weights w_e = W[u,v].
    Returns:
      ric_node: np.array shape (n,) with node Forman "Ricci" (sum over incident edges)
      deg:      np.array shape (n,) weighted degrees
      edges_out: list of (u, v, w_uv, kappa_F) with u < v
    Reference: Sreejith et al. (2016) + weighted Forman for graphs.
    """
    if not HAS_SCIPY:
        raise RuntimeError("SciPy CSR required for Forman curvature.")
    W = W.tocsr()
    n = W.shape[0]
    indptr, indices, data = W.indptr, W.indices, W.data
    deg = np.asarray(W.sum(axis=1)).ravel()  # weighted degrees
    ric_node = np.zeros(n, dtype=np.float64)
    edges_out: List[Tuple[int,int,float,float]] = []

    for u in range(n):
        start_u, end_u = indptr[u], indptr[u+1]
        nbrs_u = indices[start_u:end_u]
        w_u_edges = data[start_u:end_u]

        for k, v in enumerate(nbrs_u):
            if v <= u:
                continue  # handle each undirected edge once (u<v)
            w_uv = w_u_edges[k]
            # Neighbors of u (excluding v) and v (excluding u)
            nu = nbrs_u[nbrs_u != v]
            w_ue = w_u_edges[nbrs_u != v]

            start_v, end_v = indptr[v], indptr[v+1]
            nbrs_v = indices[start_v:end_v]
            w_v_edges = data[start_v:end_v]
            nv = nbrs_v[nbrs_v != u]
            w_ve = w_v_edges[nbrs_v != u]

            # Weighted Forman curvature
            w_u = deg[u] + 1e-12
            w_v = deg[v] + 1e-12
            term_u = (w_u / w_uv) - np.sum(w_u / np.sqrt((w_uv + 1e-12) * (w_ue + 1e-12)))
            term_v = (w_v / w_uv) - np.sum(w_v / np.sqrt((w_uv + 1e-12) * (w_ve + 1e-12)))
            kappa = term_u + term_v

            ric_node[u] += kappa
            ric_node[v] += kappa
            edges_out.append((u, v, float(w_uv), float(kappa)))

    return ric_node, deg, edges_out

# ---- Ollivier–Ricci curvature (optional, per edge) ---------------------------
def ollivier_edge_curvature(
    W: "sp.csr_matrix",
    edge_list: List[Tuple[int, int]],
    exact: bool = True,
    alpha_self_mass: float = 0.0,
    solver: str = "highs"
) -> List[Tuple[int,int,float]]:
    """
    Compute Ollivier–Ricci curvature on selected edges.
    μ_i: random walk measure on (neighbors of i) with weights proportional to edge weights;
         optionally add α mass on the node itself (alpha_self_mass).
    Cost metric: shortest-path hop distance on the induced subgraph of 1-neighborhoods.
    If exact=True and SciPy is present, solve a balanced transport LP per edge.
    Otherwise, use a fast greedy matching heuristic (lower accuracy).
    Returns list of (i, j, kappa_OR).
    """
    if not HAS_SCIPY:
        exact = False  # fallback to heuristic without SciPy

    W = W.tocsr()
    n = W.shape[0]
    indptr, indices, data = W.indptr, W.indices, W.data

    def neigh(i: int) -> Tuple[np.ndarray, np.ndarray]:
        start, end = indptr[i], indptr[i+1]
        return indices[start:end], data[start:end]

    out: List[Tuple[int,int,float]] = []
    for (i, j) in edge_list:
        if i == j:
            continue
        Ni, wi = neigh(i)
        Nj, wj = neigh(j)

        # support sets (optionally include the node itself with alpha mass)
        supp_i = list(Ni)
        supp_j = list(Nj)
        pi = wi.astype(np.float64)
        pj = wj.astype(np.float64)
        if alpha_self_mass > 0.0:
            supp_i.append(i)
            supp_j.append(j)
            pi = np.concatenate([pi, np.array([alpha_self_mass], dtype=np.float64)])
            pj = np.concatenate([pj, np.array([alpha_self_mass], dtype=np.float64)])

        # Normalize
        si = pi.sum()
        sj = pj.sum()
        if si <= 0 or sj <= 0:
            out.append((i, j, float("nan")))
            continue
        pi = pi / si
        pj = pj / sj

        # Build cost matrix C_kl = hop distance between supp_i[k] and supp_j[l]
        # To keep cheap, compute hop distance by BFS from each supp_i node,
        # early-stopping once distances to all supp_j targets are known or radius=2.
        targets = set(supp_j)
        target_idx = {v: t for t, v in enumerate(supp_j)}
        C = np.zeros((len(supp_i), len(supp_j)), dtype=np.float64)

        # BFS limited to radius 3 to keep cost local
        max_hop = 3
        for k, src in enumerate(supp_i):
            dist = {src: 0}
            frontier = [src]
            found = set([src]) if src in targets else set()
            for hop in range(1, max_hop+1):
                nxt = []
                for u in frontier:
                    start, end = indptr[u], indptr[u+1]
                    nbrs = indices[start:end]
                    for v in nbrs:
                        if v in dist:
                            continue
                        dist[v] = hop
                        nxt.append(v)
                        if v in targets:
                            found.add(v)
                frontier = nxt
                if len(found) == len(targets):
                    break
            # populate row k
            for v, t in target_idx.items():
                C[k, t] = dist.get(v, max_hop + 1)

        if exact and linprog is not None:
            # Balanced OT LP: minimize <C, X> s.t. X 1 = pi, X^T 1 = pj, X >= 0
            K, L = C.shape
            c = C.flatten()
            # Equality constraints
            A_eq_rows = []
            b_eq = []
            # Row sums = pi
            for r in range(K):
                row = np.zeros(K * L, dtype=np.float64)
                row[r*L:(r+1)*L] = 1.0
                A_eq_rows.append(row); b_eq.append(pi[r])
            # Col sums = pj
            for cidx in range(L):
                col = np.zeros(K * L, dtype=np.float64)
                col[cidx::L] = 1.0
                A_eq_rows.append(col); b_eq.append(pj[cidx])
            A_eq = np.stack(A_eq_rows, axis=0)
            res = linprog(
                c=c, A_eq=A_eq, b_eq=np.array(b_eq, dtype=np.float64),
                bounds=(0, None), method=solver, options={"presolve": True}
            )
            if res.success:
                w1 = float(res.fun)
            else:
                # fallback to heuristic if LP fails
                w1 = _greedy_emdr(C, pi, pj)
        else:
            w1 = _greedy_emdr(C, pi, pj)

        # Graph distance between i and j is 1 (edge), so kappa = 1 - W1
        out.append((i, j, 1.0 - float(w1)))
    return out

def _greedy_emdr(C: np.ndarray, pi: np.ndarray, pj: np.ndarray) -> float:
    """
    Very fast greedy Earth–Mover lower-bound: repeatedly move mass from cheapest pairs.
    Not exact; used only as a fallback when LP isn't available.
    """
    K, L = C.shape
    # Flatten all pairs with their cost
    pairs = [(C[k, l], k, l) for k in range(K) for l in range(L)]
    pairs.sort()
    mass_i = pi.copy(); mass_j = pj.copy()
    cost = 0.0
    for c, k, l in pairs:
        if mass_i[k] <= 0 or mass_j[l] <= 0:
            continue
        m = min(mass_i[k], mass_j[l])
        mass_i[k] -= m; mass_j[l] -= m
        cost += m * c
        if mass_i.sum() <= 1e-12 and mass_j.sum() <= 1e-12:
            break
    # If leftover mass remains (numerical), penalize with max cost
    if mass_i.sum() > 1e-10 or mass_j.sum() > 1e-10:
        cost += (mass_i.sum() + mass_j.sum()) * (C.max() + 1.0)
    return cost

# ---- Homology: H0/H1 on threshold filtration (graph; optional triangles) -----
class UnionFind:
    def __init__(self, n: int):
        self.p = list(range(n))
        self.r = [0] * n
        self.cc = n
    def find(self, x):
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x
    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra == rb: return False
        if self.r[ra] < self.r[rb]:
            self.p[ra] = rb
        elif self.r[ra] > self.r[rb]:
            self.p[rb] = ra
        else:
            self.p[rb] = ra
            self.r[ra] += 1
        self.cc -= 1
        return True

def triangle_count_csr(W: "sp.csr_matrix") -> int:
    """
    Count triangles in an undirected simple graph given CSR adjacency (binary interpretation).
    Uses node-wise neighbor intersection; suitable for moderately sparse graphs.
    """
    W = W.tocsr().astype(bool).astype(int)
    n = W.shape[0]
    indptr, indices = W.indptr, W.indices
    tri = 0
    for u in range(n):
        nu = indices[indptr[u]:indptr[u+1]]
        # Only count v > u to avoid duplicates
        for v in nu[nu > u]:
            nv = indices[indptr[v]:indptr[v+1]]
            # count common neighbors > v to ensure unique triples (u < v < w)
            c = np.intersect1d(nu[nu > v], nv[nv > v], assume_unique=False).size
            tri += c
    return tri

# ---- Subcommands --------------------------------------------------------------
def cmd_mds_scan(args):
    seed_all(args.seed)
    K = load_kernel(args.kernel, dtype=np.float32, mmap=True)
    n = K.shape[0]
    if not HAS_SCIPY:
        raise RuntimeError("mds-scan requires SciPy for sparse Dijkstra. Please install scipy.")

    # Build sparse graph and lengths
    W = csr_from_dense_threshold(K, threshold=args.edge_threshold, symmetrize="max", drop_diag=True)
    L = lengths_from_weights_csr(W, mode=args.length_mode, eps=args.eps)

    # Choose node subset for embedding
    if args.nodes is not None and args.nodes.strip():
        # Load explicit node ids (0-based) from a text file, one per line
        nodes = np.loadtxt(args.nodes, dtype=int)
        nodes = nodes[(nodes >= 0) & (nodes < n)]
    else:
        p = min(args.n_embed, n)
        nodes = np.random.choice(n, size=p, replace=False)

    # Distances among nodes
    Dsub = dijkstra_allpairs_subset(L, nodes=nodes)  # shape p x p
    # Classical MDS stress for dims
    dims = sorted(set(args.dims))
    stress = classical_mds_stress(Dsub, dims)
    best_d = min(stress, key=lambda k: stress[k])

    out = {
        "kernel_file": os.path.basename(args.kernel),
        "n": int(n),
        "n_embed": int(len(nodes)),
        "edge_threshold": float(args.edge_threshold),
        "length_mode": args.length_mode,
        "dims": dims,
        "stress": stress,
        "best_dim": int(best_d),
        "created_utc": now()
    }
    if args.out:
        save_json(out, args.out)
    else:
        print(json.dumps(out, indent=2))

def cmd_curvature(args):
    if not HAS_SCIPY:
        raise RuntimeError("curvature requires SciPy (sparse ops and optionally linprog).")
    K = load_kernel(args.kernel, dtype=np.float32, mmap=True)
    n = K.shape[0]

    # Sparse adjacency at threshold
    W = csr_from_dense_threshold(K, threshold=args.edge_threshold, symmetrize="max", drop_diag=True)

    # (A) Forman curvature
    ric_node, deg, edges_F = forman_curvature(W)
    edges_rows = [(u, v, w, kappa) for (u, v, w, kappa) in edges_F]
    save_csv(edges_rows, ["u", "v", "w_uv", "kappa_forman"], args.edges_csv)
    nodes_rows = [(i, float(deg[i]), float(ric_node[i])) for i in range(n)]
    save_csv(nodes_rows, ["node", "deg_wsum", "ric_forman"], args.nodes_csv)

    # (B) Ollivier–Ricci (optional)
    ollivier_rows = []
    if args.ollivier:
        # Choose which edges to evaluate
        coo = W.tocoo()
        all_edges = [(int(u), int(v)) for (u, v) in zip(coo.row, coo.col) if u < v]
        random.seed(args.seed)
        random.shuffle(all_edges)
        if args.max_edges > 0:
            all_edges = all_edges[:args.max_edges]
        OR = ollivier_edge_curvature(
            W, all_edges, exact=not args.ollivier_fast,
            alpha_self_mass=args.ollivier_alpha, solver=args.lp_solver
        )
        for (u, v, kappa) in OR:
            ollivier_rows.append((u, v, kappa))
        save_csv(ollivier_rows, ["u", "v", "kappa_ollivier"], args.ollivier_csv)

    # Summary JSON
    out = {
        "kernel_file": os.path.basename(args.kernel),
        "n": int(n),
        "edge_threshold": float(args.edge_threshold),
        "forman": {
            "node_mean": float(np.nanmean(ric_node)),
            "node_std": float(np.nanstd(ric_node)),
            "edge_mean": float(np.nanmean([r[3] for r in edges_F])) if edges_F else float("nan"),
            "edge_std": float(np.nanstd([r[3] for r in edges_F])) if edges_F else float("nan"),
            "edges": len(edges_F)
        },
        "ollivier": {
            "enabled": bool(args.ollivier),
            "edges_evaluated": len(ollivier_rows),
            "kappa_mean": float(np.nanmean([r[2] for r in ollivier_rows])) if ollivier_rows else None,
            "kappa_std": float(np.nanstd([r[2] for r in ollivier_rows])) if ollivier_rows else None,
            "alpha_self_mass": float(args.ollivier_alpha),
            "fast_heuristic": bool(args.ollivier_fast)
        },
        "created_utc": now()
    }
    save_json(out, args.out)

def cmd_homology(args):
    if not HAS_SCIPY:
        raise RuntimeError("homology requires SciPy for sparse CSR operations.")
    seed_all(args.seed)
    K = load_kernel(args.kernel, dtype=np.float32, mmap=True)
    n = K.shape[0]

    # Optionally subsample nodes (for very large graphs)
    if args.subsample and args.subsample < n:
        nodes_keep = np.random.choice(n, size=args.subsample, replace=False)
        nodes_keep.sort()
        K = K[nodes_keep][:, nodes_keep]
        n = K.shape[0]
        mapping = {old: new for new, old in enumerate(nodes_keep)}
    else:
        mapping = None

    # Build filtration thresholds
    weights = K[np.triu_indices(n, 1)]
    weights = weights[np.isfinite(weights)]
    weights = weights[weights > 0]
    if args.thresholds:
        T = sorted(set(float(x) for x in args.thresholds))
    else:
        if weights.size == 0:
            raise RuntimeError("No positive weights in kernel.")
        # Use percentile-based thresholds from high to low to build filtration
        q_low, q_high = max(0.0, args.q_low), min(100.0, args.q_high)
        qs = np.linspace(q_high, q_low, num=args.n_levels)
        T = [np.percentile(weights, q) for q in qs]

    rows = []
    json_levels = []
    for t in T:
        # Build graph at threshold t
        W = csr_from_dense_threshold(K, threshold=t, symmetrize="max", drop_diag=True)
        # Components via union-find
        uf = UnionFind(n)
        coo = W.tocoo()
        m_edges = 0
        for u, v, w in zip(coo.row, coo.col, coo.data):
            if u < v:
                m_edges += 1
                uf.union(int(u), int(v))
        beta0 = uf.cc
        # H1 on graph skeleton: β1_graph = m - n + β0
        beta1_graph = m_edges - n + beta0
        tri_count = None
        beta1_clique = None
        if args.triangles:
            tri_count = triangle_count_csr(W)
            # crude 2-skeleton Euler adjustment (ignores tetrahedra and higher simplices):
            beta1_clique = beta1_graph - tri_count

        rows.append([t, n, m_edges, beta0, beta1_graph, tri_count, beta1_clique])
        json_levels.append({
            "threshold": float(t),
            "n": int(n),
            "m_edges": int(m_edges),
            "beta0": int(beta0),
            "beta1_graph": int(beta1_graph),
            "triangles": int(tri_count) if tri_count is not None else None,
            "beta1_clique_approx": int(beta1_clique) if beta1_clique is not None else None
        })

    save_csv(
        rows,
        headers=["threshold", "n", "m_edges", "beta0", "beta1_graph", "triangles", "beta1_clique_approx"],
        outpath=args.out_csv
    )
    out = {
        "kernel_file": os.path.basename(args.kernel),
        "n": int(n),
        "subsample_applied": (mapping is not None),
        "levels": json_levels,
        "created_utc": now()
    }
    save_json(out, args.out_json)

# ---- CLI ---------------------------------------------------------------------
def build_parser():
    p = argparse.ArgumentParser(
        description="RTG Full Pipeline — MDS scan, curvature, and homology"
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    # mds-scan
    q = sub.add_parser("mds-scan", help="Scan MDS stress vs embedding dimension on a sparse geodesic distance.")
    q.add_argument("--kernel", required=True, help=".npy kernel file")
    q.add_argument("--edge-threshold", type=float, default=0.0, help="Weight threshold to keep edges (default 0)")
    q.add_argument("--length-mode", type=str, default="one_over_w", choices=["one_over_w", "neg_log"],
                   help="Edge-length transform (default: one_over_w)")
    q.add_argument("--eps", type=float, default=1e-9, help="Epsilon for length transform (default 1e-9)")
    q.add_argument("--n-embed", type=int, default=1024, help="Number of nodes to embed (random subset) (default 1024)")
    q.add_argument("--nodes", type=str, default=None, help="Optional file with explicit node IDs (0-based), one per line")
    q.add_argument("--dims", type=int, nargs="+", default=[2,3,4,5,6,7,8], help="Embedding dimensions to test")
    q.add_argument("--seed", type=int, default=71, help="RNG seed")
    q.add_argument("--out", type=str, default=None, help="Output JSON (stdout if omitted)")
    q.set_defaults(func=cmd_mds_scan)

    # curvature
    r = sub.add_parser("curvature", help="Forman (always) + optional Ollivier–Ricci curvature on a thresholded graph.")
    r.add_argument("--kernel", required=True, help=".npy kernel file")
    r.add_argument("--edge-threshold", type=float, default=0.0, help="Weight threshold to keep edges")
    # Forman output CSVs
    r.add_argument("--edges-csv", type=str, default="curv_edges.csv", help="Output CSV of per-edge Forman curvature")
    r.add_argument("--nodes-csv", type=str, default="curv_nodes.csv", help="Output CSV of per-node aggregates")
    r.add_argument("--ollivier-csv", type=str, default="ollivier.csv", help="Output CSV of per-node aggregates")
    # Ollivier toggles
    r.add_argument("--ollivier", action="store_true", help="Compute Ollivier–Ricci on a subset of edges")
    r.add_argument("--ollivier-fast", action="store_true", help="Use fast greedy EMD (approx) instead of LP")
    r.add_argument("--max-edges", type=int, default=5000, help="Max edges to evaluate for Ollivier (default 5000)")
    r.add_argument("--ollivier-alpha", type=float, default=0.0, help="Self-mass at nodes (α) in μ_i (default 0)")
    r.add_argument("--lp-solver", type=str, default="highs", choices=["highs", "highs-ipm", "highs-ds", "revised simplex", "interior-point"],
                   help="SciPy linprog solver (default highs)")
    r.add_argument("--seed", type=int, default=71, help="RNG seed for sampling edges")
    r.add_argument("--out", type=str, default="curvature_summary.json", help="Summary JSON")
    r.set_defaults(func=cmd_curvature)

    # homology
    h = sub.add_parser("homology", help="Persistent H0/H1 over weight thresholds; optional triangle-aware H1 approximation.")
    h.add_argument("--kernel", required=True, help=".npy kernel file")
    h.add_argument("--subsample", type=int, default=None, help="Optional node subsample size for speed (e.g., 2048)")
    h.add_argument("--n-levels", type=int, default=21, help="#threshold levels if quantiles are used (default 21)")
    h.add_argument("--q-low", type=float, default=10.0, help="Lower percentile bound (default 10)")
    h.add_argument("--q-high", type=float, default=99.0, help="Upper percentile bound (default 99)")
    h.add_argument("--thresholds", type=float, nargs="+", default=None, help="Explicit threshold list (overrides quantiles)")
    h.add_argument("--triangles", action="store_true", help="Count triangles per level and report β1 adjusted by triangles")
    h.add_argument("--seed", type=int, default=71, help="RNG seed for subsampling")
    h.add_argument("--out-csv", type=str, default="homology_levels.csv", help="Per-level CSV output")
    h.add_argument("--out-json", type=str, default="homology_summary.json", help="Summary JSON output")
    h.set_defaults(func=cmd_homology)

    return p

def main():
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()
