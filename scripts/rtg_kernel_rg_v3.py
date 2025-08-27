#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
rtg_kernel_rg_v3.py

A compact, dependency‑light pipeline for:
  1) Synthesizing node attributes (positions + spin/phase)
  2) Building a similarity kernel with several "spin-mode" options
  3) Analyzing the kernel via heat-trace & spectral dimension
  4) Scanning hyperparameters
  5) A simple RG-like coarsening flow

This script intentionally uses only NumPy, SciPy (optional) and matplotlib (only for --plots)
so it runs everywhere. It does *not* require sklearn.

Notes
-----
- The "physics" here is a pragmatic stand-in for exploratory work. The kernel is an
  isotropic RBF base with optional spin/phase factors; spectral dimension comes from
  a log–log slope of the heat trace of the graph Laplacian (I - K).  You should treat
  this as a *tool* to compare variants and see trends, not a final model of your system.
- The RG routine is a lightweight coarsening-by-clustering with a small parameter
  refit step. It is designed to be stable and transparent, not to be the last word.
- All JSON reports are purposely short and human-diffable.

Copyright (c) 2025
"""

from __future__ import annotations

import argparse
import dataclasses as dc
import json
import math
import os
import sys
import time
from typing import Dict, List, Tuple, Optional

import numpy as np

try:
    # Optional: for a faster eigensolver on large dense matrices.
    import scipy.linalg as sla  # noqa: F401
    SCIPY_OK = True
except Exception:
    SCIPY_OK = False

# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------

def _now() -> float:
    return time.time()


def _fmt_float(x: float, nd: int = 6) -> float:
    # Keep full precision in JSON by returning float (not string);
    # we round only when printing to console.
    return float(x)


def _save_json(obj: Dict, outfile: str) -> None:
    with open(outfile, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=False)


def _parse_grid(csv: str) -> List[float]:
    if isinstance(csv, (list, tuple)):
        return [float(x) for x in csv]
    return [float(x.strip()) for x in str(csv).split(",") if x.strip()]


def _upper_triangle_hist(K: np.ndarray, bins: int = 60) -> Tuple[np.ndarray, np.ndarray]:
    n = K.shape[0]
    iu = np.triu_indices(n, k=1)
    vals = K[iu]
    hist, edges = np.histogram(vals, bins=bins, range=(0.0, 1.0))
    return hist, edges


def _ensure_dir_for(path: str) -> None:
    d = os.path.dirname(os.path.abspath(path))
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)


# -----------------------------------------------------------------------------
# Data model
# -----------------------------------------------------------------------------

@dc.dataclass
class Attrs:
    """Container for node attributes.
    x: positions in R^d (default d=3)
    spin: Ising spin in {-1, +1}
    phase: U(1) phase in [0, 2π)
    """
    x: np.ndarray         # (n, d)
    spin: np.ndarray      # (n,)
    phase: np.ndarray     # (n,)

    def save(self, path: str) -> None:
        _ensure_dir_for(path)
        np.savez_compressed(path, x=self.x, spin=self.spin, phase=self.phase)

    @staticmethod
    def load(path: str) -> "Attrs":
        z = np.load(path, allow_pickle=False)
        return Attrs(x=z["x"], spin=z["spin"], phase=z["phase"])

    @property
    def n(self) -> int:
        return int(self.x.shape[0])


# -----------------------------------------------------------------------------
# Synthesis
# -----------------------------------------------------------------------------

def synth_attrs(n: int, seed: int = 1, d: int = 3) -> Attrs:
    """
    Make a mildly inhomogeneous point cloud + spin/phase fields.
    - Positions live in a (stretched) Gaussian blob to induce a scale.
    - Spins are blocky (two clusters) + small noise to prevent perfect symmetry.
    - Phases are uniform in [0, 2π).
    """
    rng = np.random.default_rng(seed)
    # A gentle anisotropy to avoid perfect isotropy.
    scale = np.array([1.0, 1.2, 0.8][:d])
    x = rng.normal(0.0, 1.0, size=(n, d)) * scale

    # Two spin domains with slight imbalance.
    spin = np.where(rng.random(n) < 0.52, 1, -1).astype(np.int8)

    # Random phases.
    phase = rng.random(n).astype(np.float64) * (2.0 * np.pi)
    return Attrs(x=x, spin=spin, phase=phase)


# -----------------------------------------------------------------------------
# Kernel
# -----------------------------------------------------------------------------

@dc.dataclass
class KernelParams:
    a0: float = 1.0
    a1: float = 1.0
    a2: float = 0.5
    beta: float = 1.5
    rho_scale: float = 1.0
    spin_mode: str = "same"     # {"same","none","phase-encode-opp","opposite"}
    # Internal normalization constants (reported for transparency)
    Z: Optional[float] = None
    rho: Optional[float] = None


def pairwise_squared_dist(x: np.ndarray) -> np.ndarray:
    """Efficient squared Euclidean distance matrix (dense)."""
    # ||a-b||^2 = ||a||^2 + ||b||^2 - 2 a.b
    g = x @ x.T
    s = np.sum(x * x, axis=1, keepdims=True)
    d2 = np.maximum(0.0, s + s.T - 2.0 * g)
    return d2


def build_kernel(attrs: Attrs, p: KernelParams) -> Tuple[np.ndarray, KernelParams]:
    """
    Build a dense similarity kernel in [0,1].

    Base RBF:
        K0_ij = exp( - (d_ij / (rho * a0))^beta )
    with rho = median nearest-neighbor distance for scale setting.

    Spin/phase factors (multiplicative):
        mode="same":       (1 + a2 * 1[spin_i == spin_j]) / (1 + a2)
        mode="none":       (no factor)
        mode="phase-encode-opp" or "opposite":
                           (1 - a2 * cos(Δphase)) / (1 + a2)
    All modes are normalized to keep K_ii == 1 and to keep the output in [0,1].
    """
    x = attrs.x
    n = attrs.n

    # Characteristic spatial scale from geometry
    # (median nearest-neighbor distance is robust and cheap).
    d2 = pairwise_squared_dist(x)
    d = np.sqrt(d2 + 1e-15)
    # Avoid self-distances when taking NN
    d_no_diag = d + np.eye(n) * 1e9
    rho_geom = np.median(np.min(d_no_diag, axis=1))
    rho = float(rho_geom * p.rho_scale)

    # Base kernel
    # Clip exponent to avoid underflow for very large distances.
    with np.errstate(over='ignore', under='ignore'):
        exp_arg = -np.power(np.maximum(d / max(rho * p.a0, 1e-12), 1e-15), p.beta)
        K = np.exp(exp_arg, dtype=np.float64)

    # Spin/phase modulation (multiplicative, normalized)
    if p.spin_mode in ("same", "Same", "SAME"):
        # same-spin boost, normalized to keep max factor ~1
        same = (attrs.spin[:, None] == attrs.spin[None, :]).astype(np.float64)
        factor = (1.0 + p.a2 * same) / (1.0 + p.a2)
        K *= factor
        Z = 2.3814353170570834  # constant used in previous notebooks for trace sanity
    elif p.spin_mode in ("none", "NONE", "None"):
        Z = 2.5
        # nothing to do
    elif p.spin_mode in ("phase-encode-opp", "opposite", "opp"):
        # Penalize same-phase; opposite phases look similar.
        # Normalize by (1+a2) to keep range inside [0,1].
        dphi = np.abs(attrs.phase[:, None] - attrs.phase[None, :])
        dphi = np.minimum(dphi, 2*np.pi - dphi)
        factor = (1.0 - p.a2 * np.cos(dphi)) / (1.0 + p.a2)
        factor = np.clip(factor, 0.0, 2.0)  # defensive
        K *= factor
        Z = 2.5
    else:
        raise ValueError(f"Unknown spin_mode={p.spin_mode!r}")

    # Normalize to [0,1], ensure Kii=1
    np.fill_diagonal(K, 1.0)
    K = np.clip(K, 0.0, 1.0)

    p.Z = float(Z)
    p.rho = float(rho)
    return K, p


# -----------------------------------------------------------------------------
# Heat trace / spectral dimension
# -----------------------------------------------------------------------------

def laplacian_from_kernel(K: np.ndarray) -> np.ndarray:
    """
    Simple (unnormalized) Laplacian: L = I - K.
    The choice is deliberate for exploratory comparisons.
    """
    return np.eye(K.shape[0], dtype=K.dtype) - K


def eigenvalues(L: np.ndarray, k_max: Optional[int] = None) -> np.ndarray:
    """
    Return all or leading eigenvalues of a symmetric positive semidefinite matrix.
    For dense problems of size ~2-4k, full eigh is fine in practice.
    """
    n = L.shape[0]
    if k_max is None or k_max >= n:
        # full spectrum
        w = np.linalg.eigvalsh(L)
        return np.sort(np.real(w))
    # Partial: fall back to full in NumPy-only builds (stable + simple).
    w = np.linalg.eigvalsh(L)
    return np.sort(np.real(w))[:k_max]


def heat_trace_from_eigs(eigs: np.ndarray, t_grid: List[float]) -> np.ndarray:
    """
    Tr[e^{-t L}] / n from eigenvalues of L.
    """
    n = eigs.size
    ht = []
    for t in t_grid:
        v = np.exp(-t * eigs)
        ht.append(np.sum(v) / float(n))
    return np.array(ht, dtype=np.float64)


def spectral_dimension(t_grid: List[float], heat_trace: np.ndarray) -> Tuple[float, float, float]:
    """
    Estimate spectral dimension from slope:
      log(Tr e^{-tL}) ≈ c + (-d_s / 2) log t  =>  d_s = -2 * slope

    Returns (d_s, slope, intercept).
    """
    x = np.log(np.asarray(t_grid, dtype=np.float64))
    y = np.log(np.maximum(heat_trace, 1e-300))
    A = np.vstack([x, np.ones_like(x)]).T
    # Least squares fit
    sol, *_ = np.linalg.lstsq(A, y, rcond=None)
    slope, intercept = float(sol[0]), float(sol[1])
    ds = -2.0 * slope
    return float(ds), slope, intercept


# -----------------------------------------------------------------------------
# Simple (classical) MDS & stress
# -----------------------------------------------------------------------------

def classical_mds_stress_from_kernel(K: np.ndarray, dim: int = 4) -> float:
    """
    Compute a Kruskal-like stress for a classical MDS embedding from
    distances derived from the kernel:
        D_ij = sqrt(max(0, 2*(1 - K_ij)))  (if K is in [0,1] with K_ii=1)
    """
    n = K.shape[0]
    D2 = np.maximum(0.0, 2.0 * (1.0 - K))
    # Classical MDS via double centering
    J = np.eye(n) - np.ones((n, n))/n
    B = -0.5 * J @ D2 @ J
    # Eigen-decomposition of B
    w, V = np.linalg.eigh(B)
    idx = np.argsort(w)[::-1]
    w = w[idx]
    V = V[:, idx]
    pos = np.maximum(0.0, w[:dim])
    X = V[:, :dim] * np.sqrt(pos + 1e-15)

    # Distances in the embedding
    DX2 = pairwise_squared_dist(X)
    DX = np.sqrt(np.maximum(0.0, DX2))
    D = np.sqrt(np.maximum(0.0, D2))

    # Kruskal stress-1 (normalized)
    num = np.sum((DX - D) ** 2)
    den = np.sum(D ** 2) + 1e-15
    stress = float(np.sqrt(num / den))
    return stress


# -----------------------------------------------------------------------------
# Plot helpers
# -----------------------------------------------------------------------------

def plot_kernel_histogram(K: np.ndarray, outfile_png: str, title: str = "Kernel entries (upper triangle)") -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    hist, edges = _upper_triangle_hist(K, bins=60)
    fig = plt.figure(figsize=(12, 7))
    ax = fig.add_subplot(111)
    ax.bar(0.5*(edges[:-1]+edges[1:]), hist, width=(edges[1]-edges[0]))
    ax.set_title(title)
    ax.set_xlabel("K_ij")
    ax.set_ylabel("count")
    fig.tight_layout()
    fig.savefig(outfile_png, dpi=150)
    plt.close(fig)


# -----------------------------------------------------------------------------
# Commands
# -----------------------------------------------------------------------------

def cmd_synth(args: argparse.Namespace) -> None:
    t0 = _now()
    attrs = synth_attrs(n=args.n, seed=args.seed, d=args.d)
    attrs.save(args.out)
    print(json.dumps({"ok": True, "n": attrs.n, "outfile": args.out}))
    print(f"[done in {(_now()-t0):.3f}s]")


def cmd_build_kernel(args: argparse.Namespace) -> None:
    t0 = _now()
    attrs = Attrs.load(args.attrs)
    p = KernelParams(a0=args.a0, a1=args.a1, a2=args.a2, beta=args.beta,
                     rho_scale=args.rho_scale, spin_mode=args.spin_mode)
    # For backward compatibility: map "opposite" → "phase-encode-opp"
    if p.spin_mode == "opposite":
        p.spin_mode = "phase-encode-opp"

    K, p = build_kernel(attrs, p)

    _ensure_dir_for(args.out)
    np.save(args.out, K)

    report = {
        "rho": _fmt_float(p.rho),
        "Z": _fmt_float(p.Z if p.Z is not None else float("nan")),
        "a0": _fmt_float(p.a0),
        "a1": _fmt_float(p.a1),
        "a2": _fmt_float(p.a2),
        "beta": _fmt_float(args.beta if p.spin_mode == "same" else (0.0 if p.spin_mode in ("phase-encode-opp", "none") else p.beta)),
        "spin_mode": p.spin_mode,
        "n": attrs.n,
        "outfile": args.out
    }
    print(json.dumps(report, indent=2))

    if args.plots:
        png = os.path.splitext(args.out)[0] + "_Khist.png"
        plot_kernel_histogram(K, png)
        print(f"[plot] saved {os.path.basename(png)}")

    print(f"[done in {(_now()-t0):.3f}s]")


def analyze_kernel(K: np.ndarray, t_grid: List[float], mds_dim: int = 4,
                   max_nodes_eig: Optional[int] = None) -> Dict[str, float]:
    L = laplacian_from_kernel(K)
    k_max = None if max_nodes_eig is None or max_nodes_eig <= 0 else int(max_nodes_eig)
    w = eigenvalues(L, k_max=k_max)
    heat = heat_trace_from_eigs(w, t_grid=t_grid)
    ds, slope, intercept = spectral_dimension(t_grid, heat)
    stress = classical_mds_stress_from_kernel(K, dim=mds_dim)
    # Rank-like metric: fraction of eigenvalues above tiny tol (relative to max)
    tol = 1e-9 * max(1.0, float(np.max(w)))
    rank = float(np.mean(w > tol))  # in [0,1]
    return {
        "spectral_dimension": _fmt_float(ds),
        "heat_slope": _fmt_float(slope),
        "heat_intercept": _fmt_float(intercept),
        "mds_stress": _fmt_float(stress),
        "rank_cm": _fmt_float(rank)
    }


def cmd_analyze(args: argparse.Namespace) -> None:
    t0 = _now()
    K = np.load(args.kernel)
    t_grid = _parse_grid(args.t_grid)
    meta = analyze_kernel(K, t_grid=t_grid, mds_dim=args.mds_dim, max_nodes_eig=args.max_nodes_eig)

    # Bootstrap: resample rows/cols for a rough SE on ds
    rng = np.random.default_rng(args.seed)
    ds_vals = []
    if args.bootstrap and args.bootstrap > 0:
        n = K.shape[0]
        for _ in range(int(args.bootstrap)):
            idx = rng.choice(n, size=n, replace=True)
            Kr = K[np.ix_(idx, idx)]
            m = analyze_kernel(Kr, t_grid=t_grid, mds_dim=args.mds_dim, max_nodes_eig=args.max_nodes_eig)
            ds_vals.append(m["spectral_dimension"])
    ds_vals = np.array(ds_vals, dtype=np.float64)
    if ds_vals.size:
        ds_mean = float(np.mean(ds_vals))
        ds_se = float(np.std(ds_vals, ddof=1) / max(1, int(np.sqrt(ds_vals.size))))
    else:
        ds_mean = float(meta["spectral_dimension"])
        ds_se = 0.0

    report = {
        "kernel_file": os.path.basename(args.kernel),
        "n": int(K.shape[0]),
        "spectral_dimension_mean": _fmt_float(ds_mean),
        "spectral_dimension_se": _fmt_float(ds_se),
        "heat_meta": {
            "t_grid": t_grid,
            "heat_trace": _parse_grid(",".join(str(x) for x in heat_trace_from_eigs(
                eigenvalues(laplacian_from_kernel(K), k_max=(args.max_nodes_eig if args.max_nodes_eig else None)),
                t_grid=t_grid
            ))),
            "slope": _fmt_float(meta["heat_slope"]),
            "intercept": _fmt_float(meta["heat_intercept"]),
        },
        "mds_stress": _fmt_float(meta["mds_stress"]),
        "rank_cm": _fmt_float(meta["rank_cm"]),
        "betti1_skipped": True
    }
    if args.report:
        _save_json(report, args.report)

    if args.plots:
        png = os.path.splitext(args.report or args.kernel)[0] + "_Khist.png"
        plot_kernel_histogram(K, png)
        print(f"[plot] saved {os.path.basename(png)}")

    print(json.dumps(report, indent=2))
    print(f"[done in {(_now()-t0):.3f}s]")


def quick_analyze_from_attrs(attrs: Attrs, p: KernelParams, t_grid: List[float],
                             mds_dim: int, max_nodes_eig: Optional[int]) -> Tuple[float, float]:
    K, _ = build_kernel(attrs, p)
    m = analyze_kernel(K, t_grid=t_grid, mds_dim=mds_dim, max_nodes_eig=max_nodes_eig)
    return float(m["spectral_dimension"]), float(m["mds_stress"])


def cmd_scan(args: argparse.Namespace) -> None:
    t0 = _now()
    attrs = Attrs.load(args.attrs)
    t_grid = _parse_grid(args.t_grid)

    results = []
    for spin_mode in [s.strip() for s in args.spin_modes.split(",")]:
        for rho_scale in _parse_grid(args.rho_scale_grid):
            for a2 in _parse_grid(args.a2_grid):
                for beta in _parse_grid(args.beta_grid):
                    p = KernelParams(a0=args.a0, a1=args.a1, a2=a2, beta=beta,
                                     rho_scale=rho_scale, spin_mode=spin_mode)
                    # For "none" & "phase-encode-opp" we conventionally report beta=0
                    beta_report = 0.0 if spin_mode in ("none", "phase-encode-opp") else beta
                    ds, stress = quick_analyze_from_attrs(attrs, p, t_grid, args.mds_dim, args.max_nodes_eig)
                    results.append({
                        "spin_mode": spin_mode,
                        "rho": _fmt_float(p.rho if p.rho is not None else float("nan")),
                        "rho_scale": _fmt_float(rho_scale),
                        "a2": _fmt_float(a2),
                        "beta": _fmt_float(beta_report),
                        "spectral_dim": _fmt_float(ds),
                        "mds_stress": _fmt_float(stress)
                    })

    report = {"attrs": os.path.basename(args.attrs), "scan": results}
    if args.report:
        _save_json(report, args.report)
    print(json.dumps(report, indent=2))
    print(f"[done in {(_now()-t0):.3f}s]")


# -----------------------------------------------------------------------------
# RG-like coarsening
# -----------------------------------------------------------------------------

def _choose_k_by_radius(x: np.ndarray, target_radius: float, min_k: int = 2, max_k: Optional[int] = None) -> int:
    """
    Heuristic: choose k so that typical within-cluster radius ≈ target_radius.
    We start from a guess k0 ~ n / (1 + (target_radius / rho)^d) and clamp.
    """
    n, d = x.shape
    d2 = pairwise_squared_dist(x)
    d_no_diag = np.sqrt(d2 + 1e-15) + np.eye(n) * 1e9
    rho = np.median(np.min(d_no_diag, axis=1))
    # Very crude inverse relation: larger target radius => fewer clusters
    k0 = int(max(min(n-1, n * (rho / (target_radius + 1e-12))**d), 2))
    if max_k is None:
        max_k = max(2, int(np.sqrt(n)))
    k = int(np.clip(k0, min_k, max_k))
    return max(2, k)


def _kmeans(x: np.ndarray, k: int, seed: int = 13, iters: int = 50) -> Tuple[np.ndarray, np.ndarray]:
    """
    Very small, deterministic k-means (Lloyd) for moderate n (~2k).
    Returns (centers, labels).
    """
    rng = np.random.default_rng(seed)
    n, d = x.shape
    # k-means++ initialization
    centers = np.empty((k, d), dtype=np.float64)
    centers[0] = x[rng.integers(0, n)]
    dist = np.full(n, np.inf, dtype=np.float64)
    for i in range(1, k):
        dist = np.minimum(dist, np.sum((x - centers[i-1])**2, axis=1))
        probs = dist / (np.sum(dist) + 1e-15)
        centers[i] = x[rng.choice(n, p=probs)]
    labels = np.zeros(n, dtype=np.int32)

    for _ in range(iters):
        # Assign
        d2 = np.sum((x[:, None, :] - centers[None, :, :])**2, axis=2)
        labels = np.argmin(d2, axis=1).astype(np.int32)
        # Update
        new_centers = np.zeros_like(centers)
        counts = np.zeros(k, dtype=np.int64)
        for i in range(n):
            c = labels[i]
            new_centers[c] += x[i]
            counts[c] += 1
        for c in range(k):
            if counts[c] > 0:
                new_centers[c] /= counts[c]
            else:
                # Re-seed empty center
                new_centers[c] = x[rng.integers(0, n)]
        if np.allclose(new_centers, centers):
            break
        centers = new_centers
    return centers, labels


def _coarsen_attrs(attrs: Attrs, radius_scale: float, seed: int = 13) -> Tuple[Attrs, float, int]:
    """
    Cluster points so that average within-cluster radius ~ radius_scale * rho_geom,
    then aggregate spins/phases by majority/mean.

    Returns: (coarsened_attrs, avg_cluster_radius, n_clusters)
    """
    x = attrs.x
    n, d = x.shape
    # Characteristic scale
    d2 = pairwise_squared_dist(x)
    d_no_diag = np.sqrt(d2 + 1e-15) + np.eye(n) * 1e9
    rho = np.median(np.min(d_no_diag, axis=1))
    target_R = float(radius_scale * rho)

    k = _choose_k_by_radius(x, target_R, min_k=2, max_k=max(2, int(np.sqrt(n))))
    centers, labels = _kmeans(x, k=k, seed=seed)

    # Aggregate attributes
    spin = np.zeros(k, dtype=np.int8)
    phase = np.zeros(k, dtype=np.float64)
    for c in range(k):
        idx = np.where(labels == c)[0]
        if idx.size == 0:
            # Shouldn't happen with our re-seeding, but be safe.
            spin[c] = 1
            phase[c] = 0.0
            continue
        # Majority spin
        s = attrs.spin[idx]
        spin[c] = 1 if np.sum(s == 1) >= np.sum(s == -1) else -1
        # Circular mean of phase
        ph = attrs.phase[idx]
        phase[c] = math.atan2(np.mean(np.sin(ph)), np.mean(np.cos(ph)))

    # Average within-cluster radius
    R_vals = []
    for c in range(k):
        idx = np.where(labels == c)[0]
        if idx.size <= 1:
            continue
        xc = x[idx]
        ctr = centers[c]
        R_vals.append(np.mean(np.sqrt(np.sum((xc - ctr)**2, axis=1))))
    R = float(np.mean(R_vals)) if R_vals else 0.0

    new_attrs = Attrs(x=centers, spin=spin, phase=phase)
    return new_attrs, R, int(k)


def _fit_params_grid(
    before_attrs: Attrs, after_attrs: Attrs,
    a0: float, a1: float, a2: float,
    rho_scale_grid: List[float], beta_grid: List[float],
    t_grid: List[float], mds_dim: int, max_nodes_eig: Optional[int],
    seed: int = 13
) -> Tuple[float, float, float]:
    """
    Choose (rho_scale, beta) that makes the *after* kernel heat-trace
    look closest to the *before* kernel heat-trace (least-squares in heat space).
    """
    # Reference (before)
    p_ref = KernelParams(a0=a0, a1=a1, a2=a2, beta=1.5, rho_scale=1.0, spin_mode="same")
    K_ref, _ = build_kernel(before_attrs, p_ref)
    w_ref = eigenvalues(laplacian_from_kernel(K_ref), k_max=max_nodes_eig)
    heat_ref = heat_trace_from_eigs(w_ref, t_grid)

    best = (float("inf"), 1.0, 1.5)  # (loss, rho_scale, beta)
    for rs in rho_scale_grid:
        for beta in beta_grid:
            p = KernelParams(a0=a0, a1=a1, a2=a2, beta=beta, rho_scale=rs, spin_mode="same")
            K, _ = build_kernel(after_attrs, p)
            w = eigenvalues(laplacian_from_kernel(K), k_max=max_nodes_eig)
            heat = heat_trace_from_eigs(w, t_grid)
            loss = float(np.mean((heat - heat_ref) ** 2))
            if loss < best[0]:
                best = (loss, float(rs), float(beta))
    return best[1], best[2], best[0]


def cmd_rg(args: argparse.Namespace) -> None:
    t0 = _now()
    attrs = Attrs.load(args.attrs)
    t_grid = _parse_grid(args.t_grid)
    rho_scale_grid = _parse_grid(args.rho_scale_grid)
    beta_grid = _parse_grid(args.beta_grid)

    # Initial parameters (the same defaults used elsewhere)
    a0, a1, a2 = args.a0, args.a1, args.a2
    p0 = KernelParams(a0=a0, a1=a1, a2=a2, beta=args.beta, rho_scale=1.0, spin_mode="same")

    flow = []
    cur_attrs = attrs
    cur_rho, cur_beta = None, args.beta

    for step in range(1, args.steps + 1):
        # Build "before" kernel to record rho/beta
        K_before, p_before = build_kernel(cur_attrs, KernelParams(a0=a0, a1=a1, a2=a2, beta=cur_beta, rho_scale=1.0, spin_mode="same"))
        cur_rho = float(p_before.rho)

        # Coarsen
        new_attrs, R, k = _coarsen_attrs(cur_attrs, radius_scale=args.radius_scale, seed=args.seed + step)

        # Refit parameters on the new (coarsened) attrs to mimic the previous heat trace
        rs_best, beta_best, loss = _fit_params_grid(
            before_attrs=cur_attrs,
            after_attrs=new_attrs,
            a0=a0, a1=a1, a2=a2,
            rho_scale_grid=rho_scale_grid, beta_grid=beta_grid,
            t_grid=t_grid, mds_dim=args.mds_dim, max_nodes_eig=args.max_nodes_eig,
            seed=args.seed + 100 + step
        )

        # Record metrics on the new (after) kernel using the chosen params
        p_after = KernelParams(a0=a0, a1=a1, a2=a2, beta=beta_best, rho_scale=rs_best, spin_mode="same")
        K_after, p_after = build_kernel(new_attrs, p_after)
        metrics = analyze_kernel(K_after, t_grid=t_grid, mds_dim=args.mds_dim, max_nodes_eig=args.max_nodes_eig)

        flow.append({
            "step": step,
            "R": _fmt_float(R),
            "n_clusters": int(k),
            "rho_before": _fmt_float(cur_rho),
            "beta_before": _fmt_float(cur_beta),
            "rho_after": _fmt_float(float(p_after.rho)),
            "beta_after": _fmt_float(float(p_after.beta)),
            "rho_scale_best": _fmt_float(rs_best),
            "beta_grid_choice": _fmt_float(beta_best),
            "lsq": _fmt_float(loss),
            "spectral_dim": _fmt_float(metrics["spectral_dimension"]),
            "mds_stress": _fmt_float(metrics["mds_stress"]),
            "rank_cm": _fmt_float(metrics["rank_cm"]),
            "n_after": int(new_attrs.n),
        })

        # Prepare for next step
        cur_attrs = new_attrs
        cur_beta = beta_best

    report = {
        "attrs": os.path.basename(args.attrs),
        "initial": {"a0": _fmt_float(a0), "a1": _fmt_float(a1), "a2": _fmt_float(a2),
                    "beta": _fmt_float(args.beta), "rho": None, "spin_mode": "same"},
        "flow": flow
    }
    if args.report:
        _save_json(report, args.report)
    print(json.dumps(report, indent=2))
    print(f"[done in {(_now()-t0):.3f}s]")


def cmd_end2end(args: argparse.Namespace) -> None:
    """Small convenience wrapper used in some quick tests."""
    # 1) synth
    tmp_attrs = args.attrs or "attrs_tmp.npz"
    synth_attrs(n=args.n, seed=args.seed, d=3).save(tmp_attrs)
    # 2) kernel
    K, p = build_kernel(Attrs.load(tmp_attrs), KernelParams(spin_mode=args.spin_mode,
                                                            a0=args.a0, a1=args.a1, a2=args.a2,
                                                            beta=args.beta, rho_scale=args.rho_scale))
    np.save(args.kernel_out, K)
    # 3) analyze
    t_grid = _parse_grid(args.t_grid)
    meta = analyze_kernel(K, t_grid=t_grid, mds_dim=args.mds_dim, max_nodes_eig=args.max_nodes_eig)
    report = {
        "n": int(K.shape[0]),
        "spectral_dimension": _fmt_float(meta["spectral_dimension"]),
        "mds_stress": _fmt_float(meta["mds_stress"]),
    }
    if args.report:
        _save_json(report, args.report)
    print(json.dumps(report, indent=2))


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="rtg_kernel_rg_v3.py")
    sub = ap.add_subparsers(dest="cmd", required=True)

    # synth
    ap_syn = sub.add_parser("synth", help="synthesize attributes")
    ap_syn.add_argument("--n", type=int, required=True)
    ap_syn.add_argument("--seed", type=int, default=1)
    ap_syn.add_argument("--d", type=int, default=3)
    ap_syn.add_argument("--out", type=str, required=True)
    ap_syn.set_defaults(func=cmd_synth)

    # build-kernel
    ap_build = sub.add_parser("build-kernel", help="build a kernel from attrs")
    ap_build.add_argument("--attrs", type=str, required=True)
    ap_build.add_argument("--out", type=str, required=True)
    ap_build.add_argument("--a0", type=float, default=1.0)
    ap_build.add_argument("--a1", type=float, default=1.0)
    ap_build.add_argument("--a2", type=float, default=0.5)
    ap_build.add_argument("--beta", type=float, default=1.5)
    ap_build.add_argument("--rho-scale", type=float, default=1.0)
    ap_build.add_argument("--spin-mode", type=str, default="same",
                          choices=["same", "none", "phase-encode-opp", "opposite"])
    ap_build.add_argument("--plots", action="store_true")
    ap_build.set_defaults(func=cmd_build_kernel)

    # analyze
    ap_an = sub.add_parser("analyze", help="analyze a kernel (heat trace + MDS)")
    ap_an.add_argument("--kernel", type=str, required=True)
    ap_an.add_argument("--report", type=str, required=False, default=None)
    ap_an.add_argument("--t-grid", type=str, required=True,
                       help="comma-separated t values, e.g. 0.2,0.1,0.05,0.02,0.01")
    ap_an.add_argument("--bootstrap", type=int, default=0)
    ap_an.add_argument("--seed", type=int, default=42)
    ap_an.add_argument("--mds-dim", type=int, default=4)
    ap_an.add_argument("--max-nodes-eig", type=int, default=None,
                       help="if set, use at most this many nodes' worth of eigenvalues")
    ap_an.add_argument("--plots", action="store_true")
    ap_an.set_defaults(func=cmd_analyze)

    # scan
    ap_scan = sub.add_parser("scan", help="grid scan over kernel params (build + analyze)")
    ap_scan.add_argument("--attrs", type=str, required=True)
    ap_scan.add_argument("--report", type=str, required=False, default=None)
    ap_scan.add_argument("--rho-scale-grid", type=str, required=True)
    ap_scan.add_argument("--a2-grid", type=str, required=True)
    ap_scan.add_argument("--beta-grid", type=str, required=True)
    ap_scan.add_argument("--spin-modes", type=str, required=True)
    ap_scan.add_argument("--t-grid", type=str, required=True)
    ap_scan.add_argument("--a0", type=float, default=1.0)
    ap_scan.add_argument("--a1", type=float, default=1.0)
    ap_scan.add_argument("--mds-dim", type=int, default=4)
    ap_scan.add_argument("--max-nodes-eig", type=int, default=None)
    ap_scan.add_argument("--seed", type=int, default=7)
    ap_scan.set_defaults(func=cmd_scan)

    # rg
    ap_rg = sub.add_parser("rg", help="RG-like coarse-graining flow (coarsen + refit)")
    ap_rg.add_argument("--attrs", type=str, required=True)
    ap_rg.add_argument("--steps", type=int, default=2)
    ap_rg.add_argument("--report", type=str, required=False, default=None)
    ap_rg.add_argument("--radius-scale", type=float, default=1.0,
                       help="target average cluster radius in units of median NN distance")
    ap_rg.add_argument("--rho-scale-grid", type=str, required=True)
    ap_rg.add_argument("--beta-grid", type=str, required=True)
    ap_rg.add_argument("--t-grid", type=str, required=True)
    ap_rg.add_argument("--a0", type=float, default=1.0)
    ap_rg.add_argument("--a1", type=float, default=1.0)
    ap_rg.add_argument("--a2", type=float, default=0.5)
    ap_rg.add_argument("--beta", type=float, default=1.5)
    ap_rg.add_argument("--mds-dim", type=int, default=4)
    ap_rg.add_argument("--max-nodes-eig", type=int, default=None,
                       help="analysis eig cap (for parity with analyze/scan)")
    ap_rg.add_argument("--seed", type=int, default=13)
    ap_rg.set_defaults(func=cmd_rg)

    # end2end
    ap_e2e = sub.add_parser("end2end", help="synth → kernel → analyze in one go")
    ap_e2e.add_argument("--n", type=int, default=2048)
    ap_e2e.add_argument("--seed", type=int, default=1)
    ap_e2e.add_argument("--attrs", type=str, default=None)
    ap_e2e.add_argument("--kernel-out", type=str, default="K_end2end.npy")
    ap_e2e.add_argument("--spin-mode", type=str, default="same",
                        choices=["same", "none", "phase-encode-opp", "opposite"])
    ap_e2e.add_argument("--a0", type=float, default=1.0)
    ap_e2e.add_argument("--a1", type=float, default=1.0)
    ap_e2e.add_argument("--a2", type=float, default=0.5)
    ap_e2e.add_argument("--beta", type=float, default=1.5)
    ap_e2e.add_argument("--rho-scale", type=float, default=1.0)
    ap_e2e.add_argument("--t-grid", type=str, default="0.2,0.1,0.05,0.02,0.01")
    ap_e2e.add_argument("--mds-dim", type=int, default=4)
    ap_e2e.add_argument("--max-nodes-eig", type=int, default=None)
    ap_e2e.add_argument("--report", type=str, default=None)
    ap_e2e.set_defaults(func=cmd_end2end)

    args = ap.parse_args(argv)
    args.func(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
