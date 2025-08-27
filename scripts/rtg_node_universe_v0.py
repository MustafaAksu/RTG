#!/usr/bin/env python3
"""
rtg_node_universe_v0.py — minimal RTG model (node gas + CHSH)
──────────────────────────────────────────────────────────────
• 3 "proton" nodes (ω ≈ 0, φ in shell-3 symmetry, s = [+1, -1, +1])
• Nb background nodes (default 300), ω in ±0.05 Δω*
• Canonical RTG resonance kernel + over‑damped Langevin dynamics
• Decoherence (σ) applied as ω-noise on node 1
• Live observables: CHSH, Δφ₁₂₃ (phase shape), flip rate, energy
"""

import argparse, math, time
import numpy as np

# ─── CLI ──────────────────────────────────────────────
ap = argparse.ArgumentParser()
ap.add_argument('--Nb', type=int, default=300)
ap.add_argument('--steps', type=int, default=10000)
ap.add_argument('--dt', type=float, default=1e-4)
ap.add_argument('--T', type=float, default=0.02)
ap.add_argument('--gamma', type=float, default=0.1)
ap.add_argument('--sigma', type=float, default=0.00)
ap.add_argument('--seed',  type=int, default=1234)
args = ap.parse_args()
rng = np.random.default_rng(args.seed)

# ─── RTG constants ───────────────────────────────────
Δω_star = 1.0
J, K = 1.0, 3.0
N_p, N_bg = 3, args.Nb
N_tot = N_p + N_bg
σ_abs = args.sigma * Δω_star

# ─── Node initialization ─────────────────────────────
omega = rng.uniform(-0.05, 0.05, size=N_tot)
phi   = rng.uniform(0, 2*np.pi, size=N_tot)
spin  = np.ones(N_tot, dtype=int)

# Proton shell (near-zero ω, shell-3 phases, anti-aligned spins)
omega[0] = omega[1] = 0.0
omega[2] = 0.28 * Δω_star  # optional split
phi[:3] = [0.0, 2*np.pi/3, 4*np.pi/3]
spin[0], spin[1], spin[2] = +1, -1, +1

# ─── Pair indices (upper triangle) ───────────────────
pairs = np.array([(i, j) for i in range(N_tot) for j in range(i+1, N_tot)], dtype=int)
ii, jj = pairs[:, 0], pairs[:, 1]

# ─── Resonance & Energy ──────────────────────────────
def resonance(ii, jj):
    dω = omega[ii] - omega[jj]
    gauss = np.exp(-(dω / Δω_star)**2)
    return 0.75 * (1 + np.cos(phi[ii] - phi[jj])) * (1 + spin[ii]*spin[jj]) * gauss

def energy():
    R = resonance(ii, jj)
    return (K * np.abs(omega[ii] - omega[jj]) + J * R).sum() / N_tot

# ─── CHSH Estimator (cosine-based) ───────────────────
angles = [0, np.pi/2, np.pi, 3*np.pi/2]

def E_corr(th_a, th_b):
    return np.cos(phi[0] - th_a) * np.cos(phi[1] - th_b) * spin[0] * spin[1]

def chsh_pair():
    A, B, C, D = angles
    return E_corr(A, C) + E_corr(B, C) + E_corr(A, D) - E_corr(B, D)

# ─── Phase Shape (Δφ₁₂₃) ─────────────────────────────
def proton_delta_phi():
    return ((phi[0]-phi[1]) + (phi[1]-phi[2]) + (phi[2]-phi[0])) % (2*np.pi)

# ─── Main Loop ───────────────────────────────────────
print("#  step   CHSH    Δφ123   flip_rate     E/N")
start = time.time()
obs_int = max(1, args.steps // 100)
total_flips = 0

for t in range(1, args.steps+1):
    # forces
    R = resonance(ii, jj)
    dω = omega[ii] - omega[jj]
    dφ = phi[ii] - phi[jj]
    gauss = np.exp(-(dω / Δω_star)**2)

    dR_dω = -2 * gauss * (dω / Δω_star**2) * 0.75 * (1 + np.cos(dφ)) * (1 + spin[ii]*spin[jj])
    Fω = np.zeros(N_tot)
    np.add.at(Fω, ii,  K * np.sign(dω) + J * dR_dω)
    np.add.at(Fω, jj, -K * np.sign(dω) - J * dR_dω)

    dR_dφ = -0.75 * (1 + spin[ii]*spin[jj]) * gauss * np.sin(dφ)
    Fφ = np.zeros(N_tot)
    np.add.at(Fφ, ii,  J * dR_dφ)
    np.add.at(Fφ, jj, -J * dR_dφ)

    # Langevin step (overdamped)
    omega += args.dt * (-Fω) + np.sqrt(2 * args.T * args.dt) * rng.normal(0, 1, N_tot)
    phi   += args.dt * (-Fφ) + np.sqrt(2 * args.T * args.dt) * rng.normal(0, 1, N_tot)
    phi %= 2 * np.pi

    if σ_abs > 0:
        omega[1] += rng.normal(0, σ_abs)

    # Background-only spin flips
    background = np.arange(3, N_tot)
    candidates = rng.choice(background, size=len(background)//8, replace=False)
    spin[candidates] *= -1
    total_flips += len(candidates)

    # Diagnostics
    if t % obs_int == 0 or t == 1:
        chsh = chsh_pair()
        dphi = proton_delta_phi()
        rate = total_flips / (N_tot * t)
        en   = energy()
        print(f"{t:7d}  {chsh:7.3f}  {dphi:7.3f}   {rate:7.4f}   {en:9.4f}")

elapsed = time.time()
print(f"\nDone: {args.steps} steps in {elapsed - start:.1f}s  (σ = {args.sigma} Δω*)")
