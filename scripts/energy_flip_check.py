#!/usr/bin/env python
"""
Check ΔE when we flip a single spin in an RTG snapshot.
Usage:  python energy_flip_check.py cfg_md/snap_002000.npz --site 12345
"""
import argparse, numpy as np, math, yaml
from pathlib import Path

# ---------- CLI ----------
ap = argparse.ArgumentParser()
ap.add_argument('npz', help='snapshot file')
ap.add_argument('--site', type=int, default=0, help='lattice index to flip')
ap.add_argument('--delta_omega_star', type=float, default=1.0)
args = ap.parse_args()

dat   = np.load(args.npz)
ω, φ, s = dat['omega'], dat['phi'], dat['spin']

# ---------- constants ----------
this_dir = Path(__file__).resolve().parent
yaml_path = this_dir.parents[1] / 'RTG/src/rtg/recursion_v2.yaml'
K_over_J = float(yaml.safe_load(yaml_path.read_text())['K_over_J'])
J, K   = 1.0, K_over_J
delta_omega_star    = args.delta_omega_star

# ---------- neighbour list (static cubic) ----------
def nn_pairs(L):
    idx = np.arange(L**3).reshape(L, L, L)
    out = []
    for x in range(L):
        for y in range(L):
            for z in range(L):
                i = idx[x, y, z]
                out += [(i, idx[(x+1)%L, y, z]),
                        (i, idx[x,(y+1)%L, z]),
                        (i, idx[x, y,(z+1)%L])]
    return np.asarray(out, np.int32)

L = round(len(ω)**(1/3))
pairs = nn_pairs(L)
i_idx, j_idx = pairs.T

def R(i,j):
    cos   = np.cos(φ[i]-φ[j])
    spin  = 1 + s[i]*s[j]
    gauss = np.exp(-((ω[i]-ω[j])/delta_omega_star)**2)
    return 0.75*(1+cos)*spin*gauss

def total_E():
    kin = 0.0                                  # we only need Δ, so skip
    pot = (K*np.abs(ω[i_idx]-ω[j_idx])).sum() \
        + (J*R(i_idx,j_idx)).sum()
    return kin + pot   # spring omitted; irrelevant for Δ

E_before = total_E()

# ---------- analytic ΔE ----------
site = args.site
mask_bonds = (i_idx == site) | (j_idx == site)
h_i = 0.0
for idx in np.where(mask_bonds)[0]:
    j   = j_idx[idx] if i_idx[idx]==site else i_idx[idx]
    w_ij = 0.75*(1+math.cos(φ[site]-φ[j]))*math.exp(-((ω[site]-ω[j])/delta_omega_star)**2)
    h_i += w_ij * s[j]
ΔE_formula = -2*J*s[site]*h_i

# ---------- flip and recompute ----------
s[site] *= -1
E_after  = total_E()
ΔE_direct = E_after - E_before

print(f"Site {site}:")
print(f"  ΔE  (formula) = {ΔE_formula: .6e}")
print(f"  ΔE  (direct ) = {ΔE_direct : .6e}")
