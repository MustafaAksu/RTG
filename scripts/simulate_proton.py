#!/usr/bin/env python
"""
Minimal RTG v2 three-quark Monte-Carlo.
--------------------------------------
 * Uses constants_v2.yaml  (Δω*, g*)
 * Uses recursion_v2.yaml  (a_eff, K/J)
 * 3-node equilateral triangle initialised at N links
 * Plain NumPy implementation; GPU not required for small sweeps.

Typical run:
  python simulate_proton.py --links 18 --sweeps 200000
"""

import argparse, math, time, random
from pathlib import Path
import yaml
import numpy as np

# ------------------------------------------------------------
# 1. Load constants
_repo_root = Path(__file__).resolve().parent.parent   # adjust if needed
with (_repo_root / "src" / "rtg" / "constants_v2.yaml").open() as f:
    C = yaml.safe_load(f)
with (_repo_root / "src" / "rtg" / "recursion_v2.yaml").open() as f:
    R = yaml.safe_load(f)

a_eff = R["a_eff_fm"]                # fm
K_over_J = R["K_over_J"]

# ------------------------------------------------------------
# 2. CLI arguments
p = argparse.ArgumentParser()
p.add_argument("--links",  type=int, default=18,
               help="initial triangle side length in lattice links")
p.add_argument("--sweeps", type=int, default=200_000,
               help="number of Metropolis sweeps")
p.add_argument("--T", type=float, default=1.0,
               help="dimensionless temperature (J units)")
args = p.parse_args()

side = args.links * a_eff            # fm
beta = 1.0 / args.T

# ------------------------------------------------------------
# 3. Geometry helpers
def equilateral(radius):
    """returns 3×2 array of XY coords (fm) for given circumscribed radius"""
    angles = np.array([0, 2*math.pi/3, 4*math.pi/3])
    return np.stack((radius*np.cos(angles), radius*np.sin(angles)), axis=1)

def circ_radius(side):
    return side / math.sqrt(3)

# initial positions
positions = equilateral(circ_radius(side))

# ------------------------------------------------------------
# 4. Energy model (ultra-simple placeholder)
def bond_energy(r):
    """RTG-inspired bond:  K|Δω| term replaced by K_over_J * r  (fm)"""
    return K_over_J * r

def total_energy(pos):
    d01 = np.linalg.norm(pos[0]-pos[1])
    d12 = np.linalg.norm(pos[1]-pos[2])
    d20 = np.linalg.norm(pos[2]-pos[0])
    return bond_energy(d01) + bond_energy(d12) + bond_energy(d20)

# ------------------------------------------------------------
# 5. Metropolis
rng = np.random.default_rng(seed=42)
E = total_energy(positions)
history = []

start = time.time()
for sweep in range(args.sweeps):
    for i in range(3):
        old = positions[i].copy()
        # small random step (Gaussian 0.01 fm)
        positions[i] += rng.normal(0, 0.01, size=2)
        E_new = total_energy(positions)
        if rng.random() < math.exp(-beta*(E_new-E)):   # accept
            E = E_new
        else:                                         # reject
            positions[i] = old
    if sweep % 100 == 0:
        r_circ = np.mean(np.linalg.norm(positions, axis=1))
        history.append(r_circ)

end = time.time()

history = np.array(history)
mean_r = history.mean()
std_r  = history.std()

print(f"Δω*  = {C['delta_omega_star']:.2e} s^-1   (v2)")
print(f"a_eff = {a_eff:.4f} fm   links={args.links}")
print(f"Proton circ radius  ⟨r⟩ = {mean_r:.3f} ± {std_r:.3f} fm")
print(f"MC sweeps           = {args.sweeps:,}")
print(f"Elapsed time        = {end-start:.1f} s")
