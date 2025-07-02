#!/usr/bin/env python
"""
Minimal RTG-v2 three-quark Monte-Carlo
--------------------------------------
* Reads constants_v2.yaml and recursion_v2.yaml via rtg.__init__
* Places an equilateral triangle N lattice links long
* Metropolis updates with simple bond energy  E = (K/J) * r
* Radius is measured in the triangle's COM frame (no drift inflation)
"""
from pathlib import Path
import argparse, math, time, numpy as np
from rtg import delta_omega_star          # already a float
import yaml

# --- change only two things ---------------------------------
PROP_SIGMA = 0.005    # fm  (was 0.01)



# ------------------------------------------------------------
# 1. Load recursion data (a_eff, K/J) -------------------------
_pkg = Path(__file__).resolve().parents[1] / "src" / "rtg"
with (_pkg / "recursion_v2.yaml").open() as f:
    R = yaml.safe_load(f)

a_eff = R["a_eff_fm"]                    # fm
K_over_J = R["K_over_J"]

# ------------------------------------------------------------
# 2. CLI ------------------------------------------------------
p = argparse.ArgumentParser()
p.add_argument("--links",  type=int, default=16,
               help="initial triangle side length in lattice links")
p.add_argument("--sweeps", type=int, default=300_000,
               help="number of Metropolis sweeps")
p.add_argument("--T", type=float, default=1.0,
               help="dimensionless temperature (J units)")
args = p.parse_args()

side = args.links * a_eff               # fm
beta = 1.0 / args.T

# ------------------------------------------------------------
# 3. Geometry helpers ----------------------------------------
def equilateral(radius):
    ang = np.array([0.0, 2*math.pi/3, 4*math.pi/3])
    return np.vstack((radius*np.cos(ang), radius*np.sin(ang))).T

def circ_radius(side):
    return side / math.sqrt(3)

# initial positions
pos = equilateral(circ_radius(side))

# ------------------------------------------------------------
# 4. Energy model (placeholder) -------------------------------
def bond_E(r):         # linear "string tension" (scaled)
    return K_over_J * r

def total_E(p):
    d01 = np.linalg.norm(p[0]-p[1])
    d12 = np.linalg.norm(p[1]-p[2])
    d20 = np.linalg.norm(p[2]-p[0])
    return bond_E(d01) + bond_E(d12) + bond_E(d20)

# ------------------------------------------------------------
# 5. Metropolis MC -------------------------------------------
rng = np.random.default_rng(42)
E = total_E(pos)
history = []

t0 = time.time()
for sweep in range(args.sweeps):
    for i in range(3):
        old = pos[i].copy()
        pos[i] += rng.normal(0, PROP_SIGMA, 2)
        E_new = total_E(pos)
        if rng.random() < math.exp(-beta*(E_new-E)):
            pos[i] = old                           # reject
        else:
            E = E_new

    # ---- recenter: remove CM drift ----
    pos -= pos.mean(axis=0)

    if sweep % 100 == 0:
        radius = np.mean(np.linalg.norm(pos, axis=1))
        history.append(radius)

t1 = time.time()
hist = np.array(history)
print(f"Δω*  = {delta_omega_star:.2e} s^-1   (v2)")
print(f"a_eff = {a_eff:.4f} fm   links={args.links}")
print(f"Proton circ radius  ⟨r⟩ = {hist.mean():.3f} ± {hist.std():.3f} fm")
print(f"MC sweeps           = {args.sweeps:,}")
print(f"Elapsed time        = {t1-t0:.1f} s")
