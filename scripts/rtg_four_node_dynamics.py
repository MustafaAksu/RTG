#!/usr/bin/env python3
"""
rtg_four_node_dynamics.py — RTG dynamics with two photon-like pairs
• All ω, φ evolve; spins fixed
• Energy + pairwise resonances + full node states saved
"""

import numpy as np
import matplotlib.pyplot as plt
import csv

# Parameters
steps = 10000
dt = 1e-3
J, K = 1.0, 3.0

# Initial configuration
ω = np.array([+0.55, -0.55, +0.08, -0.08])
φ = np.array([0.0, 0.0, np.pi/2, np.pi/2])
s = np.array([+1, -1, +1, -1])
ω_prev = ω.copy()
φ_prev = φ.copy()

def resonance(i, j):
    dω = ω[i] - ω[j]
    dφ = φ[i] - φ[j]
    gauss = np.exp(-dω**2)
    return 0.75 * (1 + np.cos(dφ)) * (1 + s[i] * s[j]) * gauss

def total_energy():
    E = 0.0
    for i in range(4):
        for j in range(i + 1, 4):
            E += K * abs(ω[i] - ω[j]) + J * resonance(i, j)
    return E

def forces():
    Fω = np.zeros(4)
    Fφ = np.zeros(4)
    for i in range(4):
        for j in range(4):
            if i == j: continue
            dω = ω[i] - ω[j]
            dφ = φ[i] - φ[j]
            gauss = np.exp(-dω**2)
            spin = 1 + s[i]*s[j]
            cos = np.cos(dφ)
            sin = np.sin(dφ)
            dR_dω = -2 * dω * 0.75 * (1 + cos) * spin * gauss
            dR_dφ = -0.75 * sin * spin * gauss
            Fω[i] += K * np.sign(dω) + J * dR_dω
            Fφ[i] += J * dR_dφ
    return -Fω, -Fφ

# ───── Logging ─────
E_log = []
R_log = { (i,j): [] for i in range(4) for j in range(i+1,4) }
state_log = open("rtg_four_node_states.csv", "w", newline='')
writer = csv.writer(state_log)
writer.writerow(["step", "node", "omega", "phi", "spin"])

# ───── Main Loop ─────
for t in range(steps):
    E_log.append(total_energy())
    for (i,j) in R_log:
        R_log[i,j].append(resonance(i,j))

    for i in range(4):
        writer.writerow([t, i, ω[i], φ[i], s[i]])

    Fω, Fφ = forces()
    ω_prev[:] = ω
    φ_prev[:] = φ
    ω += dt * Fω
    φ += dt * Fφ
    φ %= 2*np.pi

state_log.close()

# ───── Plot ─────
plt.figure(figsize=(10,6))
plt.plot(E_log, label="Total Energy")
for (i,j), Rvals in R_log.items():
    plt.plot(Rvals, label=f"R{i}{j}")
plt.xlabel("Step")
plt.ylabel("Energy / Resonance")
plt.title("RTG Dynamics of Two Photon-like Pairs")
plt.legend()
plt.tight_layout()
plt.savefig("rtg_four_node_dynamics.png")
plt.show()
