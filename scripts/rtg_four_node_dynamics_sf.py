#!/usr/bin/env python3
"""
rtg_four_node_dynamics.py — RTG dynamics with two photon-like pairs
• Four nodes: two symmetric ω-pairs with opposite spins and locked phases
• All nodes evolve (ω, φ), spin flips included (deterministic)
• Tracks potential energy and pairwise resonance over time
"""

import numpy as np
import matplotlib.pyplot as plt
import csv

# Parameters
steps = 10000
dt = 1e-3
J, K = 1.0, 3.0

# Initial conditions: symmetric ω, phase-locked, opposite spins
ω = np.array([1.0, 1.0, 1.0, 1.0])
φ = np.array([0.0, 0.0, 0.0, 0.0])
s = np.array([+1, +1, +1, -1])

ω_prev = ω.copy()
φ_prev = φ.copy()

# Resonance and energy

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
            spin = 1 + s[i] * s[j]
            cos = np.cos(dφ)
            sin = np.sin(dφ)
            dR_dω = -2 * dω * 0.75 * (1 + cos) * spin * gauss
            dR_dφ = -0.75 * sin * spin * gauss
            Fω[i] += K * np.sign(dω) + J * dR_dω
            Fφ[i] += J * dR_dφ
    return -Fω, -Fφ

# Logging
E_log = []
R_log = { (i,j): [] for i in range(4) for j in range(i+1,4) }

with open("rtg_four_node_states.csv", "w", newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["step", "node", "omega", "phi", "spin"])

# Main loop
for t in range(steps):
    E_log.append(total_energy())
    for (i,j) in R_log:
        R_log[i,j].append(resonance(i,j))

    # Deterministic spin flips
    for i in range(4):
        s_old = s[i]
        E_before = sum(J * resonance(i, j) for j in range(4) if j != i)
        s[i] *= -1
        E_after = sum(J * resonance(i, j) for j in range(4) if j != i)
        if E_after > E_before:
            s[i] = s_old  # Reject flip

    Fω, Fφ = forces()
    ω_prev[:] = ω
    φ_prev[:] = φ
    ω += dt * Fω
    φ += dt * Fφ
    φ %= 2 * np.pi

    if t % 100 == 0:
        with open("rtg_four_node_states.csv", "a", newline='') as f:
            writer = csv.writer(f)
            for i in range(4):
                writer.writerow([t, i, ω[i], φ[i], s[i]])

# Plotting
plt.figure(figsize=(10,6))
plt.plot(E_log, label="Total Energy")
for (i,j), Rvals in R_log.items():
    plt.plot(Rvals, label=f"R{i}{j}")
plt.xlabel("Step")
plt.ylabel("Energy / Resonance")
plt.title("RTG Dynamics of Two Photon-like Pairs (with Spin Flips)")
plt.legend()
plt.tight_layout()
plt.savefig("rtg_four_node_dynamics.png")
plt.show()
