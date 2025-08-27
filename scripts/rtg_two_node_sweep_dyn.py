#!/usr/bin/env python3
"""
rtg_two_node_scan_v2.py — Two-node RTG dynamics with evolving ω and φ
• Sweeps Δω from 0.0 to 1.0 in 0.05 steps
• For each Δω, initializes two nodes with ω = ±Δω/2
• Evolves ω and φ using gradient descent dynamics
• Logs energy over time for each Δω
• Plots energy evolution curves
"""

import numpy as np
import matplotlib.pyplot as plt

# ────────────── Constants ──────────────
Δω_star = 1.0   # working in dimensionless ω units
J, K = 1.0, 3.0
step_size = 1e-3
steps = 1000
Δω_list = np.round(np.arange(0.0, 1.01, 0.05), 2)

def resonance(ω, φ, s):
    dω = ω[0] - ω[1]
    dφ = φ[0] - φ[1]
    gauss = np.exp(-dω**2)
    return 0.75 * (1 + np.cos(dφ)) * (1 + s[0]*s[1]) * gauss

def energy(ω, φ, s):
    dω = abs(ω[0] - ω[1])
    R = resonance(ω, φ, s)
    return K * dω + J * R

def evolve(Δω_init):
    ω = np.array([-Δω_init/2, Δω_init/2])
    φ = np.array([0.0, 0.0])
    s = np.array([+1, +1])
    E_series = []

    for _ in range(steps):
        dω = ω[0] - ω[1]
        dφ = φ[0] - φ[1]
        gauss = np.exp(-dω**2)
        sin = np.sin(dφ)
        cos = np.cos(dφ)
        spin_term = (1 + s[0]*s[1])

        # Gradients
        dR_dω = -2 * dω * 0.75 * (1 + cos) * spin_term * gauss
        dR_dφ = -0.75 * sin * spin_term * gauss

        Fω = np.array([
            -K * np.sign(dω) - J * dR_dω,
            +K * np.sign(dω) + J * dR_dω
        ])
        Fφ = np.array([
            -J * dR_dφ,
            +J * dR_dφ
        ])

        ω += step_size * Fω
        φ += step_size * Fφ
        φ %= 2*np.pi

        E_series.append(energy(ω, φ, s))
    
    return E_series

# ────────────── Run sweep ──────────────
results = {}
for Δω in Δω_list:
    results[Δω] = evolve(Δω)

# ────────────── Plot ──────────────
plt.figure(figsize=(10,6))
for Δω, E in results.items():
    plt.plot(E, label=f"Δω={Δω:.2f}")
plt.xlabel("Step")
plt.ylabel("Energy")
plt.title("RTG Two-Node Energy Evolution vs Δω")
plt.legend(ncol=2, fontsize="small")
plt.grid(True)
plt.tight_layout()
plt.savefig("rtg_two_node_energy_evolution.png")
plt.show()
