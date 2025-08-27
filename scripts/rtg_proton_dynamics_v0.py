#!/usr/bin/env python3
"""
rtg_proton_dynamics_v0.py — Minimal proton dynamics in RTG node model
• One Planck observer node (index 0), three-node proton (indices 1–3)
• Canonical RTG resonance kernel
• Dynamic evolution: ω, φ, s evolve under force-based update
• Tracks energy: potential, curvature, emergent kinetic (from ω, φ changes)
• No Langevin thermostat, no lattice — pure internal RTG mechanics
• Uses dimensionless ω in units of Δω*
• Observer node does not participate in dynamics
"""

import numpy as np
import math, time
import matplotlib.pyplot as plt

# ───────── Simulation parameters ─────────
steps     = 5000
step_size = 1e-5
J, K      = 1.0, 3.0
Δω_star   = 1.45e23   # RTG value in s^-1
ħ         = 1.054571817e-34  # Planck's constant (J·s)

# ───────── Initial configuration ─────────
ω  = np.array([1.0, -0.15, 0.0, +0.15])  # ω_i / Δω*, observer at index 0
φ  = np.array([0.0, 0.0, 2*np.pi/3, 4*np.pi/3])
s  = np.array([+1, +1, +1, +1])

ω_prev = ω.copy()
φ_prev = φ.copy()

# ───────── Node-wise forces from resonance kernel ─────────
def resonance(i, j):
    dω = ω[i] - ω[j]
    dφ = φ[i] - φ[j]
    gauss = np.exp(-dω**2)
    cos_term = np.cos(dφ)
    return 0.75 * (1 + cos_term) * (1 + s[i]*s[j]) * gauss

def forces():
    Fω = np.zeros(4)
    Fφ = np.zeros(4)
    for i in range(1, 4):
        for j in range(1, 4):
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

def total_energy():
    E = 0.0
    for i in range(1, 4):
        for j in range(i+1, 4):
            dω = abs(ω[i] - ω[j])
            R = resonance(i, j)
            E += K * dω + J * R
    return E * ħ * Δω_star  # convert to Joules

def curvature_energy():
    ideal_angle = 2 * np.pi / 3  # For equilateral triangle
    indices = [(1,2), (2,3), (3,1)]
    E = 0.0
    for i, j in indices:
        dφ = ((φ[i] - φ[j] + np.pi) % (2*np.pi)) - np.pi
        deviation = dφ - ideal_angle
        E += 0.5 * deviation**2
    return E * ħ * Δω_star

def kinetic_energy_proxies():
    dφ = (φ[1:] - φ_prev[1:]) / step_size
    dω = (ω[1:] - ω_prev[1:]) / step_size
    KE_phi = 0.5 * np.sum(dφ**2)
    KE_omega = 0.5 * np.sum(dω**2)
    return KE_phi * ħ * Δω_star, KE_omega * ħ * Δω_star

def observed_freq(ω_ref, ω_i):
    return (ω_i - ω_ref) * Δω_star

def observed_phase(φ_ref, φ_i):
    return ((φ_i - φ_ref + np.pi) % (2*np.pi)) - np.pi

# ───────── Logging setup ─────────
with open("proton_dynamics_states.csv", "w") as f:
    f.write("step,node,omega_dimless,phi,spin,omega_obs_Hz,phi_obs_rad\n")
with open("proton_energy_ledger.csv", "w") as f:
    f.write("step,E_pot_J,E_curv_J,E_phi_kin_J,E_omega_kin_J,E_total_J\n")

# ───────── Simulation loop ─────────
print("  Step    Energy  R12=        R23=        R31=")
start = time.time()

for t in range(steps+1):
    if t % 50 == 0:
        E_pot = total_energy()
        E_curv = curvature_energy()
        E_phi_kin, E_omega_kin = kinetic_energy_proxies()
        E_tot = E_pot + E_curv + E_phi_kin + E_omega_kin
        R12 = resonance(1,2)
        R23 = resonance(2,3)
        R31 = resonance(3,1)
        print(f"{t:6d}  {E_tot:.6e}  R12={R12:.6f}, R23={R23:.6f}, R31={R31:.6f}")

        with open("proton_dynamics_states.csv", "a") as f:
            for i in range(4):
                ω_obs = observed_freq(ω[0], ω[i])
                φ_obs = observed_phase(φ[0], φ[i])
                f.write(f"{t},{i},{ω[i]:.8f},{φ[i]:.12f},{s[i]},{ω_obs:.6e},{φ_obs:.12f}\n")

        with open("proton_energy_ledger.csv", "a") as f:
            f.write(f"{t},{E_pot:.6e},{E_curv:.6e},{E_phi_kin:.6e},{E_omega_kin:.6e},{E_tot:.6e}\n")

    Fω, Fφ = forces()
    ω_prev[:] = ω
    φ_prev[:] = φ
    ω += step_size * Fω
    φ += step_size * Fφ
    φ %= 2*np.pi

print(f"\nDone in {time.time() - start:.1f}s")

# ───────── Plotting ─────────
import pandas as pd
log = pd.read_csv("proton_energy_ledger.csv")
plt.plot(log['step'], log['E_pot_J'], label="E_pot")
plt.plot(log['step'], log['E_curv_J'], label="E_curvature")
plt.plot(log['step'], log['E_phi_kin_J'], label="E_phi_kin")
plt.plot(log['step'], log['E_omega_kin_J'], label="E_omega_kin")
plt.plot(log['step'], log['E_total_J'], label="E_total", linestyle="--")
plt.legend()
plt.xlabel("Step")
plt.ylabel("Energy (Joules)")
plt.title("Energy Components in RTG Proton Shell")
plt.tight_layout()
plt.savefig("rtg_proton_energy_components.png")
