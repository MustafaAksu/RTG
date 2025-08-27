# rtg_two_node_sweep.py
# Two-node RTG simulation: static spin + phase, sweeping frequency difference

import numpy as np
import matplotlib.pyplot as plt

# Constants (dimensionless units)
Δω_star = 1.0
J = 1.0
K = 3.0
steps = 1000
s1, s2 = +1, +1
φ1, φ2 = 0.0, 0.0  # fixed

# Sweep over frequency differences
Δω_values = np.linspace(0.1, 1.0, 20)
resonance_vals = []
energy_vals = []

for Δω in Δω_values:
    ω1 = 0.0
    ω2 = ω1 + Δω

    dω = ω1 - ω2
    dφ = φ1 - φ2
    gauss = np.exp(-(dω / Δω_star)**2)
    cos_term = np.cos(dφ)
    R12 = 0.75 * (1 + cos_term) * (1 + s1 * s2) * gauss
    E12 = K * abs(dω) + J * R12

    resonance_vals.append(R12)
    energy_vals.append(E12)

    print(f"Δω = {Δω:.3f},  R₁₂ = {R12:.4f},  E = {E12:.4f}")

# Plotting
plt.plot(Δω_values, resonance_vals, label="Resonance R₁₂")
plt.plot(Δω_values, energy_vals, label="Total Energy", linestyle="--")
plt.xlabel("Δω (in units of Δω*)")
plt.ylabel("Value")
plt.title("RTG Two-Node Sweep")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("rtg_two_node_sweep.png")
plt.show()
