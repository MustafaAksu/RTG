import numpy as np
import matplotlib.pyplot as plt

# Physical constants
hbar = 1.0545718e-34  # J·s
c = 2.99792458e8      # m/s
K = 1.92e-12          # J·s (from RTG lattice fit)
Delta_omega_star = 1.45e23  # s⁻¹

# Cluster sizes (node count)
N = np.logspace(7, 10, 50)
G_eff = (K * hbar) / (2 * np.pi * c * Delta_omega_star * N**2)

# Target value of Newton's constant
G_phys = 6.67430e-11  # m³ kg⁻¹ s⁻²

# Plot
plt.figure(figsize=(10, 5))

# Effective G vs N
plt.subplot(1, 2, 1)
plt.plot(N, G_eff, label='RTG $G_{\\mathrm{eff}}(N)$')
plt.axhline(G_phys, color='r', linestyle='--', label='Physical $G$')
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Cluster Size N')
plt.ylabel('$G_{\\mathrm{eff}}$ (m³ kg⁻¹ s⁻²)')
plt.title('RTG Effective Newton Constant vs Cluster Size')
plt.legend()

# Relative deviation from physical G
plt.subplot(1, 2, 2)
error = 100 * np.abs((G_eff - G_phys) / G_phys)
plt.plot(N, error)
plt.axhline(3, color='green', linestyle='--', label='3% error')
plt.xscale('log')
plt.xlabel('Cluster Size N')
plt.ylabel('Deviation from $G$ (%)')
plt.title('Relative Error in $G_{\\mathrm{eff}}(N)$')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()
