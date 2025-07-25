# Re-importing necessary libraries due to reset
import matplotlib.pyplot as plt
import numpy as np

# Data for T=0.00
sigma_T000 = np.array([
    0.00, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10, 0.11, 0.12, 
    0.13, 0.14, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60
])
chsh_T000 = np.array([
    2.825, 2.824, 2.823, 2.820, 2.817, 2.813, 2.808, 2.800, 2.795, 2.787, 2.777,
    2.765, 2.758, 2.744, 2.733, 2.721, 2.642, 2.544, 2.433, 2.328, 2.198, 2.066,
    1.943, 1.796, 1.654
])

# Data for T=0.01
sigma_T001 = np.array([
    0.00, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10,
    0.11, 0.12, 0.13, 0.14, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45,
    0.50, 0.55, 0.60
])
chsh_T001 = np.array([
    2.825, 2.825, 2.823, 2.821, 2.817, 2.813, 2.808, 2.801, 2.795, 2.785, 2.779,
    2.770, 2.757, 2.747, 2.734, 2.718, 2.645, 2.546, 2.430, 2.324, 2.206, 2.050,
    1.946, 1.789, 1.668
])

# Data for T=0.02
sigma_T002 = np.array([
    0.00, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10, 0.15, 0.20, 
    0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60
])
chsh_T002 = np.array([
    2.825, 2.825, 2.823, 2.821, 2.817, 2.813, 2.808, 2.802, 2.795, 2.788, 2.779,
    2.725, 2.646, 2.549, 2.439, 2.326, 2.194, 2.092, 1.944, 1.828, 1.676
])

# Data for T=0.05
sigma_T005 = np.array([
    0.00, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10,
    0.11, 0.12, 0.13, 0.14, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45,
    0.50, 0.55, 0.60
])
chsh_T005 = np.array([
    2.825, 2.825, 2.823, 2.821, 2.817, 2.814, 2.808, 2.801, 2.794, 2.787, 2.775,
    2.768, 2.757, 2.745, 2.734, 2.722, 2.643, 2.548, 2.436, 2.310, 2.197, 2.074,
    1.940, 1.820, 1.659
])

# Data for T=0.20
sigma_T020 = np.array([
    0.00, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10,
    0.11, 0.12, 0.13, 0.14, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45,
    0.50, 0.55, 0.60
])
chsh_T020 = np.array([
    2.825, 2.825, 2.823, 2.821, 2.817, 2.813, 2.808, 2.802, 2.794, 2.786, 2.777,
    2.769, 2.755, 2.743, 2.734, 2.721, 2.643, 2.546, 2.443, 2.326, 2.209, 2.078,
    1.967, 1.828, 1.677
])


# Plotting the CHSH values for both temperatures
plt.figure(figsize=(10, 6))

# T=0.00 data
plt.plot(sigma_T000, chsh_T000, 's-', label='CHSH (T=0.00)', color='orange')

# Plot data for T=0.01
plt.plot(sigma_T001, chsh_T001, 'd-', label='CHSH (T=0.01)', color='purple')

# T=0.02 data
plt.plot(sigma_T002, chsh_T002, 'o-', label='CHSH (T=0.02)', color='blue')

# Plot data for T=0.05
plt.plot(sigma_T005, chsh_T005, '^-', label='CHSH (T=0.05)', color='green')

# Plot data for T=0.20
plt.plot(sigma_T020, chsh_T020, '^-', label='CHSH (T=0.20)', color='red')

# Quantum and classical boundary lines
plt.axhline(y=2, color='red', linestyle='--', label='Classical Limit (CHSH = 2)')
plt.axhline(y=2.828, color='green', linestyle='--', label="Quantum Limit (Tsirelson's bound ≈ 2.828)")

# Labels and title
plt.xlabel('ω-noise σ')
plt.ylabel('CHSH value')
plt.title('Quantum-to-Classical Transition in RTG Model\nComparing T=0.02 vs T=0.00')
plt.grid(True)
plt.legend(fontsize=11)
plt.tight_layout()

plt.show()
