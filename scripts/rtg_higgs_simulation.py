"""
rtg_higgs_dynamical_exotics.py
Purely RTG-dynamical Higgs-like branching ratio calculation
-----------------------------------------------------------
- Uses particle masses and RTG mass–bandwidth relation
- Computes δω from m_i c_eff^2 / ħ
- Kernel amplitude from RTG resonance kernel
- Spin gates and phase angles sampled naturally
- Includes exotic channels with predicted RTG masses
- Outputs SM vs RTG branching ratios + mass evolution plot
References:
  - RG v1.3.1 for Δω*
  - Forces v1.1 for G_ij gate
  - Particle/Nuclear v2.4 for tetrahedral resonance
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# --- RTG constants ---
hbar_GeV_s = 6.582119569e-25     # GeV·s
c = 3.0e8                        # m/s
c_eff = 0.3 * c                  # m/s (effective speed for hadronic sector)
Delta_omega_star = 1.45e23       # s^-1 (critical bandwidth)

# --- Define decay channels and particle masses (GeV/c^2) ---
# Standard Model-like channels
channels_SM = {
    "bb": 4.18,     # bottom quark mass
    "WW": 80.379,
    "gg": 0.0,      # gluons massless, treat as small effective mass scale
    "ττ": 1.77686,
    "cc": 1.27,
    "ZZ": 91.1876,
    "γγ": 0.0,      # photons massless
    "Zγ": 91.1876,  # treat with Z mass scale
    "μμ": 0.10566
}

# Exotic RTG-predicted channels (masses from hypothetical RTG resonances)
channels_exotics = {
    "corridor_X": 1000.0,   # 1 TeV
    "scalar_glue": 750.0,   # 750 GeV
    "dark_res": 300.0       # 300 GeV
}

# Combine all for RTG calculation
channels_all = {**channels_SM, **channels_exotics}

# --- Function to compute RTG kernel amplitude ---
def rtg_kernel_amplitude(mass_GeV):
    """
    Computes RTG kernel amplitude for a given particle mass.
    δω from m c_eff^2 / ħ, normalized by Δω*.
    Phase and gate sampled naturally.
    """
    # Convert mass to δω in s^-1
    # 1 GeV = 1.602e-10 J; hbar_J_s = hbar_GeV_s * 1.602e-10 J/GeV
    hbar_J_s = hbar_GeV_s * 1.602e-10
    mass_J = mass_GeV * 1.602e-10
    delta_omega = (mass_J * c_eff**2) / hbar_J_s if mass_GeV > 0 else 0.1 * Delta_omega_star
    x = delta_omega / Delta_omega_star

    # Natural phase: uniform in [0, π]
    delta_phi = np.random.uniform(0, np.pi)
    cos_term = np.cos(delta_phi)

    # Spin gate: 50% open
    gate = np.random.choice([0, 1], p=[0.5, 0.5])

    # Kernel amplitude
    R = (3/4) * (1 + cos_term) * gate * np.exp(-x**2)
    return R

# --- Compute RTG partial widths ---
np.random.seed(42)  # for reproducibility
partial_widths_RTG = {}
for ch, m in channels_all.items():
    partial_widths_RTG[ch] = rtg_kernel_amplitude(m)

# Normalize to get branching ratios (%)
total_width_RTG = sum(partial_widths_RTG.values())
BR_RTG = {ch: (pw / total_width_RTG) * 100 for ch, pw in partial_widths_RTG.items()}

# --- SM branching ratios for comparison (PDG values in %) ---
BR_SM_empirical = {
    "bb": 58.09,
    "WW": 21.52,
    "gg": 8.18,
    "ττ": 6.27,
    "cc": 2.89,
    "ZZ": 2.64,
    "γγ": 0.23,
    "Zγ": 0.15,
    "μμ": 0.021
}
# Fill zeros for exotics in SM
for ex in channels_exotics.keys():
    BR_SM_empirical[ex] = 0.0

# --- Mass evolution in RTG Higgs-like field ---
t_max = 5e-23  # s
num_steps = 200
t = np.linspace(0, t_max, num_steps)
target_mass_H = 125.0  # GeV (Higgs-like mass scale)
tau_growth = 1.5e-23   # s
mass_evolution = target_mass_H * (1 - np.exp(-t / tau_growth))

# --- Prepare DataFrame for saving ---
df = pd.DataFrame({
    "Channel": list(channels_all.keys()),
    "Mass_GeV": [channels_all[ch] for ch in channels_all.keys()],
    "BR_SM_percent": [BR_SM_empirical[ch] for ch in channels_all.keys()],
    "BR_RTG_percent": [BR_RTG[ch] for ch in channels_all.keys()],
    "Partial_width_RTG": [partial_widths_RTG[ch] for ch in channels_all.keys()]
})
df.to_csv("rtg_higgs_BR_dynamical_exotics.csv", index=False)

# --- Plot ---
fig, axs = plt.subplots(1, 2, figsize=(14, 5))

# Branching ratio comparison
x = np.arange(len(channels_all))
width = 0.35
axs[0].bar(x - width/2, [BR_SM_empirical[ch] for ch in channels_all.keys()],
           width, label="SM", color='blue')
axs[0].bar(x + width/2, [BR_RTG[ch] for ch in channels_all.keys()],
           width, label="RTG", color='orange')
axs[0].set_xticks(x)
axs[0].set_xticklabels(channels_all.keys(), rotation=45, ha="right")
axs[0].set_ylabel("Branching Ratio (%)")
axs[0].set_title("Higgs Branching Ratios: SM vs RTG (purely dynamical)")
axs[0].legend()
axs[0].grid(True, linestyle="--", alpha=0.5)

# Mass evolution
axs[1].plot(t, mass_evolution, color='green', label="Particle mass")
axs[1].set_xlabel("Time (s)")
axs[1].set_ylabel("Mass (GeV/c²)")
axs[1].set_title("Mass Evolution in RTG Higgs-like Field")
axs[1].legend()
axs[1].grid(True, linestyle="--", alpha=0.5)

plt.tight_layout()
plt.savefig("rtg_higgs_dynamical_exotics.png", dpi=300)
plt.show()

# --- Print summary ---
print("RTG Higgs-like Branching Ratios (%):")
for ch in channels_all.keys():
    print(f" {ch}: {BR_RTG[ch]:.3f}")
print("\nSaved: rtg_higgs_dynamical_exotics.png and rtg_higgs_BR_dynamical_exotics.csv")
