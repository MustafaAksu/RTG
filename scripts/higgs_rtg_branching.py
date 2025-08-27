# higgs_rtg_branching.py
# Simulation of SM vs RTG Higgs branching ratios (+5% width)
# Save plot & table for comparison

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -----------------------
# 1. Input: SM partial widths in MeV (PDG ~2024 values)
# -----------------------
channels = ["bb", "WW", "gg", "ττ", "cc", "ZZ", "γγ", "Zγ", "μμ"]
Gamma_SM = np.array([
    2.38,     # bb
    0.87,     # WW
    0.34,     # gg
    0.25,     # ττ
    0.12,     # cc
    0.11,     # ZZ
    0.0093,   # γγ
    0.006,    # Zγ
    0.00022   # μμ
])

# -----------------------
# 2. SM branching ratios
# -----------------------
Gamma_total_SM = np.sum(Gamma_SM)
BR_SM = Gamma_SM / Gamma_total_SM

# -----------------------
# 3. RTG scaling (+5% total width)
# -----------------------
scale_total_width = 1.05  # +5%
Gamma_total_RTG = Gamma_total_SM * scale_total_width
Gamma_RTG = Gamma_SM * scale_total_width
BR_RTG = Gamma_RTG / Gamma_total_RTG

# -----------------------
# 4. Output table
# -----------------------
df = pd.DataFrame({
    "Channel": channels,
    "Gamma_SM_MeV": Gamma_SM,
    "BR_SM_%": BR_SM * 100,
    "Gamma_RTG_MeV": Gamma_RTG,
    "BR_RTG_%": BR_RTG * 100
})

print("\n=== Higgs Branching Ratios: SM vs RTG (+5% total width) ===")
print(df.to_string(index=False, formatters={
    "Gamma_SM_MeV": "{:.5f}".format,
    "BR_SM_%": "{:.3f}".format,
    "Gamma_RTG_MeV": "{:.5f}".format,
    "BR_RTG_%": "{:.3f}".format
}))

# Save to CSV for sharing
df.to_csv("higgs_BR_results.csv", index=False)

# -----------------------
# 5. Plot comparison
# -----------------------
x = np.arange(len(channels))
plt.figure(figsize=(9,6))
plt.bar(x - 0.15, BR_SM*100, width=0.3, label="SM")
plt.bar(x + 0.15, BR_RTG*100, width=0.3, label="RTG")
plt.xticks(x, channels)
plt.ylabel("Branching Ratio (%)")
plt.title("Higgs Branching Ratios: SM vs RTG (+5% total width)")
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig("higgs_BR.png", dpi=300)

print("\nSaved: higgs_BR_results.csv and higgs_BR.png")
