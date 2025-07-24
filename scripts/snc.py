import numpy as np
from pathlib import Path

snapshots = sorted(Path('cfg_md_dyn').glob('snap_*.npz'))

gate_factors = []
phase_diffs = []

for snap in snapshots:
    data = np.load(snap)
    s, φ = data['spin'], data['phi']
    
    gate = 1 + s[0] * s[1]
    gate_factors.append(gate)
    
    dphi = (φ[1] - φ[0] - np.pi + np.pi) % (2*np.pi) - np.pi
    phase_diffs.append(dphi)

gate_factors = np.array(gate_factors)
phase_diffs = np.array(phase_diffs)

print(f"Gate factor (1 + s₀s₁) values: {np.unique(gate_factors)}")
print(f"Phase difference (Δφ - π): mean={phase_diffs.mean():.3e}, std={phase_diffs.std():.3e}")
