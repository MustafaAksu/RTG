"""
rtg package initialiser
-----------------------
Exposes the universal constants defined in constants_v2.yaml
so that every script / notebook pulls *exactly* the same numbers.
"""

import importlib.resources as rsc
import yaml

# ------------------------------------------------------------
# 1. Load YAML file that ships with the package
with rsc.open_text("rtg", "constants_v2.yaml") as f:
    _c = yaml.safe_load(f)

# ------------------------------------------------------------
# 2. Cast every numeric to float
delta_omega_star = float(_c["delta_omega_star"])
g_star            = float(_c["g_star"])
beta_coeffs       = [float(x) for x in _c["beta_coeffs"]]

# (Nothing else executed at import-time; package stays lightweight.)
