import importlib.resources as rsc, yaml
with rsc.open_text('rtg', 'constants_v2.yaml') as f:
    _c = yaml.safe_load(f)
delta_omega_star = _c['delta_omega_star']
g_star            = _c['g_star']
beta_coeffs       = _c['beta_coeffs']
