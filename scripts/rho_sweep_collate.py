import json, glob, re
import numpy as np

pat = re.compile(r'^K_(\d+)_same_rho(\d+)p(\d+)_a2p55_b1p50_smallt\.json$')
rows=[]
for f in glob.glob('K_*_same_rho*_a2p55_b1p50_smallt.json'):
    m = pat.match(f)
    if not m: 
        continue
    n = int(m.group(1))
    rs = float(f"{m.group(2)}.{m.group(3)}")
    d = json.load(open(f))
    rows.append((n, rs, d["spectral_dimension_mean"], d["spectral_dimension_se"]))
rows.sort()
print("n, rho_scale, sdim, se")
for r in rows:
    print(*r, sep=", ")
