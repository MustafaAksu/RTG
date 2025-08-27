import json, glob, re
import numpy as np
from collections import defaultdict

pat = re.compile(r'^K_(\d+)_same_rho1p16_a2p55_b(\d+)p(\d+)_tfine\.json$')
curves = defaultdict(list)

for f in glob.glob('K_*_same_rho1p16_a2p55_b*_tfine.json'):
    m = pat.match(f)
    if not m: continue
    n = int(m.group(1))
    beta = float(f"{m.group(2)}.{m.group(3)}")
    d = json.load(open(f))
    sdim = d["spectral_dimension_mean"]
    se   = d["spectral_dimension_se"]
    curves[n].append((beta, sdim, se))

# sort by beta and print
for n,v in sorted(curves.items()):
    v.sort()
    print(f"n={n}")
    for beta,sdim,se in v:
        print(f"  beta={beta:.2f}  sdim={sdim:.6f} ± {se:.6f}")

# simple pairwise crossing estimate by linear interpolation
def crossings(a, b):
    X=[]
    A=sorted(a); B=sorted(b)
    for i in range(1, min(len(A),len(B))):
        b0, a0, _ = A[i-1]; b1, a1, _ = A[i]
        c0, d0, _ = B[i-1]; c1, d1, _ = B[i]
        if abs((b1-b0)-(c1-c0))>1e-9: # assume same beta grid
            continue
        if (a0-d0)*(a1-d1) <= 0:  # sign change -> crossing
            t = (0 - (a0-d0)) / ((a1-d1) - (a0-d0) + 1e-12)
            beta_star = b0 + t*(b1-b0)
            X.append(beta_star)
    return X

ns = sorted(curves.keys())
for i in range(len(ns)-1):
    n1, n2 = ns[i], ns[i+1]
    xs = crossings(curves[n1], curves[n2])
    if xs:
        print(f"Crossing between n={n1} and n={n2} at β ≈ {xs}")
