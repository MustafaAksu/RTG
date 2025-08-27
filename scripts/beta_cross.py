import json, glob, re, math
from collections import defaultdict

pat = re.compile(r'^K_(\d+)_same_rho1p16_a2p55_b(\d+)p(\d+)_(smallt|tfine)\.json$')
curves = defaultdict(list)

files = glob.glob('K_*_same_rho1p16_a2p55_b*_smallt.json') or \
        glob.glob('K_*_same_rho1p16_a2p55_b*_tfine.json')

for f in files:
    m = pat.match(f.split('\\')[-1])
    if not m: 
        continue
    n = int(m.group(1))
    beta = float(f"{m.group(2)}.{m.group(3)}")
    d = json.load(open(f))
    sdim = d["spectral_dimension_mean"]
    se   = d["spectral_dimension_se"]
    curves[n].append((beta, sdim, se))

def sort_and_print():
    for n,v in sorted(curves.items()):
        v.sort()
        print(f"n={n}")
        for beta,sdim,se in v:
            print(f"  beta={beta:.2f}  sdim={sdim:.6f} ± {se:.6f}")
    print()

def crossings(a, b):
    A=sorted(a); B=sorted(b)
    xs=[]
    for i in range(1, min(len(A),len(B))):
        b0, a0, _ = A[i-1]; b1, a1, _ = A[i]
        c0, d0, _ = B[i-1]; c1, d1, _ = B[i]
        if abs((b1-b0)-(c1-c0))>1e-9:  # require same beta grid
            continue
        f0 = a0 - d0
        f1 = a1 - d1
        if f0 == 0: xs.append(b0)
        elif f1 == 0: xs.append(b1)
        elif f0 * f1 < 0:
            t = -f0 / (f1 - f0)
            xs.append(b0 + t*(b1 - b0))
    return xs

sort_and_print()
ns = sorted(curves.keys())
for i in range(len(ns)-1):
    n1, n2 = ns[i], ns[i+1]
    xs = crossings(curves[n1], curves[n2])
    if xs:
        print(f"Crossing between n={n1} and n={n2}: β* ≈ {[round(x, 4) for x in xs]}")
