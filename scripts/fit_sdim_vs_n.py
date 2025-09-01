import json, glob, re, math, numpy as np

# Accept either CURRENT_* or explicit K1 filenames (smallt)
cands = glob.glob("K_*_CURRENT_smallt.json") + glob.glob("K_*_same_rho1p16_a2p55_b1p50_smallt.json")
pat = re.compile(r'^K_(\d+)')

data=[]
for f in cands:
    m = pat.match(f);  n = int(m.group(1)) if m else None
    if not n: continue
    d = json.load(open(f))
    y = float(d["spectral_dimension_mean"])
    se = float(d["spectral_dimension_se"])
    data.append((n, y, se, f))
data.sort()
if len(data) < 3:
    print("Need ≥3 sizes (e.g., 2048,4096,8192). Found:", [x[0] for x in data]); raise SystemExit

N  = np.array([x[0] for x in data], dtype=float)
Y  = np.array([x[1] for x in data], dtype=float)
W  = 1.0 / np.maximum(np.array([x[2] for x in data], dtype=float), 1e-9)**2

best = (1e9, None, None, None)
for omega in np.linspace(0.25, 2.0, 176):   # grid over ω
    X = np.vstack([np.ones_like(N), N**(-omega)]).T
    # weighted least squares
    Aw = X * W[:,None];  bw = Y * W
    theta, *_ = np.linalg.lstsq(Aw, bw, rcond=None)
    yhat = X @ theta
    sse = np.sum(W * (Y - yhat)**2)
    if sse < best[0]:
        best = (sse, omega, theta[0], theta[1])

sse, omega, d_inf, c = best
print("Best power-law fit (weighted):")
print(f"  omega={omega:.3f},  d_inf={d_inf:.6f},  c={c:.6e},  weighted_SSE={sse:.6e}")
print("Points used:")
for n,y,se,f in data:
    print(f"  n={n:5d}  sdim={y:.9f} ± {se:.9f}  <- {f}")
