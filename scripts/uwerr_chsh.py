#!/usr/bin/env python
# uwerr_chsh.py – robust version
import glob, argparse, math, numpy as np

# ---------- CHSH with full resonance term ----------------------------
def chsh_from_cfg(ω, φ, s, sigma):
    i, j = 0, 1
    a, a_p = 0.0, math.pi/2
    b, b_p = math.pi/4, -math.pi/4
    ωj = ω[j] + (np.random.normal(0.0, sigma) if sigma else 0.0)

    def R(θa, θb):
        cos = math.cos((φ[i] + θa) - (φ[j] + θb))
        spin = 1 + np.real(1j * s[i] * 1j * s[j])     # = 2 for Bell pair
        gauss = math.exp(-((ω[i] - ωj)**2))
        return 0.75 * (1 + cos) * spin * gauss

    def E(θa, θb):
        return 2*R(θa, θb)/3.0 - 1.0
    return abs(E(a, b) + E(a, b_p) + E(a_p, b) - E(a_p, b_p))

# ---------- Γ-method / UWerr -----------------------------------------
def uwerr(x):
    x = np.asarray(x, float)
    n  = len(x)
    mean = x.mean()
    var  = x.var(ddof=1)
    if var == 0.0:                    # constant signal → zero error
        return mean, 0.0, 0.0

    W = min(n//2, 100)
    gamma = []
    for t in range(1, W):
        c = np.corrcoef(x[:-t], x[t:])[0, 1]
        gamma.append(c)
        if c < 0:
            break
    tau_int = 0.5 + sum(gamma)
    err = math.sqrt(2 * tau_int * var / n)
    return mean, err, tau_int

# ---------- CLI & main loop ------------------------------------------
ap = argparse.ArgumentParser()
ap.add_argument('cfgdir')
ap.add_argument('--sigma', type=float, default=0.01)
args = ap.parse_args()

values = []
for fn in sorted(glob.glob(f'{args.cfgdir}/cfg_*.npz')):
    d = np.load(fn)
    values.append(chsh_from_cfg(d['omega'], d['phi'], d['spin'], args.sigma))

mean, err, tau = uwerr(values)
print(f'⟨CHSH⟩ = {mean:.5f} ± {err:.5f}   (τ_int ≈ {tau:.2f} traj, σ={args.sigma})')
