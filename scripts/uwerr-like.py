import numpy as np
import argparse

ap = argparse.ArgumentParser()
ap.add_argument('infile', help='Input file with CHSH values (one per line)')
args = ap.parse_args()

data = np.loadtxt(args.infile)

def autocorr(x):
    n = len(x)
    mean = np.mean(x)
    var = np.var(x)
    ac = np.correlate(x - mean, x - mean, mode='full') / (var * n)
    return ac[n-1:]  # Positive lags

def tau_int(ac, window=5):
    # Integrated autocorr up to window where ac<0
    cum = 0.5 + np.cumsum(ac[1:])
    tau = cum[np.where(cum < window)[0][-1]] if np.any(cum < window) else cum[-1]
    return tau

ac = autocorr(data)
tau = tau_int(ac)
eff_n = len(data) / (2 * tau)
err = np.std(data) / np.sqrt(eff_n)

print(f"τ_int = {tau:.2f}")
print(f"Effective N = {eff_n:.1f}")
print(f"Binned error = {err:.4f}")