# sanity_phi.py
import argparse, glob, numpy as np, matplotlib.pyplot as plt
ap = argparse.ArgumentParser(); ap.add_argument('cfgdir')
args = ap.parse_args()

phis = []
for f in sorted(glob.glob(f'{args.cfgdir}/cfg_*.npz')):
    data = np.load(f)
    phis.append((data['phi'][1] - data['phi'][0]) % (2*np.pi))
phis = np.asarray(phis)

plt.hist((phis - np.pi), bins=50, density=True)
plt.xlabel(r'$\Delta\phi-\pi$'); plt.ylabel('P(Δφ)')
plt.title('Phase locking of Bell pair')
plt.axvline(0, ls='--'); plt.tight_layout(); plt.show()
