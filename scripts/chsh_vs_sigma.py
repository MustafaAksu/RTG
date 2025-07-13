# chsh_vs_sigma.py
import numpy as np, matplotlib.pyplot as plt
sigma  = np.array([.01,.02,.03,.04,.05,.06,.07])
chsh   = np.array([2.82338,2.82319,2.82287,2.82227,2.82191,2.82049,2.81935])
errors = np.array([0.00061,0.00056,0.00055,0.00075,0.00040,0.00028,0.00046])

plt.errorbar(sigma, chsh, yerr=errors, fmt='o')
plt.axhline(2*np.sqrt(2), ls='--', label='Tsirelson')
plt.xlabel(r'Gaussian noise $\sigma/\Delta\omega^*$')
plt.ylabel(r'$\langle\text{CHSH}\rangle$')
plt.ylim(2.79, 2.83); plt.legend(); plt.tight_layout()
plt.savefig('chsh_sigma.png', dpi=300)