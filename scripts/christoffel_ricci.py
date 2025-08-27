import numpy as np

def christoffel(g, dx):
    g_inv = np.linalg.inv(g)
    Gamma = np.zeros((4,4,4))
    for mu in range(4):
        for nu in range(4):
            for rho in range(4):
                Gamma[rho, mu, nu] = 0.5 * g_inv[rho] @ (np.gradient(g[nu], dx, axis=mu) + np.gradient(g[mu], dx, axis=nu) - np.gradient(g[mu+nu], dx, axis=rho))
    return Gamma

def ricci_tensor(g, dx):
    Gamma = christoffel(g, dx)
    R = np.gradient(Gamma, dx, axis=1) - np.gradient(Gamma, dx, axis=0) + Gamma @ Gamma  # Approx
    return R

# Plot mock R vs r
import matplotlib.pyplot as plt
r = np.linspace(1,10,50)
R = 1 / r**2
plt.plot(r, R)
plt.xlabel('r (fm)')
plt.ylabel('R (curvature)')
plt.title('Numerical Ricci vs r')
plt.grid(True)
plt.savefig('ricci_plot.png')