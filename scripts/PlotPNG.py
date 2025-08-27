import numpy as np
def riemann_tensor(g, dx=1e-3):
    Gamma = christoffel(g, dx)
    R = np.zeros((4,4,4,4))
    for rho in range(4):
        for sigma in range(4):
            for mu in range(4):
                for nu in range(4):
                    R[rho, sigma, mu, nu] = (
                        np.gradient(Gamma[rho, nu, sigma], dx, axis=mu) -
                        np.gradient(Gamma[rho, mu, sigma], dx, axis=nu) +
                        np.einsum('l,lsn', Gamma[rho, mu], Gamma[l, nu, sigma]) -
                        np.einsum('l,lsm', Gamma[rho, nu], Gamma[l, mu, sigma])
                    )
    return R

# Mock plot: R vs r for rho=0, sigma=1, mu=2, nu=3
import matplotlib.pyplot as plt
r = np.linspace(1,10,50)
R_mock = 1 / r**3  # Approx component
plt.plot(r, R_mock)
plt.xlabel('r (fm)')
plt.ylabel('R Component')
plt.title('Riemann Component vs r')
plt.grid(True)
plt.savefig('riemann_plot.png')