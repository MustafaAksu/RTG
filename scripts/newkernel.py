import numpy as np
from numpy.linalg import eigh, svd
from sklearn.manifold import MDS, Isomap
from sklearn.metrics import pairwise_distances
import networkx as nx
import gudhi as gd
from scipy.stats import entropy
from scipy.special import kv, gamma

# Seed for reproducibility
rng = np.random.default_rng(42)

# --- Synthetic data generators ---
def sample_attributes(N, dataset="torus", noise=0.05):
    phi = rng.uniform(0, 2*np.pi, N)
    omega = rng.normal(0, 1, N)
    sigma = rng.choice([-1, 1], size=N)
    if dataset == "torus":
        u = rng.uniform(0, 2*np.pi, N)
        v = rng.uniform(0, 2*np.pi, N)
        phi = u % (2*np.pi)
        omega = np.sin(v) + noise * rng.normal(0, 1, N)
    return phi, omega, sigma

# --- Kernel pieces ---
def k_phi_1(delta_phi):
    return 1.0 + np.cos(delta_phi)

def k_phi_2(delta_phi):
    return 1.0 + np.cos(delta_phi) + 0.5 * np.cos(2 * delta_phi)

def k_phi_3(delta_phi):
    return 0.5 + 1.5 * np.cos(delta_phi)

def k_omega_rational(delta_omega, ell):
    return (1.0 + np.abs(delta_omega)/ell)**(-2)

def k_omega_exponential(delta_omega, ell):
    return np.exp(-np.abs(delta_omega)/ell)

def k_omega_polynomial(delta_omega, ell):
    return (1.0 + (np.abs(delta_omega)/ell)**2)**(-1)

def spin_pd_mask(sig):
    return (sig[:,None] == sig[None,:]).astype(float)

def spin_soft_same(sig, beta=1.0):
    return 0.5 * (1 + np.tanh(beta * sig[:,None] * sig[None,:]))

def spin_soft_opposite(sig, beta=1.0):
    return 0.5 * (1 - np.tanh(beta * sig[:,None] * sig[None,:]))

# --- Kernel flavors of Family A ---
def kernel_A1(phi, omega, sigma, ell=None):
    N = len(phi)
    if ell is None:
        ell = np.median(np.abs(omega[:,None] - omega[None,:]) + 1e-12)
    dphi = (phi[:,None] - phi[None,:] + np.pi) % (2*np.pi) - np.pi
    dw = omega[:,None] - omega[None,:]
    K = k_phi_1(dphi) * k_omega_rational(dw, ell) * spin_pd_mask(sigma)
    diag = np.sqrt(np.maximum(np.diag(K), 1e-12))
    K = (K / diag[:,None]) / diag[None,:]
    return K

def kernel_A2(phi, omega, sigma, ell=None, beta=1.5):
    N = len(phi)
    if ell is None:
        ell = np.median(np.abs(omega[:,None] - omega[None,:]) + 1e-12)
    dphi = (phi[:,None] - phi[None,:] + np.pi) % (2*np.pi) - np.pi
    dw = omega[:,None] - omega[None,:]
    K = k_phi_2(dphi) * k_omega_exponential(dw, ell) * spin_soft_same(sigma, beta)
    diag = np.sqrt(np.maximum(np.diag(K), 1e-12))
    K = (K / diag[:,None]) / diag[None,:]
    return K

def kernel_A3(phi, omega, sigma, ell=None, beta=1.5):
    N = len(phi)
    if ell is None:
        ell = np.median(np.abs(omega[:,None] - omega[None,:]) + 1e-12)
    dphi = (phi[:,None] - phi[None,:] + np.pi) % (2*np.pi) - np.pi
    dw = omega[:,None] - omega[None,:]
    K = k_phi_3(dphi) * k_omega_polynomial(dw, ell) * spin_soft_opposite(sigma, beta)
    diag = np.sqrt(np.maximum(np.diag(K), 1e-12))
    K = (K / diag[:,None]) / diag[None,:]
    return K

def kernel_legacy(phi, omega, sigma, delta_omega_star=1.0):
    N = len(phi)
    dphi = phi[:,None] - phi[None,:]
    dw = omega[:,None] - omega[None,:]
    A = 0.75 * (1 + np.cos(dphi)) * np.exp(-(dw/delta_omega_star)**2)
    gate = (1 - (sigma[:,None] * sigma[None,:]))
    R = A * gate
    diag = np.sqrt(np.maximum(np.diag(R) + 1e-9, 1e-9))
    R = (R / diag[:,None]) / diag[None,:]
    return np.maximum(R, 0.0)

# --- PD-induced metric & graph ---
def hilbert_metric(K):
    M = np.sqrt(np.clip(2.0 * (1.0 - np.clip(K, 0.0, 1.0)), 0.0, None))
    np.fill_diagonal(M, 0.0)
    return M

def threshold_graph(W, tau):
    A = (W >= tau).astype(int)
    np.fill_diagonal(A, 0)
    G = nx.from_numpy_array(A)
    return G, A

# --- Dimensions ---
def spectral_dimension(W, ts=(0.1, 0.05, 0.02, 0.01)):
    D = np.diag(W.sum(1))
    L = D - W
    evals = eigh(L)[0]
    Pts = [np.mean(np.exp(-t * evals)) for t in ts]
    x = np.log(np.array(ts))
    y = np.log(np.array(Pts))
    m = np.polyfit(x, y, 1)[0]
    return float(-2 * m)

def minkowski_dim(Dist, R_grid):
    N = Dist.shape[0]
    NRs = [(Dist <= R).sum(axis=1).mean() for R in R_grid]
    x = np.log(R_grid + 1e-9)
    y = np.log(np.array(NRs) + 1e-9)
    m = np.polyfit(x, y, 1)[0]
    return float(m)

# --- Embeddings & stress ---
def embedding_stress(Dist, k=3, method="mds", n_neighbors=10):
    try:
        if method == "mds":
            emb = MDS(n_components=k, dissimilarity='precomputed', n_init=1, max_iter=300, eps=1e-6).fit_transform(Dist)
            Demb = pairwise_distances(emb)
        else:
            iso = Isomap(n_neighbors=n_neighbors, n_components=k)
            emb = iso.fit_transform(Dist)
            Demb = pairwise_distances(emb)
        num = ((Dist - Demb)**2).sum()
        den = (Dist**2).sum() + 1e-12
        return float(num/den)
    except:
        return np.inf

# --- Cayley–Menger ---
def cm_tetra_volume_sq(D):
    CM = np.ones((5,5))
    CM[0,0] = 0
    CM[1:,1:] = D**2
    det = np.linalg.det(CM)
    return det / 288.0

# --- Ollivier curvature ---
def ollivier_curvature_approx(W, Dist):
    N = W.shape[0]
    deg = W.sum(1)
    curvs = []
    for i in range(N):
        for j in range(i+1, N):
            if W[i,j] <= 0: continue
            mu_i = W[i] / (deg[i] + 1e-12)
            mu_j = W[j] / (deg[j] + 1e-12)
            xi = (Dist[i] * mu_i).sum()
            xj = (Dist[j] * mu_j).sum()
            Wij = Dist[i,j] + 1e-9
            kappa = 1.0 - np.abs(xi - xj) / Wij
            curvs.append(kappa)
    return np.array(curvs) if curvs else np.array([0.0])

# --- Rank/CM concurrence ---
def rank_cm_concurrence(W, Dist, k=3, n_samples=100):
    N = W.shape[0]
    rank_matches = 0
    for _ in range(n_samples):
        idx = rng.choice(N, k+1, replace=False)
        D_sub = Dist[np.ix_(idx, idx)]
        W_sub = W[np.ix_(idx, idx)]
        D_v = (D_sub[1:] - D_sub[0]) / np.linalg.norm(D_sub[1:] - D_sub[0], axis=1)[:,None]
        G_v = D_v.T @ D_v
        rank = (svd(G_v, compute_uv=False) > 1e-6).sum()
        cm_vol = cm_tetra_volume_sq(D_sub) if k == 3 else 0
        cm_dim = 3 if cm_vol > 1e-6 else (2 if rank >= 2 else 1)
        rank_matches += (rank == cm_dim)
    return rank_matches / n_samples

# --- Persistent homology ---
def persistent_homology(W, tau_grid):
    N = W.shape[0]
    betti_1 = []
    for tau in tau_grid:
        G, A = threshold_graph(W, tau)
        st = gd.RipsComplex(distance_matrix=pairwise_distances(A)).create_simplex_tree(max_dimension=2)
        st.persistence()
        betti_1.append(sum(1 for dim, (birth, death) in st.persistence() if dim == 1 and death - birth > 0.1))
    return np.mean(betti_1)

# --- Main block for multiple trials ---
def run_trials(N=1000, dataset="torus", k=3, n_trials=20, seeds=None):
    if seeds is None:
        seeds = rng.integers(0, 10000, n_trials)
    kernels = ["legacy", "A1", "A2", "A3"]
    results = {kernel: [] for kernel in kernels}
    tau_grid = np.linspace(0.05, 0.95, 10)
    R_grid = np.geomspace(0.05, 20, 10)
    
    for seed in seeds:
        phi, omega, sigma = sample_attributes(N, dataset=dataset)
        for kernel in kernels:
            try:
                if kernel == "legacy":
                    K = kernel_legacy(phi, omega, sigma)
                elif kernel == "A1":
                    K = kernel_A1(phi, omega, sigma)
                elif kernel == "A2":
                    K = kernel_A2(phi, omega, sigma, beta=1.5)
                elif kernel == "A3":
                    K = kernel_A3(phi, omega, sigma, beta=1.5)
                D = hilbert_metric(K)
                stress = embedding_stress(D, k=k)
                ds = spectral_dimension(K)
                dm = minkowski_dim(D, R_grid)
                curvs = ollivier_curvature_approx(K, D)
                cond = np.linalg.cond(K + 1e-9 * np.eye(N))
                rank_cm = rank_cm_concurrence(K, D, k=k)
                betti_1 = persistent_homology(K, tau_grid)
                D_mat = np.diag(K.sum(1)) - K
                evals = eigh(D_mat)[0]
                gap_stability = np.var(evals[1:k+2] / (evals[:k+1] + 1e-12))
                percolation_width = np.std([spectral_dimension(nx.to_numpy_array(threshold_graph(K, tau)[0])) for tau in tau_grid])
                results[kernel].append({
                    "stress": stress,
                    "ds": ds,
                    "dm": dm,
                    "curv_mean": float(np.mean(curvs)),
                    "cond": cond,
                    "rank_cm": rank_cm,
                    "betti_1": betti_1,
                    "gap_stability": gap_stability,
                    "percolation_width": percolation_width
                })
            except Exception as e:
                print(f"Error in kernel {kernel}, seed {seed}: {e}")
                continue

    # Aggregate results
    summary = {kernel: {} for kernel in kernels}
    for kernel in kernels:
        data = results[kernel]
        if not data: continue
        summary[kernel] = {
            "stress_mean": np.mean([d["stress"] for d in data]),
            "stress_std": np.std([d["stress"] for d in data]),
            "ds_mean": np.mean([d["ds"] for d in data]),
            "ds_std": np.std([d["ds"] for d in data]),
            "dm_mean": np.mean([d["dm"] for d in data]),
            "dm_std": np.std([d["dm"] for d in data]),
            "curv_mean": np.mean([d["curv_mean"] for d in data]),
            "curv_std": np.std([d["curv_mean"] for d in data]),
            "cond_mean": np.mean([d["cond"] for d in data]),
            "cond_std": np.std([d["cond"] for d in data]),
            "rank_cm_mean": np.mean([d["rank_cm"] for d in data]),
            "rank_cm_std": np.std([d["rank_cm"] for d in data]),
            "betti_1_mean": np.mean([d["betti_1"] for d in data]),
            "betti_1_std": np.std([d["betti_1"] for d in data]),
            "gap_stability_mean": np.mean([d["gap_stability"] for d in data]),
            "gap_stability_std": np.std([d["gap_stability"] for d in data]),
            "percolation_width_mean": np.mean([d["percolation_width"] for d in data]),
            "percolation_width_std": np.std([d["percolation_width"] for d in data])
        }
    
    # Bayesian scoring
    sigma = {"stress": 0.1, "ds": 0.5, "dm": 0.5, "curv_mean": 0.2, "cond": 1000, "rank_cm": 0.1, "betti_1": 1.0, "gap_stability": 0.1, "percolation_width": 0.5}
    prior = {"legacy": 0.1, "A1": 0.4, "A2": 0.3, "A3": 0.2}  # Favor simplicity/PD
    scores = {kernel: 0.0 for kernel in kernels}
    for kernel in kernels:
        if not summary[kernel]: continue
        log_likelihood = 0.0
        for metric in ["stress", "ds", "dm", "curv_mean", "cond", "rank_cm", "betti_1", "gap_stability", "percolation_width"]:
            mean = summary[kernel][f"{metric}_mean"]
            log_likelihood -= (mean / sigma[metric])**2
        scores[kernel] = log_likelihood + np.log(prior[kernel])
    
    best_kernel = max(scores, key=scores.get)
    return summary, scores, best_kernel

# Run trials
if __name__ == "__main__":
    N = 1000
    k = 3
    n_trials = 20
    summary, scores, best_kernel = run_trials(N=N, dataset="torus", k=k, n_trials=n_trials)
    print("Summary of Results:")
    for kernel, metrics in summary.items():
        if not metrics: continue
        print(f"\nKernel: {kernel}")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.4f}")
    print("\nBayesian Scores:", {k: f"{v:.4f}" for k, v in scores.items()})
    print(f"Best Kernel: {best_kernel}")