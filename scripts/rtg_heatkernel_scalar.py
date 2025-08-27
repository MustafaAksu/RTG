import numpy as np, scipy.sparse as sp, scipy.sparse.linalg as spla

def laplacian_4d(L, eps=0.0):
    """Sparse 4‑D laplacian with tiny curvature eps."""
    V = L**4
    rows, cols, data = [], [], []
    def idx(t,x,y,z):  # periodic index
        return (((t % L)*L + x % L)*L + y % L)*L + z % L

    for t in range(L):
        for x in range(L):
            for y in range(L):
                for z in range(L):
                    i = idx(t,x,y,z)
                    diag = 0.0
                    for dt,dx,dy,dz in [(1,0,0,0),(-1,0,0,0),
                                        (0,1,0,0),(0,-1,0,0),
                                        (0,0,1,0),(0,0,-1,0),
                                        (0,0,0,1),(0,0,0,-1)]:
                        j = idx(t+dt,x+dx,y+dy,z+dz)
                        rows.append(i); cols.append(j); data.append(-1.0*(1-eps))
                        diag += 1.0*(1-eps)
                    rows.append(i); cols.append(i); data.append(8.0*(1-eps))
    return sp.csr_matrix((data,(rows,cols)),shape=(V,V))

def logdet(lap, n_ev=400, m_shift=1e-6):
    """
    Compute log det by Lanczos on the lowest n_ev modes after
    removing zero‑mode via a small positive shift (m_shift).
    """
    # shift ensures strictly positive spectrum
    lap_shift = lap + m_shift * sp.eye(lap.shape[0])
    vals = spla.eigsh(lap_shift, k=n_ev, which='SM', return_eigenvectors=False)
    # keep only positive values (numerical safety)
    vals = vals[vals > 0]
    return np.sum(np.log(vals))

if __name__ == "__main__":
    L = 8            # 8^4 = 4096 sites   (quick test)
    eps_list = [0.0, 1e-4, 2e-4, 3e-4]
    logs = []
    for eps in eps_list:
        lap = laplacian_4d(L, eps)
        ld  = logdet(lap, n_ev=600)
        logs.append(ld)
        print(f"eps={eps:.2e}  logdet={ld: .6e}")
    # finite‑difference slope  d/d eps at eps=0
    slope = (logs[1] - logs[0]) / (eps_list[1]-eps_list[0])
    print(f"\nFinite‑diff slope  d(log det)/dε ≈ {slope:.3e}")
