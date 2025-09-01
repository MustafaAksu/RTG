# corr2pt.py  —  estimate radial two‑point correlation of omega vs graph distance
# Usage (PowerShell):
#   python .\corr2pt.py --kernel K_4096_same_rho1p16_a2p55_b1p50.npy `
#                       --attrs attrs_4096.npz `
#                       --quantile 0.998 `
#                       --maxdist 12 `
#                       --out corr_omega_4096_k1.json

import argparse, json, numpy as np
from collections import deque, defaultdict

def top_edges_threshold(K, quantile):
    w = K[np.triu_indices_from(K,1)]
    tau = np.quantile(w[w>0], quantile)
    return tau

def build_unweighted_adj(K, tau):
    n = K.shape[0]
    nbrs = [[] for _ in range(n)]
    I,J = np.where((K >= tau) & (np.triu(np.ones_like(K),1) > 0))
    for i,j in zip(I,J):
        nbrs[i].append(j); nbrs[j].append(i)
    return nbrs

def bfs_dists(nbrs, src, maxdist):
    n = len(nbrs)
    dist = [-1]*n; dist[src]=0
    dq = deque([src])
    while dq:
        u = dq.popleft()
        if dist[u] == maxdist: continue
        for v in nbrs[u]:
            if dist[v] == -1:
                dist[v] = dist[u]+1
                dq.append(v)
    return dist

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kernel", required=True)
    ap.add_argument("--attrs",  required=True)
    ap.add_argument("--quantile", type=float, default=0.998)
    ap.add_argument("--maxdist", type=int, default=12)
    ap.add_argument("--samples", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=71)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    K = np.load(args.kernel)
    attrs = np.load(args.attrs)
    if "omega" not in attrs: raise SystemExit("attrs must contain 'omega'")
    x = attrs["omega"].astype(float)
    x = (x - x.mean()) / (x.std() + 1e-12)  # normalize

    tau = top_edges_threshold(K, args.quantile)
    nbrs = build_unweighted_adj(K, tau)

    n = K.shape[0]
    rs = np.random.RandomState(args.seed)
    idx = rs.choice(n, size=min(args.samples,n), replace=False)

    # accumulate correlations by graph distance
    sums = defaultdict(float); counts = defaultdict(int)
    for s in idx:
        dist = bfs_dists(nbrs, s, args.maxdist)
        for t, d in enumerate(dist):
            pass
        for t in range(n):
            d = dist[t]
            if d >= 0:
                sums[d] += x[s]*x[t]
                counts[d] += 1

    out = []
    for d in sorted(k for k in counts.keys() if k>0):
        out.append({"r": int(d),
                    "corr": float(sums[d]/counts[d]),
                    "pairs": int(counts[d])})

    with open(args.out,"w") as f:
        json.dump({
          "kernel": args.kernel,
          "attrs": args.attrs,
          "quantile": args.quantile,
          "maxdist": args.maxdist,
          "samples": args.samples,
          "corr_vs_r": out
        }, f, indent=2)
    print(f"[OK] wrote {args.out}")
if __name__ == "__main__":
    main()
