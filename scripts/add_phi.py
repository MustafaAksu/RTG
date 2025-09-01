# add_phi.py
import argparse, numpy as np

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--infile", required=True)
    ap.add_argument("--outfile", required=True)
    ap.add_argument("--seed", type=int, default=71)
    ap.add_argument("--dist", choices=["uniform","gaussian"], default="uniform")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    d = dict(np.load(args.infile))
    n = None
    # infer n from any 1D attribute we already have
    for k,v in d.items():
        if getattr(v, "ndim", 0)==1:
            n = len(v); break
    if n is None:
        raise SystemExit("Could not infer n from infile; please include at least one 1D array.")

    if "phi" in d:
        print(f"[info] infile already has 'phi' (n={len(d['phi'])}); overwriting with new draw.")
    if args.dist == "uniform":
        phi = rng.uniform(0.0, 2.0*np.pi, size=n)
    else:
        # center 0, wrap to [0,2π)
        phi = (rng.normal(0.0, 1.0, size=n) % (2.0*np.pi)).astype(float)

    d["phi"] = phi
    np.savez(args.outfile, **d)
    print(f"[ok] wrote {args.outfile} with phi (radians) of length {n}")

if __name__ == "__main__":
    main()
