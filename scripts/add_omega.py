# add_omega.py
import numpy as np, argparse

p = argparse.ArgumentParser()
p.add_argument("--in", dest="inp", required=True)
p.add_argument("--out", dest="out", required=True)
p.add_argument("--mode", choices=["phi","phase","x0"], default="phi",
               help="where to read angles from; x0 uses first column of x")
args = p.parse_args()

d = np.load(args.inp)
keys = set(d.keys())

if "omega" in keys:
    raise SystemExit("Input already has 'omega'")

if args.mode == "phi" and "phi" in keys:
    ang = d["phi"].astype(float)
elif args.mode == "phase" and "phase" in keys:
    ang = d["phase"].astype(float)
elif args.mode == "x0" and "x" in keys:
    x = d["x"].astype(float)
    ang = np.asarray(x[:,0], dtype=float)
else:
    raise SystemExit("Requested source not found in NPZ")

# unwrap to get a smooth 1D “frequency-like” coordinate
omega = np.unwrap(ang)

# write all original arrays + omega
outdict = {k: d[k] for k in d.files}
outdict["omega"] = omega
np.savez(args.out, **outdict)
print({"written": args.out, "n": int(omega.shape[0])})
