import json, glob, math, numpy as np
def sdim_phys(t, P):  # vectors
    y = np.log(np.maximum(P-1.0, 1e-15))
    x = np.log(t)
    # local slopes by central differences
    dx = np.diff(x); dy = np.diff(y)
    slope = np.concatenate(([dy[0]/dx[0]], (dy[1:]-dy[:-1])/(dx[1:]-dx[:-1]+1e-15), [dy[-1]/dx[-1]]))
    return -2.0*slope

for f in glob.glob("K_*_smallt.json"):
    d = json.load(open(f))
    t = np.array(d["heat_meta"]["t_grid"], float)
    P = np.array(d["heat_meta"]["heat_trace"], float)
    ds = sdim_phys(t, P)
    print(f, "IR plateau ~", ds[-4:].mean(), "UV ~", ds[:4].mean())
