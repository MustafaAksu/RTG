import json, sys, numpy as np

def sdim_from_heat(json_file):
    d = json.load(open(json_file))
    P = np.array(d["heat_meta"]["heat_trace"], dtype=float)
    # Guess the t-grid from the file; tfine contains 13 points, smallt contains 8
    # We rely on the d["heat_meta"]["t_grid"] if present (your files include it).
    T = np.array(d["heat_meta"]["t_grid"], dtype=float)
    # finite-difference derivative of log P wrt log t
    lp = np.log(P); lt = np.log(T)
    sdim = -2.0 * np.gradient(lp, lt, edge_order=2)
    return T, sdim

if __name__ == "__main__":
    jf = sys.argv[1] if len(sys.argv)>1 else "K_4096_same_rho1p16_a2p55_b1p50_smallt.json"
    T, sdim = sdim_from_heat(jf)
    print("t, sdim_inst")
    for t, s in zip(T, sdim):
        print(f"{t:.6g}, {s:.9f}")
