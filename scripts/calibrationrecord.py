import json, glob, csv, os, re
from collections import defaultdict

rows = []
# 1) spectral: pick the *small-t* jsons at K1
for f in glob.glob("K_*_same_rho1p16_a2p55_b1p50_smallt.json"):
    with open(f) as fh:
        d = json.load(fh)
    m = re.search(r'^K_(\d+)_', d["kernel_file"])
    n = int(m.group(1)) if m else None
    rows.append({
        "n": n, "section":"spectral", "tau":"small-t",
        "metric":"sdim", "mean": d["spectral_dimension_mean"], "se": d["spectral_dimension_se"]
    })

# 2) curvature summaries (ollivier/forman) if you produced them
def pull_curv(prefix, method):
    # try both fixed-tau and level-based outputs you used earlier
    for f in glob.glob(f"{prefix}_{method}_summary.json") + glob.glob(f"{prefix}_{method}.json"):
        try:
            with open(f) as fh: d = json.load(fh)
        except Exception: continue
        n = d.get("n") or d.get("meta",{}).get("n")
        tau = d.get("tau") or d.get("level_tau") or d.get("meta",{}).get("tau")
        if method=="oll":
            mu = d.get("edge_mean", d.get("ollivier",{}).get("mean"))
            sd = d.get("edge_std",  d.get("ollivier",{}).get("std"))
            rows.append({"n":n,"section":"curvature","tau":tau,"metric":"ollivier_edge_mean","mean":mu,"se":None})
        if method=="forman":
            muE = d.get("edge_mean"); muV = d.get("node_mean")
            rows.append({"n":n,"section":"curvature","tau":tau,"metric":"forman_edge_mean","mean":muE,"se":None})
            rows.append({"n":n,"section":"curvature","tau":tau,"metric":"forman_node_mean","mean":muV,"se":None})

# adapt prefixes you used (examples you shared): curv_4096_oll_summary.json, curv_4096_forman_summary.json
pull_curv("curv_4096","oll")
pull_curv("curv_4096","forman")
pull_curv("curv_8192","oll")
pull_curv("curv_8192","forman")

# 3) homology: summarize if present
for f in glob.glob("homology_*_summary.json"):
    with open(f) as fh: d = json.load(fh)
    n = d.get("n")
    # prefer a representative low tau (e.g., smallest tau row)
    recs = d.get("levels", d.get("rows", []))
    if not recs: continue
    rec = min(recs, key=lambda r: r.get("tau", 1.0))
    rows.append({"n":n,"section":"homology","tau":rec.get("tau"),
                 "metric":"triangles", "mean": rec.get("triangles"), "se":None})
    rows.append({"n":n,"section":"homology","tau":rec.get("tau"),
                 "metric":"clique_edges", "mean": rec.get("edges"), "se":None})
    if "betti1" in rec:
        rows.append({"n":n,"section":"homology","tau":rec.get("tau"),
                     "metric":"betti1", "mean": rec.get("betti1"), "se":None})

# write CSV
rows.sort(key=lambda r: (r["section"], r["n"], str(r["metric"])))
with open("K1_calibration_summary.csv","w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=["n","section","tau","metric","mean","se"])
    w.writeheader(); w.writerows(rows)

print("Wrote K1_calibration_summary.csv with", len(rows), "rows")
