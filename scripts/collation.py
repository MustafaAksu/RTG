import json, glob, re
from math import isnan
def parse(f):
    m = re.search(r'K_(\d+)_same_rho(\d+)p(\d+)_a2p55_b(\d+)p(\d+)_(tfine|smallt)\.json$', f)
    if not m: return None
    n = int(m.group(1)); rho = float(f'{m.group(2)}.{m.group(3)}')
    beta = float(f'{m.group(4)}.{m.group(5)}'); win = m.group(6)
    d = json.load(open(f))
    return n, rho, beta, win, d['spectral_dimension_mean'], d['spectral_dimension_se']
rows = [parse(f) for f in glob.glob('K_*_a2p55_b*_*.json')]
rows = [r for r in rows if r]
for n in sorted(set(r[0] for r in rows)):
    print(f'\n== n={n}')
    for win in ('smallt','tfine'):
        R = [r for r in rows if r[0]==n and r[3]==win and abs(r[2]-1.50)<1e-9]
        if R:
            for r in sorted(R, key=lambda x: x[1]):
                print(f'  {win}  rho={r[1]:.2f}  sdim={r[4]:.6f} ± {r[5]:.6f}')
