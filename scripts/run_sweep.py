import subprocess
import numpy as np

cfgdir = "cfg_prod_rcut22"
L = 32
links = 18
a_eff = 0.0808


noise_values = np.round(np.arange(0.00, 0.71, 0.01), 2).tolist()

for noise in noise_values:
    cmd = [
        "python", "measure_observables.py",
        "--cfgdir", cfgdir,
        "--L", str(L),
        "--links", str(links),
        "--a_eff", str(a_eff),
        "--noise", str(noise)
    ]
    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
