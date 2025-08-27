import pandas as pd
import re

raw_output = """
C:\GIT\rtg-resonance-2025\RTG\scripts>python run_sweep.py
Running: python measure_observables.py --cfgdir cfg_prod_rcut22 --L 32 --links 18 --a_eff 0.0808 --noise 0.0
Loaded 1500 configs  (snap format)
Proton circ‑radius ⟨r⟩ =   0.840 ±  0.000  fm
CHSH (σ=0.0)      =   2.828 ±  0.001
Running: python measure_observables.py --cfgdir cfg_prod_rcut22 --L 32 --links 18 --a_eff 0.0808 --noise 0.01
Loaded 1500 configs  (snap format)
Proton circ‑radius ⟨r⟩ =   0.840 ±  0.000  fm
CHSH (σ=0.01)      =   2.827 ±  0.001
Running: python measure_observables.py --cfgdir cfg_prod_rcut22 --L 32 --links 18 --a_eff 0.0808 --noise 0.02
Loaded 1500 configs  (snap format)
Proton circ‑radius ⟨r⟩ =   0.840 ±  0.000  fm
CHSH (σ=0.02)      =   2.826 ±  0.001
Running: python measure_observables.py --cfgdir cfg_prod_rcut22 --L 32 --links 18 --a_eff 0.0808 --noise 0.03
Loaded 1500 configs  (snap format)
Proton circ‑radius ⟨r⟩ =   0.840 ±  0.000  fm
CHSH (σ=0.03)      =   2.823 ±  0.001
Running: python measure_observables.py --cfgdir cfg_prod_rcut22 --L 32 --links 18 --a_eff 0.0808 --noise 0.04
Loaded 1500 configs  (snap format)
Proton circ‑radius ⟨r⟩ =   0.840 ±  0.000  fm
CHSH (σ=0.04)      =   2.820 ±  0.001
Running: python measure_observables.py --cfgdir cfg_prod_rcut22 --L 32 --links 18 --a_eff 0.0808 --noise 0.05
Loaded 1500 configs  (snap format)
Proton circ‑radius ⟨r⟩ =   0.840 ±  0.000  fm
CHSH (σ=0.05)      =   2.816 ±  0.002
Running: python measure_observables.py --cfgdir cfg_prod_rcut22 --L 32 --links 18 --a_eff 0.0808 --noise 0.06
Loaded 1500 configs  (snap format)
Proton circ‑radius ⟨r⟩ =   0.840 ±  0.000  fm
CHSH (σ=0.06)      =   2.810 ±  0.002
Running: python measure_observables.py --cfgdir cfg_prod_rcut22 --L 32 --links 18 --a_eff 0.0808 --noise 0.07
Loaded 1500 configs  (snap format)
Proton circ‑radius ⟨r⟩ =   0.840 ±  0.000  fm
CHSH (σ=0.07)      =   2.804 ±  0.002
Running: python measure_observables.py --cfgdir cfg_prod_rcut22 --L 32 --links 18 --a_eff 0.0808 --noise 0.08
Loaded 1500 configs  (snap format)
Proton circ‑radius ⟨r⟩ =   0.840 ±  0.000  fm
CHSH (σ=0.08)      =   2.797 ±  0.002
Running: python measure_observables.py --cfgdir cfg_prod_rcut22 --L 32 --links 18 --a_eff 0.0808 --noise 0.09
Loaded 1500 configs  (snap format)
Proton circ‑radius ⟨r⟩ =   0.840 ±  0.000  fm
CHSH (σ=0.09)      =   2.789 ±  0.003
Running: python measure_observables.py --cfgdir cfg_prod_rcut22 --L 32 --links 18 --a_eff 0.0808 --noise 0.1
Loaded 1500 configs  (snap format)
Proton circ‑radius ⟨r⟩ =   0.840 ±  0.000  fm
CHSH (σ=0.1)      =   2.780 ±  0.003
Running: python measure_observables.py --cfgdir cfg_prod_rcut22 --L 32 --links 18 --a_eff 0.0808 --noise 0.11
Loaded 1500 configs  (snap format)
Proton circ‑radius ⟨r⟩ =   0.840 ±  0.000  fm
CHSH (σ=0.11)      =   2.770 ±  0.004
Running: python measure_observables.py --cfgdir cfg_prod_rcut22 --L 32 --links 18 --a_eff 0.0808 --noise 0.12
Loaded 1500 configs  (snap format)
Proton circ‑radius ⟨r⟩ =   0.840 ±  0.000  fm
CHSH (σ=0.12)      =   2.760 ±  0.004
Running: python measure_observables.py --cfgdir cfg_prod_rcut22 --L 32 --links 18 --a_eff 0.0808 --noise 0.13
Loaded 1500 configs  (snap format)
Proton circ‑radius ⟨r⟩ =   0.840 ±  0.000  fm
CHSH (σ=0.13)      =   2.748 ±  0.005
Running: python measure_observables.py --cfgdir cfg_prod_rcut22 --L 32 --links 18 --a_eff 0.0808 --noise 0.14
Loaded 1500 configs  (snap format)
Proton circ‑radius ⟨r⟩ =   0.840 ±  0.000  fm
CHSH (σ=0.14)      =   2.736 ±  0.006
Running: python measure_observables.py --cfgdir cfg_prod_rcut22 --L 32 --links 18 --a_eff 0.0808 --noise 0.15
Loaded 1500 configs  (snap format)
Proton circ‑radius ⟨r⟩ =   0.840 ±  0.000  fm
CHSH (σ=0.15)      =   2.723 ±  0.008
Running: python measure_observables.py --cfgdir cfg_prod_rcut22 --L 32 --links 18 --a_eff 0.0808 --noise 0.16
Loaded 1500 configs  (snap format)
Proton circ‑radius ⟨r⟩ =   0.840 ±  0.000  fm
CHSH (σ=0.16)      =   2.709 ±  0.009
Running: python measure_observables.py --cfgdir cfg_prod_rcut22 --L 32 --links 18 --a_eff 0.0808 --noise 0.17
Loaded 1500 configs  (snap format)
Proton circ‑radius ⟨r⟩ =   0.840 ±  0.000  fm
CHSH (σ=0.17)      =   2.694 ±  0.011
Running: python measure_observables.py --cfgdir cfg_prod_rcut22 --L 32 --links 18 --a_eff 0.0808 --noise 0.18
Loaded 1500 configs  (snap format)
Proton circ‑radius ⟨r⟩ =   0.840 ±  0.000  fm
CHSH (σ=0.18)      =   2.678 ±  0.013
Running: python measure_observables.py --cfgdir cfg_prod_rcut22 --L 32 --links 18 --a_eff 0.0808 --noise 0.19
Loaded 1500 configs  (snap format)
Proton circ‑radius ⟨r⟩ =   0.840 ±  0.000  fm
CHSH (σ=0.19)      =   2.662 ±  0.015
Running: python measure_observables.py --cfgdir cfg_prod_rcut22 --L 32 --links 18 --a_eff 0.0808 --noise 0.2
Loaded 1500 configs  (snap format)
Proton circ‑radius ⟨r⟩ =   0.840 ±  0.000  fm
CHSH (σ=0.2)      =   2.646 ±  0.018
Running: python measure_observables.py --cfgdir cfg_prod_rcut22 --L 32 --links 18 --a_eff 0.0808 --noise 0.21
Loaded 1500 configs  (snap format)
Proton circ‑radius ⟨r⟩ =   0.840 ±  0.000  fm
CHSH (σ=0.21)      =   2.627 ±  0.020
Running: python measure_observables.py --cfgdir cfg_prod_rcut22 --L 32 --links 18 --a_eff 0.0808 --noise 0.22
Loaded 1500 configs  (snap format)
Proton circ‑radius ⟨r⟩ =   0.840 ±  0.000  fm
CHSH (σ=0.22)      =   2.610 ±  0.022
Running: python measure_observables.py --cfgdir cfg_prod_rcut22 --L 32 --links 18 --a_eff 0.0808 --noise 0.23
Loaded 1500 configs  (snap format)
Proton circ‑radius ⟨r⟩ =   0.840 ±  0.000  fm
CHSH (σ=0.23)      =   2.591 ±  0.026
Running: python measure_observables.py --cfgdir cfg_prod_rcut22 --L 32 --links 18 --a_eff 0.0808 --noise 0.24
Loaded 1500 configs  (snap format)
Proton circ‑radius ⟨r⟩ =   0.840 ±  0.000  fm
CHSH (σ=0.24)      =   2.570 ±  0.029
Running: python measure_observables.py --cfgdir cfg_prod_rcut22 --L 32 --links 18 --a_eff 0.0808 --noise 0.25
Loaded 1500 configs  (snap format)
Proton circ‑radius ⟨r⟩ =   0.840 ±  0.000  fm
CHSH (σ=0.25)      =   2.552 ±  0.031
Running: python measure_observables.py --cfgdir cfg_prod_rcut22 --L 32 --links 18 --a_eff 0.0808 --noise 0.26
Loaded 1500 configs  (snap format)
Proton circ‑radius ⟨r⟩ =   0.840 ±  0.000  fm
CHSH (σ=0.26)      =   2.529 ±  0.036
Running: python measure_observables.py --cfgdir cfg_prod_rcut22 --L 32 --links 18 --a_eff 0.0808 --noise 0.27
Loaded 1500 configs  (snap format)
Proton circ‑radius ⟨r⟩ =   0.840 ±  0.000  fm
CHSH (σ=0.27)      =   2.510 ±  0.039
Running: python measure_observables.py --cfgdir cfg_prod_rcut22 --L 32 --links 18 --a_eff 0.0808 --noise 0.28
Loaded 1500 configs  (snap format)
Proton circ‑radius ⟨r⟩ =   0.840 ±  0.000  fm
CHSH (σ=0.28)      =   2.487 ±  0.043
Running: python measure_observables.py --cfgdir cfg_prod_rcut22 --L 32 --links 18 --a_eff 0.0808 --noise 0.29
Loaded 1500 configs  (snap format)
Proton circ‑radius ⟨r⟩ =   0.840 ±  0.000  fm
CHSH (σ=0.29)      =   2.466 ±  0.048
Running: python measure_observables.py --cfgdir cfg_prod_rcut22 --L 32 --links 18 --a_eff 0.0808 --noise 0.3
Loaded 1500 configs  (snap format)
Proton circ‑radius ⟨r⟩ =   0.840 ±  0.000  fm
CHSH (σ=0.3)      =   2.445 ±  0.052
Running: python measure_observables.py --cfgdir cfg_prod_rcut22 --L 32 --links 18 --a_eff 0.0808 --noise 0.31
Loaded 1500 configs  (snap format)
Proton circ‑radius ⟨r⟩ =   0.840 ±  0.000  fm
CHSH (σ=0.31)      =   2.422 ±  0.055
Running: python measure_observables.py --cfgdir cfg_prod_rcut22 --L 32 --links 18 --a_eff 0.0808 --noise 0.32
Loaded 1500 configs  (snap format)
Proton circ‑radius ⟨r⟩ =   0.840 ±  0.000  fm
CHSH (σ=0.32)      =   2.398 ±  0.059
Running: python measure_observables.py --cfgdir cfg_prod_rcut22 --L 32 --links 18 --a_eff 0.0808 --noise 0.33
Loaded 1500 configs  (snap format)
Proton circ‑radius ⟨r⟩ =   0.840 ±  0.000  fm
CHSH (σ=0.33)      =   2.374 ±  0.065
Running: python measure_observables.py --cfgdir cfg_prod_rcut22 --L 32 --links 18 --a_eff 0.0808 --noise 0.34
Loaded 1500 configs  (snap format)
Proton circ‑radius ⟨r⟩ =   0.840 ±  0.000  fm
CHSH (σ=0.34)      =   2.350 ±  0.070
Running: python measure_observables.py --cfgdir cfg_prod_rcut22 --L 32 --links 18 --a_eff 0.0808 --noise 0.35
Loaded 1500 configs  (snap format)
Proton circ‑radius ⟨r⟩ =   0.840 ±  0.000  fm
CHSH (σ=0.35)      =   2.327 ±  0.078
Running: python measure_observables.py --cfgdir cfg_prod_rcut22 --L 32 --links 18 --a_eff 0.0808 --noise 0.36
Loaded 1500 configs  (snap format)
Proton circ‑radius ⟨r⟩ =   0.840 ±  0.000  fm
CHSH (σ=0.36)      =   2.305 ±  0.079
Running: python measure_observables.py --cfgdir cfg_prod_rcut22 --L 32 --links 18 --a_eff 0.0808 --noise 0.37
Loaded 1500 configs  (snap format)
Proton circ‑radius ⟨r⟩ =   0.840 ±  0.000  fm
CHSH (σ=0.37)      =   2.279 ±  0.088
Running: python measure_observables.py --cfgdir cfg_prod_rcut22 --L 32 --links 18 --a_eff 0.0808 --noise 0.38
Loaded 1500 configs  (snap format)
Proton circ‑radius ⟨r⟩ =   0.840 ±  0.000  fm
CHSH (σ=0.38)      =   2.255 ±  0.091
Running: python measure_observables.py --cfgdir cfg_prod_rcut22 --L 32 --links 18 --a_eff 0.0808 --noise 0.39
Loaded 1500 configs  (snap format)
Proton circ‑radius ⟨r⟩ =   0.840 ±  0.000  fm
CHSH (σ=0.39)      =   2.227 ±  0.097
Running: python measure_observables.py --cfgdir cfg_prod_rcut22 --L 32 --links 18 --a_eff 0.0808 --noise 0.4
Loaded 1500 configs  (snap format)
Proton circ‑radius ⟨r⟩ =   0.840 ±  0.000  fm
CHSH (σ=0.4)      =   2.202 ±  0.106
Running: python measure_observables.py --cfgdir cfg_prod_rcut22 --L 32 --links 18 --a_eff 0.0808 --noise 0.41
Loaded 1500 configs  (snap format)
Proton circ‑radius ⟨r⟩ =   0.840 ±  0.000  fm
CHSH (σ=0.41)      =   2.177 ±  0.109
Running: python measure_observables.py --cfgdir cfg_prod_rcut22 --L 32 --links 18 --a_eff 0.0808 --noise 0.42
Loaded 1500 configs  (snap format)
Proton circ‑radius ⟨r⟩ =   0.840 ±  0.000  fm
CHSH (σ=0.42)      =   2.153 ±  0.113
Running: python measure_observables.py --cfgdir cfg_prod_rcut22 --L 32 --links 18 --a_eff 0.0808 --noise 0.43
Loaded 1500 configs  (snap format)
Proton circ‑radius ⟨r⟩ =   0.840 ±  0.000  fm
CHSH (σ=0.43)      =   2.129 ±  0.121
Running: python measure_observables.py --cfgdir cfg_prod_rcut22 --L 32 --links 18 --a_eff 0.0808 --noise 0.44
Loaded 1500 configs  (snap format)
Proton circ‑radius ⟨r⟩ =   0.840 ±  0.000  fm
CHSH (σ=0.44)      =   2.099 ±  0.131
Running: python measure_observables.py --cfgdir cfg_prod_rcut22 --L 32 --links 18 --a_eff 0.0808 --noise 0.45
Loaded 1500 configs  (snap format)
Proton circ‑radius ⟨r⟩ =   0.840 ±  0.000  fm
CHSH (σ=0.45)      =   2.074 ±  0.133
Running: python measure_observables.py --cfgdir cfg_prod_rcut22 --L 32 --links 18 --a_eff 0.0808 --noise 0.46
Loaded 1500 configs  (snap format)
Proton circ‑radius ⟨r⟩ =   0.840 ±  0.000  fm
CHSH (σ=0.46)      =   2.051 ±  0.137
Running: python measure_observables.py --cfgdir cfg_prod_rcut22 --L 32 --links 18 --a_eff 0.0808 --noise 0.47
Loaded 1500 configs  (snap format)
Proton circ‑radius ⟨r⟩ =   0.840 ±  0.000  fm
CHSH (σ=0.47)      =   2.019 ±  0.148
Running: python measure_observables.py --cfgdir cfg_prod_rcut22 --L 32 --links 18 --a_eff 0.0808 --noise 0.48
Loaded 1500 configs  (snap format)
Proton circ‑radius ⟨r⟩ =   0.840 ±  0.000  fm
CHSH (σ=0.48)      =   1.999 ±  0.151
Running: python measure_observables.py --cfgdir cfg_prod_rcut22 --L 32 --links 18 --a_eff 0.0808 --noise 0.49
Loaded 1500 configs  (snap format)
Proton circ‑radius ⟨r⟩ =   0.840 ±  0.000  fm
CHSH (σ=0.49)      =   1.967 ±  0.160
Running: python measure_observables.py --cfgdir cfg_prod_rcut22 --L 32 --links 18 --a_eff 0.0808 --noise 0.5
Loaded 1500 configs  (snap format)
Proton circ‑radius ⟨r⟩ =   0.840 ±  0.000  fm
CHSH (σ=0.5)      =   1.942 ±  0.172
Running: python measure_observables.py --cfgdir cfg_prod_rcut22 --L 32 --links 18 --a_eff 0.0808 --noise 0.51
Loaded 1500 configs  (snap format)
Proton circ‑radius ⟨r⟩ =   0.840 ±  0.000  fm
CHSH (σ=0.51)      =   1.914 ±  0.178
Running: python measure_observables.py --cfgdir cfg_prod_rcut22 --L 32 --links 18 --a_eff 0.0808 --noise 0.52
Loaded 1500 configs  (snap format)
Proton circ‑radius ⟨r⟩ =   0.840 ±  0.000  fm
CHSH (σ=0.52)      =   1.898 ±  0.178
Running: python measure_observables.py --cfgdir cfg_prod_rcut22 --L 32 --links 18 --a_eff 0.0808 --noise 0.53
Loaded 1500 configs  (snap format)
Proton circ‑radius ⟨r⟩ =   0.840 ±  0.000  fm
CHSH (σ=0.53)      =   1.870 ±  0.189
Running: python measure_observables.py --cfgdir cfg_prod_rcut22 --L 32 --links 18 --a_eff 0.0808 --noise 0.54
Loaded 1500 configs  (snap format)
Proton circ‑radius ⟨r⟩ =   0.840 ±  0.000  fm
CHSH (σ=0.54)      =   1.835 ±  0.197
Running: python measure_observables.py --cfgdir cfg_prod_rcut22 --L 32 --links 18 --a_eff 0.0808 --noise 0.55
Loaded 1500 configs  (snap format)
Proton circ‑radius ⟨r⟩ =   0.840 ±  0.000  fm
CHSH (σ=0.55)      =   1.823 ±  0.204
Running: python measure_observables.py --cfgdir cfg_prod_rcut22 --L 32 --links 18 --a_eff 0.0808 --noise 0.56
Loaded 1500 configs  (snap format)
Proton circ‑radius ⟨r⟩ =   0.840 ±  0.000  fm
CHSH (σ=0.56)      =   1.791 ±  0.210
Running: python measure_observables.py --cfgdir cfg_prod_rcut22 --L 32 --links 18 --a_eff 0.0808 --noise 0.57
Loaded 1500 configs  (snap format)
Proton circ‑radius ⟨r⟩ =   0.840 ±  0.000  fm
CHSH (σ=0.57)      =   1.761 ±  0.209
Running: python measure_observables.py --cfgdir cfg_prod_rcut22 --L 32 --links 18 --a_eff 0.0808 --noise 0.58
Loaded 1500 configs  (snap format)
Proton circ‑radius ⟨r⟩ =   0.840 ±  0.000  fm
CHSH (σ=0.58)      =   1.753 ±  0.224
Running: python measure_observables.py --cfgdir cfg_prod_rcut22 --L 32 --links 18 --a_eff 0.0808 --noise 0.59
Loaded 1500 configs  (snap format)
Proton circ‑radius ⟨r⟩ =   0.840 ±  0.000  fm
CHSH (σ=0.59)      =   1.716 ±  0.235
Running: python measure_observables.py --cfgdir cfg_prod_rcut22 --L 32 --links 18 --a_eff 0.0808 --noise 0.6
Loaded 1500 configs  (snap format)
Proton circ‑radius ⟨r⟩ =   0.840 ±  0.000  fm
CHSH (σ=0.6)      =   1.700 ±  0.226
Running: python measure_observables.py --cfgdir cfg_prod_rcut22 --L 32 --links 18 --a_eff 0.0808 --noise 0.61
Loaded 1500 configs  (snap format)
Proton circ‑radius ⟨r⟩ =   0.840 ±  0.000  fm
CHSH (σ=0.61)      =   1.664 ±  0.243
Running: python measure_observables.py --cfgdir cfg_prod_rcut22 --L 32 --links 18 --a_eff 0.0808 --noise 0.62
Loaded 1500 configs  (snap format)
Proton circ‑radius ⟨r⟩ =   0.840 ±  0.000  fm
CHSH (σ=0.62)      =   1.647 ±  0.247
Running: python measure_observables.py --cfgdir cfg_prod_rcut22 --L 32 --links 18 --a_eff 0.0808 --noise 0.63
Loaded 1500 configs  (snap format)
Proton circ‑radius ⟨r⟩ =   0.840 ±  0.000  fm
CHSH (σ=0.63)      =   1.616 ±  0.251
Running: python measure_observables.py --cfgdir cfg_prod_rcut22 --L 32 --links 18 --a_eff 0.0808 --noise 0.64
Loaded 1500 configs  (snap format)
Proton circ‑radius ⟨r⟩ =   0.840 ±  0.000  fm
CHSH (σ=0.64)      =   1.602 ±  0.264
Running: python measure_observables.py --cfgdir cfg_prod_rcut22 --L 32 --links 18 --a_eff 0.0808 --noise 0.65
Loaded 1500 configs  (snap format)
Proton circ‑radius ⟨r⟩ =   0.840 ±  0.000  fm
CHSH (σ=0.65)      =   1.589 ±  0.246
Running: python measure_observables.py --cfgdir cfg_prod_rcut22 --L 32 --links 18 --a_eff 0.0808 --noise 0.66
Loaded 1500 configs  (snap format)
Proton circ‑radius ⟨r⟩ =   0.840 ±  0.000  fm
CHSH (σ=0.66)      =   1.562 ±  0.272
Running: python measure_observables.py --cfgdir cfg_prod_rcut22 --L 32 --links 18 --a_eff 0.0808 --noise 0.67
Loaded 1500 configs  (snap format)
Proton circ‑radius ⟨r⟩ =   0.840 ±  0.000  fm
CHSH (σ=0.67)      =   1.541 ±  0.276
Running: python measure_observables.py --cfgdir cfg_prod_rcut22 --L 32 --links 18 --a_eff 0.0808 --noise 0.68
Loaded 1500 configs  (snap format)
Proton circ‑radius ⟨r⟩ =   0.840 ±  0.000  fm
CHSH (σ=0.68)      =   1.512 ±  0.281
Running: python measure_observables.py --cfgdir cfg_prod_rcut22 --L 32 --links 18 --a_eff 0.0808 --noise 0.69
Loaded 1500 configs  (snap format)
Proton circ‑radius ⟨r⟩ =   0.840 ±  0.000  fm
CHSH (σ=0.69)      =   1.497 ±  0.276
Running: python measure_observables.py --cfgdir cfg_prod_rcut22 --L 32 --links 18 --a_eff 0.0808 --noise 0.7
Loaded 1500 configs  (snap format)
Proton circ‑radius ⟨r⟩ =   0.840 ±  0.000  fm
CHSH (σ=0.7)      =   1.477 ±  0.280

"""

data = re.findall(r"σ=([\d\.]+)\)\s+=\s+([\d\.]+)\s+±\s+([\d\.]+)", raw_output)
df = pd.DataFrame(data, columns=["sigma", "CHSH", "error"]).astype(float)
df.to_csv('chsh_data.csv', index=False)
