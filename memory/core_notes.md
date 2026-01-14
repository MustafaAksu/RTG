# Relational Time Geometry (RTG) — Core Notes (Concise)

**Version**: v1.17 (Lattice Geometry & Frustration Synthesis)
**Last Revised**: 2026-01-09
**Authors**: Mustafa Aksu, Claude, Grok, ChatGPT, Gemini
**Purpose**: Tier‑1 **authoritative** reference for RTG principles, constants, and equations. Stable anchors for RAG; Tier‑2 dynamics live in `findings.yaml`.

---

## 0. Navigation

- §1 **Constants** (core → EFT → particle → cosmology)
- §2 **Memory Architecture** (how Tier‑1 anchors Tier‑2/3)
- §3 **Core Mechanics** (nodes/edges, clocks)
- §4 **Glossary** (resonance, mapping, gauge, gravity, thermo, special nodes)
- §5 **Core Principles** (unification, observer, thermodynamics, particle & cosmology summaries)
- §6 **Key Equations** (microscopic, continuum/EFT, RG, thermo, particle, cosmology)
- §7 **Benchmarks** (CHSH, proton/Ca‑40, CMB, BH tail)
- §7.5 **ESM Benchmarks** (vacuum growth, Hodge decomposition, quantum foam)
- §8 **Applications & Open Questions**
- §9 **Maintenance & Conventions**
- **§10 Synthesis: The Frustrated Simplicial Vacuum** *(New)*

---

## 1. Key Constants  *(calibrated via RG/MD; see Tier‑2 for evolving fits)*

### 1.1 Core physical / geometric
| Symbol | Value | Units | Notes |
|---|---:|---|---|
| \( \Delta\omega^* \) | \(1.45(8)\times10^{23}\) | rad·s⁻¹ | Critical bandwidth (RG‑fixed). |
| \( c \) | 299792458 | m·s⁻¹ | Speed of light. |
| \( \hbar \) | \(1.0545718\times10^{-34}\) | J·s | Reduced Planck constant. |
| Proton radius | \(0.84\pm0.01\) | fm | Calibration target. |
| \( g^* \) | \( \approx 1.14\pm0.02 \) | – | Scalar RG fixed point. |
| \( \sigma_{\rm crit} \) | \( \approx 0.589 \) | – | CHSH decoherence threshold (σ_noise/Δω*). |
| \( \ell^* \) | \(2.07\pm0.11\) | fm | Spectral length \(c/\Delta\omega^*\). |
| \( r^* \) | \(13.0\pm0.7\) | fm | Beat length \(2\pi\ell^*\). |
| **\( \delta_\theta \)** | **\( \approx 7.36^\circ \)** | deg | **Tetrahedral angular deficit (360° - 5×70.53°).** |
| \( f_{\mathrm{irr}} \) | 0.752 | – | Hodge‑irreducible fraction of \(\mathrm{Var}(z)\) (coexact + harmonic) in ESM. |
| \( \eta_Q \) | 0.977 | – | Boundary‑charge screening efficiency in ESM (example: \(Q: -1.60 \to -0.04\)). |
| \( \lvert z \rvert_{\mathrm{floor}} \) | ~0.46 | – | Median \(\lvert z \rvert\) after IRLS fitting (ESM “geometric floor”). |

### 1.2 Couplings / elasticities
| Symbol | Value | Units | Notes |
|---|---:|---|---|
| \( K' \) | 12.0 (±0.5%) | MeV | Frequency penalty scale. |
| \( J \) | \(3.24\pm0.12\) | MeV | Resonance coupling. |
| \( J_{\rm ex} \) | \(2.20\pm0.08\) | MeV | Exchange coupling. |
| \( \alpha \) | \( \approx 938 \) | MeV·fm² | Mass form factor. |
| \( \kappa_c \) | 1 | MeV·fm | Curvature scale. |
| \( \rho_s \) | \( \approx 15.9 \) | MeV·fm⁻¹ | Phase stiffness (a_lat = 0.08 fm). |
| \( \kappa_B \) | ~1 | MeV·fm | Gauge stiffness; EFT curl \(B\). |
| \( C_\kappa^\infty \) (U(1), SU(2), U(1)²) | 0.009116 / 0.009104 / 0.009115 | – | Window‑averaged Wilson coefficients. |
| Proton binding | \( \approx 48\pm3 \) | MeV | 3‑node equilibrium. |
| **\( \sigma_{\rm exch} \)** | **O(\(\Delta\omega^*\))** | rad·s⁻¹ | **UV regulator; independent of σ_noise.** |
| **\( T_{\rm spec} \)** | **\( \hbar\sigma_\omega/k_B\)** | K | **Spectral temperature (frame‑invariant).** |
| \( E_0 \) | \(\approx 1/12\) | – | Thermodynamic Floor (variance tax). |

### 1.3 Gravity & running
| Symbol | Value | Units | Notes |
|---|---:|---|---|
| \( G_0 \) | \( \sim \rho_s \ell^{*2} \) | m³·kg⁻¹·s⁻² | Newton (tree‑level, provisional). |
| \( \Lambda_0 \) | TBD | m⁻² | Coarse‑grained cosmological constant. |
| \( B_1 \) | ~0.1–0.2 | – | \( \beta_{g_N} \) coeff (scheme‑dep.). |
| \( C_0,C_2,C_4 \) | TBD | – | \( \beta_\Lambda \) coeffs. |
| \( e \) | ~1/137 | – | U(1) charge (provisional). |

### 1.4 Particle‑scale (from Particle & Nuclear v2.4)
| Symbol | Value | Units | Notes |
|---|---:|---|---|
| \( \omega_q \) | \(2.51\times10^{23}\) | rad·s⁻¹ | Quark‑level frequency. |
| \( \omega_{q,sub} \) | \(1.26\times10^{20}\) | rad·s⁻¹ | Sub‑node oscillator. |
| \( r_{\rm sub} \) | \(5\times10^{-17}\) | fm | Sub‑node spacing (gluon‑like layer). |
| \( \beta \) | \( \approx 10^{-3} \) | – | RG‑scaled sub‑node flip factor. |

### 1.5 Cosmology‑scale (from Cosmological v2.8)
| Symbol | Value | Units | Notes |
|---|---:|---|---|
| \( H_0 \) | \(68\pm2\) | km·s⁻¹·Mpc⁻¹ | Expansion from ε‑drift. |
| \( t_0 \) | \( \approx 13.7 \) | Gyr | Age from recombination freeze‑out. |
| \( r \) | < 0.01 | – | Tensor/scalar ratio (prediction). |
| \( \sigma_\phi \) | \( \approx 1\times10^{-4} \) | – | CMB phase scatter (lattice). |

---

## 2. Memory Architecture (Tier‑1 → Tier‑2/3)
- **Tier‑1** (this file): stable primitives, constants, canonical equations.
- **Tier‑2** (`findings.yaml`): experiments, thresholds, predictions (append‑only).
- **Tier‑3**: articles, notebooks, code; referenced via Tier‑2.
- **Guardrails**: periodic Tier‑1 refresh to limit multi‑session drift (>0.1% without re‑anchor).

---

## 3. Core Mechanics
- **Nodes**: \( \omega_i, \phi_i, \), spin (analytic \(s_i=\pm i\); code \( \sigma_i=\pm1\)); intrinsic time \( t_i=\tilde\phi/\omega_i \).
- **Edges**: weighted by resonance; gauge link \( U_{ij} \).
- **Observers**: Δω*‑observer frames for comparisons; **superposition is relational** (unresolved phases until interaction).
- **Tooling**: lattice MD/HMC; continuum/EFT post‑processing.

### 3.5 ESM (Emergent Simplicial Manifold)

ESM is a constructive “emergent geometry” module built on top of an underlying oscillator graph. A **cluster** is a set of triangular faces (“triads”) glued along edges.

- **Edge curvature field**: each graph edge \(e=(i,j)\) carries a residual \(z_e\) computed from frequency‑derived and embedding‑derived distances.
- **Boundary definition**: the boundary \(\partial\mathcal{M}\) is the set of edges incident to exactly one triad in the cluster.
- **Dual boundary observables** (independent):
  - **Boundary charge** \(Q_{\partial} = \sum_{e \in \partial\mathcal{M}} z_e\) (Stokes analog).
  - **Boundary roughness** \(C_{\partial} = \sqrt{\left\langle z_e^2 \right\rangle_{e \in \partial\mathcal{M}}}\).
- **Flux‑balanced growth**: attaching a triad updates charge by \( \Delta Q_{\partial} = -z_{\rm old} + z_{\rm new,1} + z_{\rm new,2} \).
- **Breathing mechanism**: strict monotone minimization of \(\lvert Q_{\partial} \rvert\) can stall. Growth requires **inertial breathing** (controlled temporary increases in \(\lvert Q_{\partial} \rvert\)) to bypass geometric bottlenecks.
- **Refinement trade‑off**: aggressive smoothing can fragment connectivity (\(\beta_0 > 1\)). This corresponds to **Inertial Fracture** (escape from the constraint manifold).

---

## 4. Glossary

### 4.1 Resonance & interaction metrics
- **Resonance kernel** \( \mathcal R_{ij}=\frac{3}{4}[1+\cos(\phi_i-\phi_j)](1+s_is_j)\exp[-(\Delta\omega/\Delta\omega^*)^2] \).
- **Beat distance** \( r=2\pi c/|\Delta\omega| \).
- **Gate** \( G(x) \) (0–1 stiffness field).
- **Whirling frequency**: Local precession scale \( \Omega_{whirl} = \sum \sin(\phi_i - \phi_j) \) (code units); used to characterize vorticity and orbital fits.

### 4.2 Continuum mapping
- **Geometric Mapping Warning**: Analytic cubic factors (e.g., \( \rho_s=(3/2)J \)) are invalid for ESM. Use **Test-Field Calibration** (§10.4).
- TV penalty \( K'_{\rm TV}\|\nabla\omega\|_1/\Delta\omega^* \).
- Small‑angle bridge \( \sum G_{ij}(\Delta\phi)^2 \to a^{2-d}\!\int G|\nabla\phi|^2 \).

### 4.3 Dimensionality windows (δω/Δω*)
| Range | Effective dim | Regime / symmetry |
|---|---|---|
| < 0.28 (±0.02) | 2D | U(1) sheets |
| 0.28–0.70 | 3D | SU(2) shells |
| 0.70–1.55 | 4D | Hyper‑corridors |
| 1.55–1.70 | Transitional | U(1)² phase shells (ε≈SU(3)) |
| ≥ 1.70 | >5D | Dim‑6 anomalies? |

> **Diagnostic**: The stability of the 2D→3D transition is strictly defined by the **Cayley–Menger** tetrahedral volume \( V_{CM} \). The lattice is considered 3D-stable when \( V_{CM} \) rises non-linearly beyond the threshold at \( \delta\omega > 0.28 \).

### 4.4 Gauge symmetries (unified)
- **U(1)**: \( B_{ij}=aA_{ij} \), \( U_{ij}=e^{i(\Delta\phi_{ij}-B_{ij})} \).
- **SU(2)**: \( \psi_i=[z_i, i s_i z_i]^T \), rotations \( R(\theta) \); **Witten anomaly** → even doublets.
- **U(1)² ≈ SU(3)**: three phases with one constraint → two independent; ε‑coupling approximates strong sector; **validated by proton binding**.
- **Exchange**: \( \sin(\Delta\phi-B) \) ≡ \(-\cos(\Delta\phi-B-\pi/2)\).
- **Anomalies**: U(1) charge/cubic sums = 0; SU(2) perturbative \(d^{abc}=0\) (global Witten persists).

### 4.5 Emergent gravity (tree‑level)
- **Structure tensor** \( S_{\mu\nu}=(1/\mathcal N)\sum w_{ij}\Delta x_{ij,\mu}\Delta x_{ij,\nu} \) (w = ℛ or adjacency).
- **Tetrad** from \( S \) eigenvectors; metric \( g_{\mu\nu}=\Omega^2\eta_{ab}e^a{}_\mu e^b{}_\nu \).
- EH action \( S_{\rm grav}=(1/16\pi G_0)\int\sqrt{-g}(R-2\Lambda_0) \).
- Matter sector \( S_{\phi,G}=\int\sqrt{-g}[(\rho_s/2)G g^{\mu\nu}\partial_\mu\phi\partial_\nu\phi + \kappa(1-G)] \).
- U(1) minimal coupling: \( B\to A_\mu \), \( S_B\!\to\! F_{\mu\nu} \).

### 4.6 Quantum gravity & running
- Dual scales: \( \mu \) (s⁻¹ scalar) and \( \mu_\ell \) (fm⁻¹ gravity).
- \( g(\mu)=\bar J/K' \), \( \bar J=\tfrac32J \); \( \beta_g=0.72g-0.63g^2-0.011g^3 \).
- \( \beta_{g_N}=2g_N+B_1 g_N^2+O(g_N^3),\ g_N=\mu_\ell^2 G \).
- \( \beta_\Lambda\approx C_0\mu_\ell^4+C_2\mu_\ell^2\Lambda+C_4\Lambda^2 \).
- \( \beta_e=(e^3/12\pi^2)\sum q_f^2 + (e^3/48\pi^2)\sum q_s^2 + O(e^5)\).
- U(1)² shell mixing: \( V_{\rm shell}=J_{\rm ex}\!\sum\! \cos(\theta_a-\theta_b)\,e^{-(\Delta\omega/\sigma_{\rm exch})^2} \).
- Renorm condition: \( [\sqrt{-g}G_{\mu\nu}]_k=8\pi G_k[\sqrt{-g}T_{\mu\nu}]_k \) (two \( \mu_\ell \) with ratio ≈ 2).

### 4.7 Thermodynamic primitives
- **Internal energy** \( U=\sum\hbar\omega_i + \tfrac12\sum E_{ij} \).
- **Force** \( F_{ij}=(2\pi c/r^2)\,\partial E_{ij}/\partial(\Delta\omega) \), \( r=2\pi c/|\Delta\omega| \).
- **Work** \( W=\int \mathbf F\cdot d\mathbf r \).
- **Temperatures**: \( T_{\rm RTG}=\hbar(\langle\omega\rangle-\omega_{\rm obs})/k_B \) (can be negative); \( T_{\rm spec}=\hbar\sigma_\omega/k_B\ge 0 \).
- **Heat** \( \dot q=\hbar(\langle\omega\rangle_1-\langle\omega\rangle_2)\mathcal R_{12} \), \( Q=\int \dot q\,dt \).


### 4.8 Emergent Geometry (ESM)
- **Triad**: a triangular face (2-simplex) used as the elementary “surface atom” in ESM clusters.
- **Angular Deficit**: ~7.36° per edge in 3D (tetrahedral packing); canonical source of frustration.
- **z-residual (edge curvature)**: \(z_{ij} = \ln(r_{\rm em,ij}) - \ln(s\,r_{\rm geo,ij})\).
- **Boundary charge**: \(Q_{\partial} = \sum_{e \in \partial\mathcal{M}} z_e\) (Stokes-theorem analog).
- **Boundary roughness**: \(C_{\partial} = \sqrt{\langle z_e^2 \rangle_{e\in\partial\mathcal{M}}}\).
- **Hodge decomposition (discrete 1-form)**: \(z = d f + \delta g + h\). Empirically, the coexact+harmonic fraction is \(\approx 75.2\%\) (the “geometric floor”).
- **Quantum foam**: boundary-neutral yet internally rough regime: \(Q_{\partial} \approx 0\) while \(C_{\partial} > 0\).
- **Fragmentation**: loss of global connectivity under aggressive refinement (connected components \(\beta_0 > 1\)).

### 4.9 Special Node Configurations
- **Photon Object**: Defined structurally as a spin–anti‑spin pair (\( s_i = -s_j \)) sharing a common phase \( \phi_i = \phi_j \) and carrier frequency \( \omega_{carrier} \). The internal frequency difference is \( \Delta\omega = 0 \), ensuring the object is effectively massless and propagates at \( c \).

---

## 5. Core Principles

**Unification** — *All four forces from the same resonance kernel \( \mathcal R_{ij} \):*
U(1) (gauge phases) · SU(2) (spin rotations, exchange) · U(1)²≈SU(3) (phase‑shell mixing) · Gravity via \( S_{\mu\nu} \) (residual elasticity). **No external gauge fields.**

**Observer‑relational quantum** — Entanglement (CHSH 2.827±0.002 in U(1)), superposition (unresolved phases until interaction), decoherence at \( \sigma_{\rm crit}\approx0.589 \).

**Emergent space‑time** — \( \Delta\tau \approx \Delta\phi/\Delta\omega \); photon propagation through gated bonds.

**Machian Vacuum (ESM)** — Spacetime is not a pre‑existing container but a curvature‑screening field generated by defects. In ESM, the correct macroscopic flux is the boundary integral \(Q_{\partial} = \sum_{e\in\partial\mathcal{M}} z_e\) (Stokes analog). A near‑vacuum is boundary‑neutral (\(Q_{\partial}\approx 0\)) while boundary roughness \(C_{\partial}\) remains non‑zero due to a Hodge‑irreducible geometric floor (empirically \(\approx 75.2\%\) of \(\mathrm{Var}(z)\)).

**Lattice→continuum** — \( \sum G_{ij}(\Delta\phi)^2 \to a^{2-d}\int G|\nabla\phi|^2 \); **Calibration Required** (§10.4).

**Equilibrium & mass** — Proton binding ~48 MeV; emergent mass \( m_i=[\hbar\omega_i-\tfrac12\sum J\mathcal R_{ij}]/c^2\ge 0 \).

**Thermodynamics & rotation** — \( U \) with ½ double‑count fix; \( F \) from beat‑distance gradients; \( T_{\rm RTG} \) (frame‑dependent) vs \( T_{\rm spec} \) (invariant). Orbital demos in elastic regime.

**Particle & nuclear (multilayer graphs)** — Trions at Δφ=2π/3 (quarks/nucleons), **sub‑nodes** (gluon‑like, \( r_{\rm sub}\)), **buffers** (~40/proton, ~200/Ca‑40) absorb drift.

**Cosmology (ε‑drift)** — Expansion from frequency drift \( \langle\dot\omega\rangle=-\epsilon(t)\langle\omega\rangle \), \( \epsilon\propto a^{-3\alpha} \) with \( \alpha=d_f/3 \); clustering \( \delta\propto t^{2/3} \); CMB peaks (ℓ≈200); **BH corridor** at δω>0.70Δω* hardens Hawking tail by ~5% (>100 keV).

---

## 6. Key Equations

### 6.1 Microscopic (lattice)
- **Bond energy**
  \( E_{ij}=K'|\Delta\omega|/\Delta\omega^* + J\mathcal R_{ij} + J_{\rm ex}\sin(\Delta\phi-B_{ij})\,e^{-(\Delta\omega)^2/\sigma_{\rm exch}^2} \)
  (sin↔−cos equivalence; \( \sigma_{\rm exch}\neq\sigma_{\rm noise} \)).

- **Spin flip (Kinetic Compensation)**: \( \pi\phi_i \leftarrow \pi\phi_i - \text{sign}(\Delta E)\sqrt{2 M_\phi |\Delta E|} \) (conserves energy via phase momentum \( M_\phi \)).
- **Curvature penalty**: \( U_{\rm curv} = \kappa_c a^2 \sum_{\langle ijk \rangle} (1 - \bar{R}_{ijk}/3)^2 \) where \( \bar{R}_{ijk} \) is the triad average.
- **Gauge curl** \( S_B=(\kappa_B/2)\sum_\square (\mathrm{curl}\,B)^2 \).

### 6.2 Continuum & EFT
- **Action** \( S=\int[(\rho_s/2)G(\partial\phi)^2+\kappa(1-G)] + K'_{\rm TV}\|\nabla\omega\|_1/\Delta\omega^* + J_{\rm ex}\sin(\cdot) \).
  (Use \( \|\cdot\|_2 \) variant for non‑local tests.)
- **EFT** \( \mathcal L_{\rm EFT}=(\kappa_B/4)F_{\mu\nu}F^{\mu\nu}+\dots \).
- **Polarization** \( \Pi_{\rm ex}(0)=(1/6\pi^2)(J_{\rm ex}^2/K')\mu^3 \), \( \Pi_{\rm ex}(\omega)\approx\Pi_{\rm ex}(0)[1+\tfrac12(\omega/\mu)^2] \).
- **Gravity** \( S_{\rm grav}=(1/16\pi G_0)\int\sqrt{-g}(R-2\Lambda_0) \), \( S_{\phi,G} \) as in §4.5.

### 6.3 Running / RG
- **Dimensionless variables**: \( \tilde{J} \equiv J/(\hbar\Delta\omega^*) \), \( \tilde{K}' \equiv K'/(\hbar\Delta\omega^*) \).
- **Beta functions**: \( \beta_{\tilde{J}} = -\tilde{J} + O(\tilde{J}^3) \) and \( \beta_{\tilde{K}} = -\tfrac{1}{2}\tilde{K}\tilde{J} + O(\tilde{J}^3) \).
- **Finite-size correction**: Simulation ratio \( J/K' \approx 0.27 \) maps to fixed point \( g^* \approx 1.14 \) after finite-size adjustments.

### 6.4 Thermodynamics
- **U, F, W, \(T_{\rm RTG}\), \(T_{\rm spec}\), \( \dot q, Q \)** as in §4.7.
- **Emergent mass** \( m_i=[\hbar\omega_i-\tfrac12\sum J\mathcal R_{ij}]/c^2\ge0 \).

### 6.5 Particle lattice (RTG ↔ LQCD bridge)
- \( S=\sum_{\langle ij\rangle}\Big[\frac{\sigma}{2}|1-\mathcal U_{ij}|^2+\kappa(1-G_{ij})+\frac{K'}{\Delta\omega^*}|\omega_i-\omega_j|\Big]+\sum_i \frac{m_i^2 a^3}{2} \),
  \( \mathcal U_{ij}=G_{ij}e^{i\Delta\phi_{ij}} \), continuum \( V(r)\propto \sigma r - \kappa/r \).
  Mapping: \( U_{ij}\!\leftrightarrow\!U_\mu \), \( G_{ij}\!\leftrightarrow\! \)color projector, buffers↔sea; \( \Delta\omega^*\leftrightarrow \Lambda_{\rm QCD} \).

### 6.6 Cosmology (ε‑drift, CMB, BH corridor)
- \( \dot\omega_i=-(K'/\Delta\omega^*)\sum_j (\omega_i-\omega_j)\mathcal R_{ij} \),
  \( \epsilon(t)=K' N_{\rm eff}\sigma_\omega^2/\Delta\omega^* \propto a^{-3\alpha} \) with \( \alpha=d_f/3 \).
- CMB: \( \Delta T/T = \Delta\omega/\langle\omega\rangle,\ \Delta\omega=0.28\,\Delta\omega^*(1+z)^{-1} \);
  \( C_\ell = A_s (0.28\Delta\omega^*)^2/[\ell(\ell+1)]\,e^{-\ell(\ell+1)\sigma_\phi^2} \).
- BH: corridor for δω>0.70Δω* → \( \Gamma_H\propto e^{-(\delta\omega/\Delta\omega^*)^2} \), \( T_H^{\rm RTG}\approx T_H^{\rm GR}\Gamma_H \).

### 6.7 ESM (Emergent Geometry)
- **Edge residual / curvature field**:
  \[
    z_{ij} = \ln(r_{\rm em,ij}) - \ln(s\,r_{\rm geo,ij}), \qquad
    r_{\rm em,ij} = \frac{\kappa}{\lvert \Delta\omega_{ij} \rvert}.
  \]
- **Boundary charge (Stokes analog)**:
  \[
    Q_{\partial} = \sum_{e\in\partial\mathcal{M}} z_e
  \]
- **Boundary roughness**:
  \[
    C_{\partial} = \sqrt{\left\langle z_e^2 \right\rangle_{e\in\partial\mathcal{M}}}
  \]
- **Triad attachment update** (when one boundary edge becomes internal and two new boundary edges are created):
  \[
    \Delta Q_{\partial} = -z_{e_{\rm old}} + z_{e_{\rm new,1}} + z_{e_{\rm new,2}}.
  \]
- **Hodge decomposition** (discrete 1‑form):
  \[
    z = d f + \delta g + h.
  \]
  Empirically, the exact fraction is \(\approx 24.8\%\) and the coexact+harmonic fraction is \(\approx 75.2\%\) (the “geometric floor”).

---

## 7. Simulation Benchmarks
- **Quantum**: CHSH (U(1)) \(2.827\pm0.002\); decoherence at \( \sigma_{\rm crit}\approx0.589 \).
- **CHSH Decay**: Explicit fit \( S(\sigma_{\rm noise}) = 2\sqrt{2} \exp(-\sigma_{\rm noise}^2) \).
- **Ward residuals**: \( (0.10\pm0.97)\times10^{-3} \).
- **Exchange fraction**: 1.083–1.514%.
- **C_\kappa** extraction (L=32/64/96) spread <0.13%.
- **Drift** < \(4.3\times10^{-4}\); **spin flips** 0.02–0.03 (thermostat cap 0.30).
- **Renorm**: \( \mu_{\ell,2}/\mu_{\ell,1}\approx2 \).
- **Thermo**: Carnot \( \eta\approx1-T_c/T_h \) (≤3% err for small Δφ,Δω); \( \langle\nabla\cdot J\rangle\lesssim10^{-3} \).
- **Particle/Nuclear**:
  Proton \( m_p=938.3\pm6.4 \) MeV, \( r_p=0.84\pm0.009 \) fm;
  \(^{40}\)Ca \( R=4.80\pm0.05 \) fm; binding \(340\pm4\) MeV;
  Exotics: tetra \(2573\pm17\) MeV; penta width ~33 MeV;
  HMC 64³ (12k traj), autocorr \(38\pm9\), anomaly < \(10^{-3}\).
- **Cosmology**: \( H(z\!=\!1)\approx68\pm2 \) km·s⁻¹·Mpc⁻¹; CMB peaks ℓ≈200–1000 with \( \sigma_\phi\approx10^{-4} \);
  BH tail hardening **+5%** (>100 keV), PBH lifetime ~ **−5%**; fractal \( d_f\approx2.0\pm0.1 \).


### 7.5 ESM Benchmarks (Emergent Geometry)
- **Vacuum growth (demonstration)**: dipole seed (4 nodes, 2 triads) \(\to\) 16 triads (14 nodes); boundary charge \(Q: -1.60 \to -0.04\) (**97.7% screened**).
- **Dual metrics**: boundary roughness \(C_{\partial}\approx 0.68\) (bounded); near‑vacuum requires \(Q_{\partial}\approx 0\) without forcing \(C_{\partial}\to 0\).
- **Geometric floor / Hodge limit**: IRLS embedding fits stall at median \(\lvert z \rvert\approx 0.46\); Hodge decomposition shows coexact+harmonic \(\approx 75.2\%\) of \(\mathrm{Var}(z)\) (irreducible).
- **Refinement trade‑off**: aggressive smoothing can fragment connectivity (\(\beta_0>1\)). Practical workflows either penalize fragmentation during surgery or prune to the largest component and re‑grow.

---

## 8. Applications & Open Questions

**Applications**:
Hadrons/nuclei (proton, Ca‑40, exotics); Bell violations/quantum measurement; **emergent geometry (ESM)**: vacuum growth via boundary‑charge screening (quantum foam); cosmology (H(z), CMB \( C_\ell \), BH spectra, PBH constraints).

**Open** (selection; see Tier‑2 for full list & priorities):
- **Hypothesis Test (Floor)**: Verify \( 0.752 \times \mathrm{Var}(z) \approx E_0 \approx 1/12 \).
- **Diagnostic Tool**: Implement **Trace Gap** (\(\Delta_{\rm Tr} = |\Tr(U) - 2\cos\theta|\)) to detect topological defects.
- **Monopole Test**: Inject charged simplex, measure screening response (proto-gravity).
- ESM scaling: test dipole shell‑closure targets \(\{70,140,252,\dots\}\).
- Refinement under connectivity constraints: surgery with fragmentation penalties.
- Dynamic extension: couple ESM clusters to phase flow (\(\phi,\omega\)) and explore 4D/Lorentzian formulations.

---

## 9. Maintenance & Conventions
- **Append‑only** for numerical values; deprecations noted in Tier‑2.
- **Units/notation**: ω in rad·s⁻¹; \( \sigma_{\rm exch}\neq\sigma_{\rm noise} \); U(1) can be global or gauged; sin/−cos exchange equivalence; spins within gauge choices; EFT amplitudes off by default; dual μ conventions (s⁻¹ vs fm⁻¹).

---

## 10. Synthesis: The Frustrated Simplicial Vacuum
*New in v1.17*

### 10.1 The Core Obstruction (Embedding Mismatch)
RTG spacetime is not built from simplices; it is constrained by their impossibility.

- **2D vs 3D**: While ESM simulations operate on 2D triads, the frustration principle is universal. The **~7.36° angular deficit** of regular 3D tetrahedra (dihedral angle \(\approx 70.53^\circ\) vs \(72^\circ\) ideal) is the canonical example of the irreducible obstruction between relational (frequency-derived) and geometric (embedding) distances.

### 10.2 Origins of Structure (Theoretical Mappings)
- **Geometric Floor ↔ Thermodynamic Floor**: The observed Hodge-irreducible curvature (\(\eta_{irr} \approx 0.752\)) is **consistent with** the thermodynamic floor (\(E_0 \approx 1/12\)) required to contain the vacuum state. (Verification target: \(0.752 \times \mathrm{Var}(z) \approx E_0\)).
- **Vacuum Nucleation ↔ Inertial Spark**: Vacuum growth qualitatively matches **Neimark-Sacker bifurcation** dynamics, requiring a critical coupling (\(K^*\)) to ignite.
- **Tetrahedral Gap ↔ Trace Gap**: We propose that angular deficits map to the **Trace Gap** (\(\Delta_{Tr}\)), measuring the obstruction to an SU(2) lift.

### 10.3 Inertial Fracture (Validated)
- **Definition**: The validated transition to hyperchaos and mesh fragmentation (\(\beta_0 \uparrow\)) when the effective coupling exceeds the manifold's stability bound.

### 10.4 Unified Definition
**ESM Vacuum**: A frustrated simplicial complex operating in a bounded stability window, sustained by geometric frustration that manifests as an irreducible curvature floor.
- **Key Observables**: Boundary charge \(Q_{\partial} \approx 0\); Boundary roughness \(C_{\partial} > 0\); Hodge-irreducible fraction \(\eta_{irr} \approx 75\%\).

### 10.5 Conclusion
Geometry emerges as the least-frustrated configuration of relational frequencies under embedding constraints, producing a vacuum that is neutral only statistically and curved irreducibly.

**End of Core Notes (v1.17)**
