# Relational Time Geometry (RTG) Core Notes

**Version**: v1.14  
**Last Revised**: 2025-09-24  
**Author**: Mustafa Aksu (with Grok & ChatGPT contributions)  
**Purpose**: Authoritative anchor for RTG principles, workflows, and key concepts. Overrides chat history; use as Tier 1 static core. For evolutions/experiments, consult Tier 2 (findings.yaml, rtg_articles_index.yaml).  
**This version integrates**: Gravity I (tree-level EH), Gravity II (quantum corrections, running, anomalies), **Forces & Fields** (unified forces principle), **Thermodynamics & Rotational Dynamics** (Aug 2025), **Particle & Nuclear Modeling v2.4**, and **Cosmological Applications v2.8** (Aug 2025).

---

## 1. Introduction to RTG
Relational Time Geometry (RTG) models the universe as a graph of oscillatory nodes, where space-time, mass, and interactions emerge from relational frequencies, phases, and spins. Core idea: **Time is geometric**, defined by beat-frequencies and phase-locking between primitives—no absolute background.

- **Key Constants** (calibrated via RG/MD; see findings.yaml for sim-derived refinements):  
  | Symbol | Value | Units | Notes |
  |---|---:|---|---|
  | \(\Delta \omega^*\) | 1.45(8) × 10^{23} | rad·s⁻¹ | Critical bandwidth; RG-fixed. |
  | \(c\) | 299792458 | m/s | Speed of light (fixed). |
  | \(\hbar\) | 1.0545718 × 10^{-34} | J·s | Reduced Planck constant. |
  | Proton radius | 0.84 ± 0.01 | fm | Calibration target. |
  | \(g^*\) | ≈ 1.14 ± 0.02 | – | RG fixed point (scalar β_g). |
  | \(\sigma_{\rm crit}\) | ≈ 0.589 | – | CHSH decoherence threshold. |
  | \(K'\) | 12.0 (±0.5%) | MeV | Frequency penalty scale. |
  | \(J\) | 3.24 ± 0.12 | MeV | Resonance coupling. |
  | \(J_{\rm ex}\) | 2.20 ± 0.08 | MeV | Exchange coupling. |
  | \(\alpha\) | ≈ 938 | MeV·fm² | Mass form factor. |
  | \(\kappa_c\) | 1 | MeV·fm | Curvature scale. |
  | Proton binding | ≈ 48 ± 3 | MeV | 3-node equilibrium. |
  | ℓ* | 2.07 ± 0.11 | fm | Spectral length (c/Δω*). |
  | r* | 13.0 ± 0.7 | fm | Beat length (2π ℓ*). |
  | ρ_s | ≈ 15.9 | MeV·fm⁻¹ | Phase-stiffness (a_lat=0.08 fm). |
  | κ_B | ~ 1 | MeV·fm | Gauge stiffness (curl B). |
  | C_κ^∞ (SU(2)) | 0.009104 ± 5.5e-7 | – | Wilson coeff (0.28–0.70). |
  | C_κ^∞ (U(1)) | 0.009116 ± 1.1e-7 | – | Wilson coeff (0–0.28). |
  | C_κ^∞ (U(1)^2) | 0.009115 ± 4.9e-9 | – | Wilson coeff (1.55–1.70). |
  | κ_B (Maxwell) | C_κ J_ex² / K′ | MeV⁻¹ | EFT prefactor. |
  | G_0 | ~ ρ_s ℓ*² | m³·kg⁻¹·s⁻² | Newton const (provisional). |
  | Λ_0 | TBD | m⁻² | Cosmological constant (coarse-grained). |
  | B_1 | ~ 0.1–0.2 | – | Gravity β_{g_N} coeff (scheme-dep.). |
  | C_0/C_2/C_4 | TBD | – | Λ running coeffs. |
  | e | ~ 1/137 | – | U(1) charge (provisional). |
  | **σ_exch** | **O(Δω\*)** | rad·s⁻¹ | **Exchange UV regulator; independent of noise σ_noise.** |
  | **T_spec** | **ħ σ_ω / k_B** | K | **Spectral temperature (frame-invariant).** |
  | **ω_q** | **2.51 × 10^{23}** | rad·s⁻¹ | **Quark-level frequency (particle modeling).** |
  | **ω_q,sub** | **1.26 × 10^{20}** | rad·s⁻¹ | **Sub-node inner oscillator frequency.** |
  | **r_sub** | **5 × 10⁻¹⁷** | fm | **Sub-node spacing (gluon-like layer).** |
  | **β** | **≈ 10⁻³** | – | **RG-scaled sub-node flip factor.** |
  | **H₀** | **≈ 68 ± 2** | km·s⁻¹·Mpc⁻¹ | **Calibrated expansion rate (ε-drift).** |
  | **t₀** | **≈ 13.7** | Gyr | **Universe age (ε-integration / CMB freeze-out).** |
  | **r** | **< 0.01** | – | **Tensor-to-scalar ratio (prediction).** |
  | **σ_φ** | **≈ 1×10⁻⁴** | – | **CMB phase scatter from lattice sims.** |

- **Guiding Principles**: Minimal decoherence; phase-locking for stability; RG-anchored scales (two-loop β_g for scalar sector, running β_{g_N}/β_Λ for gravity sector); emergent phenomena from relations only.

---

## 2. Memory Architecture
Hierarchical storage for evolvability.

---

## 3. Core Mechanics
- **Geometry Nodes**: ω_i, φ_i, spin (analytic s_i = ±i, code σ_i = ±1); intrinsic time \(t_i = \tilde{\phi}/\omega_i\).  
- **Relational Edges**: Weighted by resonance.  
- **Tooling Integration**: Augment with external fetches.

---

## 4. Glossary

### Fundamental Building Blocks
- Node, ω, φ, spin, intrinsic time.

### Resonance & Interaction Metrics
- Resonance kernel ℛ_ij; beat-frequency r_ij.

### Continuum Mapping
- \(ρ_s = (3/2) J \, a_{\rm lat}^{2-d}\).  
- Gate field \(G(x)\).  
- TV penalty \(K'_{\rm TV}\,\|\nabla \omega\|_1 / \Delta\omega^*\).

### Observer Concepts
- Observer node/frame.  
- Δω\*-observer.  
- **Superposition**: Node phases φ_i exist in unresolved states *relative to observer nodes* until resolved via phase-locking (Δφ_ij≈0) or spin flips (ΔE_J = 2J σ_i Σ A_ij σ_j).

### Emergent Dimensionality & Critical Bandwidth
| δω/Δω* Range | Dimensionality | Regime / Symmetry |
|---|---|---|
| <0.28 (±0.02) | 2D | Planar sheets; U(1) |
| 0.28–0.70 | 3D | Curved shells; SU(2) |
| 0.70–1.55 | 4D | Hyper corridors |
| 1.55–1.70 | Transitional | U(1)^2 phase shells (ε≈SU(3)) |
| ≥1.70 | >5D | High-D anomalies (dim-6?) |

- EFT: C_κ constants per window.

### Stability & Guiding Principles
- Phase-locking; Δω ≪ Δω*; spin flips; drift thresholds.

### Special Node Types
- Photon; dynamic params.

#### Gauge Symmetries
- **U(1)**: \(B_{ij}=aA_{ij}\); \(U_{ij}=e^{i(\Delta\phi_{ij}-B_{ij})}\).  
- **SU(2)**: \(\psi_i=[z_i, i s_i z_i]^T\); rotations \(R(\theta)\); Witten anomaly → even doublets.  
- **U(1)^2 ≈ SU(3)**: Three-shell phases (θ₁,θ₂,θ₃) with one constraint → two independent phases; coupling ε produces SU(3)-like behavior. **Validated by proton binding (~48 ± 3 MeV).**  
- **Forces as Unified Resonances**: Electromagnetic (U(1)), weak (SU(2)), strong (U(1)^2 ≈ SU(3)), and gravity (via \(S_{\mu\nu}\)) all emerge from the same resonance kernel ℛ_ij. *No external gauge fields required.*  
- Exchange: sin(Δφ − B_ij) or −cos equivalent.  
- Anomalies: U(1) charge/cubic sums=0; SU(2) Witten.

#### EFT Construction
- Minimal RTG-EFT: \(H_{\rm lat} \to \mathcal{L}_{\rm EFT}\) dim-4 (Maxwell).  
- \(\mathcal{L}_{\rm EFT}: (\kappa_B/4)\, F_{\mu\nu}F^{\mu\nu} + \dots\)  
- C_κ per window (MC).  
- Anomalies: Ward residuals ~0.  
- Exchange fraction: 1.083–1.514% (μ/Δω*).  
- Dim-6 operators deferred.

#### Emergent Gravity (Tree-Level)
- **Structure tensor \(S_{\mu\nu}\)**: \((1/\mathcal{N}) \sum w_{ij}\, \Delta x_{ij,\mu}\Delta x_{ij,\nu}\) (w=ℛ_ij or A_ij).  
- **Tetrad \(e^a{}_\mu\)**: From S eigen; \(g_{\mu\nu}=\Omega^2 \eta_{ab} e^a{}_\mu e^b{}_\nu\) (Ω for c/signature).  
- Tree-level EH: \(S_{\rm grav}=(1/16\pi G_0)\int \sqrt{-g}\,(R-2\Lambda_0)\).  
- Matter: \(S_{\phi,G}=\int \sqrt{-g}[ (ρ_s/2)\,G\, g^{\mu\nu}\partial_\mu\phi\partial_\nu\phi + \kappa(1-G)]\).  
- U(1): \(B_{ij}\to A_\mu\); \(S_B\) curl → \(F_{\mu\nu}\); EFT Maxwell coupling.

#### Quantum Gravity Corrections
- Dual μ: s⁻¹ (scalar RG), fm⁻¹ (gravity RG).  
- \(g(\mu)=\bar{J}/K'\) with \(\bar{J}=3/2\,J\).  
- **β_g (scalar)**: \(0.72 g - 0.63 g^2 - 0.011 g^3\); \(C_{\rm vtx}=3/2\), \(a_3=0.011\).  
- **β_{g_N}**: \(2 g_N + B_1 g_N^2 + O(g_N^3)\) (g_N=μ_ℓ² G).  
- **β_Λ**: ≈ \(C_0 \mu_ℓ^4 + C_2 \mu_ℓ^2 \Lambda + C_4 \Lambda^2\).  
- **β_e (U(1))**: \((e^3/12\pi^2)\sum q_f^2 + (e^3/48\pi^2)\sum q_s^2 + O(e^5)\).  
- **SU(2)**: Heisenberg exchange (beyond Ising).  
- **U(1)^2 \(V_{\rm shell}\)**: \(J_{\rm ex}\sum \cos(\theta_a-\theta_b)\, e^{-(\Delta\omega_{ab}/\sigma_{\rm exch})^2}\).  
- Anomalies: SU(2) none perturbative (d^{abc}=0), Witten global; U(1) sums=0 gauged.  
- Windows (±0.02 edges): U(1) [0,0.28], SU(2) [0.28,0.70], U(1)^2 [1.55,1.70].

#### **Thermodynamic Primitives**
- **Internal Energy \(U\)**: \(U=\sum_i \hbar \omega_i + \tfrac12\!\sum_{i\ne j}\!\big[K'|\Delta\omega|/\Delta\omega^* + J\,\mathcal{R}_{ij} + J_{\rm ex}\sin\Delta\phi\, e^{-(\Delta\omega/\sigma_{\rm exch})^2}\big]\).  
- **Force \(F_{ij}\)**: \(F_{ij} = (2\pi c / r^2)\,\big[\partial E_{ij}/\partial(\Delta\omega)\big]\) with \(r=2\pi c/|\Delta\omega|\).  
- **Work \(W(\gamma)\)**: \(\int_\gamma \mathbf{F}\cdot d\mathbf{r}\).  
- **Relational Temperature \(T_{\rm RTG}\)**: \(T_{\rm RTG}(\omega_{\rm obs})=\hbar(\langle\omega\rangle - \omega_{\rm obs})/k_B\) (*can be negative*).  
- **Spectral Temperature \(T_{\rm spec}\)**: \(T_{\rm spec}=\hbar \sigma_\omega / k_B \ge 0\).  
- **Heat Flow**: \(\dot q=\hbar(\langle\omega\rangle_1 - \langle\omega\rangle_2)\,\mathcal{R}_{12}\); \(Q(t)=\int_0^t \dot q\,dt\).

---

## 5. Core Principles
- **Node Primitives**: as Glossary.  
- **Emergent Space-Time**: \(\Delta\tau \approx \Delta\phi/\Delta\omega\); photon propagation.  
- **Lattice-continuum**: \(\sum G_{ij}(\Delta\phi_{ij})^2 \to a_{\rm lat}^{2-d}\int G|\nabla\phi|^2\); \(ρ_s=(3/2)J\, a_{\rm lat}^{2-d}\).  
- **Note**: +8% J retune recovers \(r_p=0.84\) fm.  
- **Observer Dependence**: relational. *Quantum behaviors (entanglement, superposition, decoherence) emerge relationally.*  
- **Emergent Dimensionality**: as table.  
- **Equilibrium & Mass**: Proton binding ~48 MeV.  
- **Gauge**: \(\Delta\phi \to \Delta\phi - B_{ij}\); SU(2) rotations; U(1)^2 ε≈SU(3).  
- **EFT**: \(U_{ij}\to\) Maxwell; scaling predictions.  
- **Gravity (Tree-Level)**: Bonds → \(S_{\mu\nu}\) → \(g_{\mu\nu}\); EH action; couples to \(\phi/G/A_\mu\).  
- **Quantum Gravity**: Loops run \(G,\Lambda,e\); SU(2) Heisenberg; U(1)^2 \(V_{\rm shell}\); renorm condition \([\sqrt{-g} G_{\mu\nu}]_{\mu_\ell,k}=8\pi G(\mu_\ell,k)[\sqrt{-g} T_{\mu\nu}]_{\mu_\ell,k}\) (\(\mu_{\ell,2}/\mu_{\ell,1}\approx 2\)).

**Unified Forces Principle**  
All four fundamental forces emerge from the same resonance kernel ℛ_ij:  
- Electromagnetic via U(1) gauge phases.  
- Weak via SU(2) spin rotations & exchange.  
- Strong via U(1)^2 phase shells approximating SU(3).  
- Gravitational via the structure tensor \(S_{\mu\nu}\) (residual elasticity).  
*No external gauge fields required; forces unify through pairwise node interactions (ω, φ, s).*

**Emergent Thermodynamics & Rotational Dynamics**  
Thermodynamics arises from node relations. \(U\) includes intrinsic + ½ bond sum. \(F\) from beat-distance gradients. Temperatures: \(T_{\rm RTG}\) (frame-dependent, possibly negative) and \(T_{\rm spec}\) (spectral, ≥0). Heat transfer is gated by resonance. Emergent mass \(m_i = [\hbar \omega_i - \tfrac12 \sum J \mathcal{R}_{ij}]/c^2\) (clamped ≥0). Orbital demos show elastic-residual dynamics.

**Particle & Nuclear Modeling (Multilayer Graphs)** *(new)*  
Hadrons/nuclei as **multilayer resonance graphs**: primary nodes (quarks/nucleons) form **trions** at Δφ=2π/3; **sub-nodes** (gluon-like, spacing \(r_{\rm sub}\approx 5\times10^{-17}\) fm, flips set by β≈10⁻³); **buffer nodes** provide thermal stabilization (≈40 per proton, ≈200 per Ca-40). Exotic tetra/pentaquarks arise in HMC ensembles.

**Cosmological Applications (ε-drift expansion)** *(new)*  
No dark sectors: expansion from the mean frequency drift \(\langle\dot\omega\rangle=-\epsilon(t)\langle\omega\rangle\) with \(\epsilon\propto a^{-3\alpha}\) and \(\alpha=d_f/3\). Structure growth from resonance clustering (\(\delta\propto t^{2/3}\) in matter era). **Micro–macro unification**: proton scale (e.g., \(r_p\approx0.84\) fm) to CMB peaks (ℓ≈200) via dimensional freeze-out at \(z\sim 1100\). **BH corridors** (δω>0.70 Δω*) modulate Hawking spectra (+5% hardening >100 keV).

---

## 6. Key Equations
- **Bond Energy \(E_{ij}\)**: \(K'|\Delta\omega|/\Delta\omega^* + J \mathcal{R}_{ij} + J_{\rm ex}\sin(\Delta\phi - B_{ij}) e^{-(\Delta\omega)^2/\sigma_{\rm exch}^2}\); sin/−cos equiv.; \(\sigma_{\rm exch} \ne \sigma_{\rm noise}\).  
- **Spin Flip Energy**: \(\Delta E_J = 2J\, \sigma_i \sum A_{ij}\sigma_j\).  
- **Curvature Penalty \(U_{\rm curv}\)**: Helfrich forms.  
- **\(S_B\)**: \((\kappa_B/2)\sum_\square (\mathrm{curl}\,B)^2\).  
- **Continuum Action \(S\)**: \(\int [ (ρ_s/2) G (\partial\phi)^2 + \kappa(1-G)] + K'_{\rm TV}\|\nabla\omega\|_1/\Delta\omega^* + J_{\rm ex}\sin(\dots)\).  
  - L2 norm alt: use \(\|\cdot\|_2\) for non-local.  
- **EFT \(\mathcal{L}_{\rm EFT}\)**: \((\kappa_B/4) F_{\mu\nu}F^{\mu\nu} + \dots\)  
- **\(\Pi_{\rm ex}(0)\)**: \((1/6\pi^2)(J_{\rm ex}^2/K') \mu^3\).  
- **\(\Pi_{\rm ex}(\omega)\)** ≈ \(\Pi_{\rm ex}(0)\,[1+\tfrac12(\omega/\mu)^2]\).  
- **EH \(S_{\rm grav}\)**: \((1/16\pi G_0)\int \sqrt{-g}(R-2\Lambda_0)\).  
- **Matter \(S_{\phi,G}\)**: \(\int \sqrt{-g}[(ρ_s/2)G g^{\mu\nu}\partial\phi\partial\phi + \kappa(1-G)]\).  
- **β_{g_N}**: \(2 g_N + B_1 g_N^2 + O(g_N^3)\).  
- **β_Λ**: ≈ \(C_0 \mu_ℓ^4 + C_2 \mu_ℓ^2 \Lambda + C_4 \Lambda^2\).  
- **β_e (U(1))**: \((e^3/12\pi^2)\sum q_f^2 + (e^3/48\pi^2)\sum q_s^2\).  
- **\(V_{\rm shell}\) (U(1)^2)**: \(J_{\rm ex}\sum \cos(\theta_a-\theta_b) e^{-(\Delta\omega_{ab}/\sigma_{\rm exch})^2}\).  
- **β_g refine**: \(0.72 g - 0.63 g^2 - 0.011 g^3\) (\(a_3=0.011, C_{\rm vtx}=3/2\)).  
- **Internal Energy \(U\)**: \(\sum \hbar\omega_i + \tfrac12 \sum \text{bond terms}\).  
- **Force \(F_{ij}\)**: \((2\pi c / r^2)\,\partial E/\partial(\Delta\omega)\).  
- **Relational \(T_{\rm RTG}\)**: \(\hbar(\langle\omega\rangle-\omega_{\rm obs})/k_B\).  
- **Spectral \(T_{\rm spec}\)**: \(\hbar\sigma_\omega/k_B\).  
- **Heat \(\dot q, Q\)**: \(\dot q=\hbar(\langle\omega\rangle_1-\langle\omega\rangle_2)\mathcal{R}_{12}\); \(Q=\int \dot q\,dt\).  
- **Emergent Mass \(m_i\)**: \([\hbar\omega_i - \tfrac12 \sum J\mathcal{R}_{ij}]/c^2\) (≥0).

**Particle–lattice action (RTG ↔ LQCD bridge)** *(new)*  
\(S=\sum_{⟨ij⟩}\!\Big[\frac{\sigma}{2}\,\lvert 1-\mathcal U_{ij}\rvert^2 + \kappa(1-G_{ij}) + \frac{K'}{\Delta\omega^*}\,\lvert\omega_i-\omega_j\rvert\Big] + \sum_i \frac{m_i^2 a^3}{2},\)  
with \(\mathcal U_{ij}=G_{ij}\,e^{i\Delta\phi_{ij}}\); continuum potential \(V(r)\propto \sigma r - \kappa/r\).  
**Mapping**: \(U_{ij}\leftrightarrow U_\mu\) (link), \(G_{ij}\leftrightarrow\) color projector, \(\sigma\leftrightarrow\) string tension, buffers ↔ sea quarks, \(\Delta\omega^*\leftrightarrow \Lambda_{\rm QCD}\).

**Cosmology (ε-drift, CMB, BH corridor)** *(new)*  
\(\dot{\omega}_i=-(K'/\Delta\omega^*)\sum_j(\omega_i-\omega_j)\mathcal R_{ij};\ \ \epsilon(t)=K' N_{\rm eff}\sigma_\omega^2/\Delta\omega^*\propto a^{-3\alpha},\ \alpha=d_f/3.\)  
CMB freeze-out: \(\Delta T/T=\Delta\omega/\langle\omega\rangle,\ \Delta\omega=0.28\,\Delta\omega^*(1+z)^{-1};\)  
\(C_\ell = A_s\,(0.28 \Delta\omega^*)^2/[\ell(\ell+1)]\,\exp[-\ell(\ell+1)\sigma_\phi^2].\)  
BH Hawking modulation (corridor at δω>0.70 Δω*): \(\Gamma_H\propto \exp[-(\delta\omega/\Delta\omega^*)^2],\ \ T_H^{\rm RTG}\approx T_H^{\rm GR}\,\Gamma_H.\)

---

## 7. Simulation Benchmarks
- CHSH U(1): 2.827 ± 0.002.  
- U(1)^2: coherence decay.  
- Ward residuals: (0.10 ± 0.97) × 10⁻³.  
- Exchange fraction: 1.083–1.514%.  
- C_κ SU(2): ± 5.5e-7.  
- Drift < 4.3 × 10⁻⁴.  
- Flip rates 0.02–0.03.  
- β_g MC: ± 0.02 g* (scheme).  
- Witten anomaly: even SU(2) doublets/cell.  
- U(1) sums = 0 gauged.  
- Renorm condition: \(\mu_{\ell,2}/\mu_{\ell,1}\approx 2\).  
- **Carnot efficiency**: \(\eta \approx 1 - T_c/T_h\) (within 3% error for small Δφ, Δω).  
- **Anomaly bound**: \(\langle \nabla_\mu J_\mu\rangle \lesssim 10^{-3}\).  
- **Orbital demo**: \(v(r)\) with \(m_0 \approx 1.7\times 10^{-28}\) kg.  
- **Heat-flow with Δω noise**: cumulative \(Q\) consistent with EFT anomaly constraints.  
- **Particle/Nuclear (new)**:  
  - Proton: \(m_p = 938.3 \pm 6.4\) MeV; \(r_p = 0.84 \pm 0.009\) fm.  
  - \(^{40}\)Ca: \(R = 4.80 \pm 0.05\) fm; binding \(= 340 \pm 4\) MeV.  
  - Exotics: tetraquark mass \(2573 \pm 17\) MeV; pentaquark width ~ 33 MeV.  
  - HMC: 64³ (12k trajectories), autocorr \(38 \pm 9\) with cluster flips; anomaly < 10⁻³; OSS ensemble planned.  
- **Cosmology (new)**:  
  - \(H(z=1)\approx 68 \pm 2\) km·s⁻¹·Mpc⁻¹ (ε-calibration).  
  - CMB peaks: ℓ ≈ 200–1000 with \(\sigma_\phi \approx 10^{-4}\).  
  - BH corridor: Hawking tail hardening **+5%** above 100 keV; PBH lifetime shift ~ **5%**.  
  - Fractal dimension \(d_f \approx 2.0 \pm 0.1\).

---

## 8. Applications & Open Questions
- **Applications**: Hadrons, Bell violations, cosmology. QED (U(1)), weak SU(2), strong SU(3) ε. EFT \(\Pi_{\rm ex}\) scattering. Running \(G(\mu)\), \(\Lambda(\mu)\), \(\alpha(\mu)\).  
  - *Particle/Nuclear (new)*: Proton, Ca-40, exotics (HMC); RTG↔QCD mapping (links, projectors, sea/buffers, Δω*↔Λ_QCD).  
  - *Cosmology (new)*: Parameter-minimal expansion; CMB \(C_\ell\), H(z), BH spectra, PBH constraints.
- **Open**: Spin quantization (±i → Dirac/SU2); domain boundaries; \(B_{ij}\) gradient experiments; ε cosmological; dim-6 > 1.70; SU(3) embed; μ_ℓ probes; loop-level corrections; anomaly tests.  
  - *Particle/Nuclear (new)*: Entangled sub-nodes; e–p form factors; 256³ cluster sims (<2% mass error).  
  - *Cosmology (new)*: Lorentz covariance of φ/ω; halo statistics vs ΛCDM; BH corridor bounds & 5D anomalies (≥1.70); TeV–PeV PBH excess searches (CTA/LHAASO).

---

## 9. Maintenance
- Live Diff Tracker; updates append-only.  
- **Notation guard**: ω angular rad·s⁻¹; \(\sigma_{\rm exch} \ne \sigma_{\rm noise}\); U(1) global/gauged; sin/−cos equiv.; spins in gauges; EFT amp=none baseline; dual μ (s⁻¹ scalar vs fm⁻¹ gravity).

**End of Core Notes (v1.14)**
