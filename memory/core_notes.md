# Relational Time Geometry (RTG) Core Notes

**Version**: v1.8.1  
**Last Revised**: 2025-09-23  
**Author**: Mustafa Aksu (with Grok & ChatGPT contributions)  
**Purpose**: Authoritative anchor for RTG principles, workflows, and key concepts. Overrides chat history; use as Tier 1 static core. For evolutions/experiments, consult Tier 2 (findings.yaml, rtg_articles_index.yaml). This version integrates Gauge Symmetries, EFT construction, and lattice–continuum refinements. Minor polishes from v1.8: expanded dim table, L2 norm option, +8% J retune note.

## 1. Introduction to RTG
Relational Time Geometry (RTG) models the universe as a graph of oscillatory nodes, where space-time, mass, and interactions emerge from relational frequencies, phases, and spins. Core idea: Time is geometric, defined by beat-frequencies and phase-locking between primitives—no absolute background.

- **Key Constants** (calibrated via RG/MD; see findings.yaml for sim-derived refinements):  
  | Symbol | Value | Units | Notes |
  |--------|--------|--------|-------|
  | \(\Delta \omega^*\) | 1.45(8) × 10^{23} | rad·s⁻¹ | Critical bandwidth; RG-fixed. |
  | \(c\) | 299792458 | m/s | Speed of light (fixed). |
  | \(\hbar\) | 1.0545718 × 10^{-34} | J·s | Reduced Planck constant. |
  | Proton radius | 0.84 ±0.01 | fm | Calibration target. |
  | \(g^*\) | ≈1.14 ±0.02 | - | RG fixed point. |
  | \(\sigma_{\rm crit}\) | ≈0.589 | - | CHSH decoherence threshold. |
  | \(K'\) | 12.0 (±0.5%) | MeV | Frequency penalty scale. |
  | \(J\) | 3.24 ±0.12 | MeV | Resonance coupling. |
  | \(J_{\rm ex}\) | 2.20 ±0.08 | MeV | Exchange coupling. |
  | \(\alpha\) | ≈938 | MeV·fm² | Mass form factor. |
  | \(\kappa_c\) | 1 | MeV·fm | Curvature scale. |
  | Proton binding | ≈48 ±3 | MeV | 3-node equilibrium. |
  | ℓ* | 2.07 ±0.11 | fm | Spectral length (c / Δω*). |
  | r* | 13.0 ±0.7 | fm | Beat length (2π ℓ*). |
  | ρ_s | ≈15.9 | MeV·fm^{-1} | Phase-stiffness (a_lat=0.08 fm). |
  | κ_B | ~1 | MeV·fm | Gauge stiffness (curl B). |
  | C_κ^∞ (SU(2)) | 0.009104 ±5.5e-7 | - | Wilson coeff (0.28–0.70). |
  | C_κ^∞ (U(1)) | 0.009116 ±1.1e-7 | - | Wilson coeff (0–0.28). |
  | C_κ^∞ (U(1)^2) | 0.009115 ±4.9e-9 | - | Wilson coeff (1.55–1.70). |
  | κ_B (Maxwell) | C_κ J_ex^2 / K' | MeV^{-1} | EFT prefactor. |

- **Guiding Principles**: Minimal decoherence; phase-locking for stability; RG-anchored scales (two-loop \(\beta_g ≈ 0.72 g - 0.63 g^2 - 0.011 g^3\)); emergent phenomena from relations only.

## 2. Memory Architecture
Hierarchical storage for evolvability.

## 3. Core Mechanics
- **Geometry Nodes**: ω_i, φ_i, spin (analytic s_i=±i, code σ_i=±1); intrinsic time t_i=φ̃/ω_i.  
- **Relational Edges**: Weighted by resonance.  
- **Tooling Integration**: Augment with external fetches.

## 4. Glossary

### Fundamental Building Blocks
- Node, ω, φ, spin, intrinsic time.

### Resonance & Interaction Metrics
- Resonance kernel ℛ_ij; beat-frequency r_ij.

### Continuum Mapping
- ρ_s = (3/2) J a_lat^{2-d}.  
- Gate field G(x).  
- TV penalty K'_TV ∥∇ω∥_1 / Δω*.

### Observer Concepts
- Observer node/frame.  
- Δω*-observer.

### Emergent Dimensionality & Critical Bandwidth
| δω/Δω* Range | Dimensionality | Regime / Symmetry |
|--------------|----------------|-------------------|
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
- **U(1)**: B_ij = a A_ij; U_ij = e^{i(Δφ_ij - B_ij)}.  
- **SU(2)**: ψ_i=[z_i, i s_i z_i]^T; rotations R(θ); Witten anomaly → even doublets.  
- **U(1)^2**: φ_1, φ_2, φ_3 (2 indep.); ε cosmological ≈ SU(3).  
- Exchange: sin(Δφ - B_ij) or -cos equiv.  
- Anomalies: U(1) sums=0; SU(2) Witten.

#### EFT Construction
- Minimal RTG-EFT: H_lat → L_EFT dim-4 (Maxwell).  
- L_EFT: (κ_B/4) F_{μν}F^{μν} + ...  
- C_κ per window (MC).  
- Anomalies: Ward residuals ~0.  
- Exchange fraction: 1.083–1.514% (μ/Δω*).  
- Dim-6 operators deferred.

## 5. Core Principles
- Node Primitives: as Glossary.  
- Emergent Space-Time: Δτ ≈ Δφ/Δω; photon propagation.  
- Lattice-continuum: Σ G_ij (Δφ_ij)^2 → a_lat^{2-d} ∫ G |∇φ|^2; ρ_s=(3/2)J a_lat^{2-d}.  
- Note: +8% J retune recovers r_p=0.84 fm (see findings.yaml).  
- Observer Dependence: relational.  
- Emergent Dimensionality: as table.  
- Equilibrium & Mass: Proton binding ~48 MeV.  
- Gauge: Δφ → Δφ - B_ij; SU(2) rotations; U(1)^2 ε≈SU(3).  
- EFT: U_ij → Maxwell; scaling predictions.

## 6. Key Equations
- **Bond Energy E_ij**: K'|Δω|/Δω* + J ℛ_ij + J_ex sin(Δφ - B_ij) exp[-(Δω)^2/σ_exch^2]; sin/-cos equiv.; σ_exch ≠ σ_noise.  
- **Spin Flip Energy**: ΔE_J = 2J σ_i Σ A_ij σ_j.  
- **Curvature Penalty U_curv**: Helfrich forms.  
- **S_B**: (κ_B/2) Σ_□ (curl B)^2.  
- **Continuum Action S**: ∫ [ρ_s/2 G (∂φ)^2 + κ(1-G)] + K'_TV ∥∇ω∥_1/Δω* + J_ex sin(...).  
  - L2 norm alt: use ||·||_2 for non-local.  
- **EFT L_EFT**: (κ_B/4) F F + ...  
- **Π_ex(0)**: (1/6π^2)(J_ex^2/K') μ^3.  
- **Π_ex(ω)** ≈ Π_ex(0)[1+(1/2)(ω/μ)^2].

## 7. Simulation Benchmarks
- CHSH U(1): 2.827±0.002.  
- U(1)^2: coherence decay.  
- Ward residuals: (0.10±0.97)×10^-3.  
- Exchange fraction: 1.083–1.514%.  
- C_κ SU(2): ±5.5e-7.  
- Drift <4.3×10^-4.  
- Flip rates 0.02–0.03.

## 8. Applications & Open Questions
- Apps: Hadrons, Bell violations, cosmology. QED (U(1)), weak SU(2), strong SU(3) ε. EFT Π_ex scattering.  
- Open: Spin quantization (±i→Dirac/SU2); domain boundaries; B_ij gradient experiments; ε cosmological; dim-6 >1.70; SU(3) embed; μ/Δω* probes.

## 9. Maintenance
- Live Diff Tracker; updates append-only.  
- Notation guard: ω angular rad·s^-1; σ_exch ≠ σ_noise; U(1) global/gauged; sin/-cos equiv.; spins in gauges; EFT amp=none baseline.

**End of Core Notes (v1.8.1)**.
