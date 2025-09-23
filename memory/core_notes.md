# Relational Time Geometry (RTG) Core Notes

**Version**: v1.4  
**Last Revised**: 2025-09-23  
**Author**: Mustafa Aksu (with Grok & ChatGPT contributions)  
**Purpose**: Authoritative anchor for RTG principles, workflows, and key concepts. Overrides chat history; use as Tier 1 static core. For evolutions/experiments, consult Tier 2 (findings.yaml, rtg_articles_index.yaml). This version merges prior sections for conciseness—no redundancies; derivations folded into principles/equations.

## 1. Introduction to RTG
Relational Time Geometry (RTG) models the universe as a graph of oscillatory nodes, where space-time, mass, and interactions emerge from relational frequencies, phases, and spins. Core idea: Time is geometric, defined by beat-frequencies and phase-locking between primitives—no absolute background.

- **Key Constants** (calibrated via RG/MD; see findings.yaml for sim-derived refinements):  
  | Symbol | Value | Units | Notes |
  |--------|--------|--------|-------|
  | \(\Delta \omega^*\) | \(1.45 \pm 0.08 \times 10^{23}\) | rad·s⁻¹ | Critical bandwidth; RG-fixed. |
  | \(c\) | 299792458 | m/s | Speed of light (fixed). |
  | \(\hbar\) | \(1.0545718 \times 10^{-34}\) | J·s | Reduced Planck constant. |
  | Proton radius | \(0.84 \pm 0.01\) | fm | Calibration target. |
  | \(g^*\) | \(\approx 1.14 \pm 0.02\) | - | RG fixed point. |
  | \(\sigma_{\rm crit}\) | \(\approx 0.589\) | - | CHSH decoherence threshold. |
  | \(K'\) | \(12.0\) (±0.5%) | MeV | Frequency penalty scale. |
  | \(J\) | \(3.24 \pm 0.12\) | MeV | Resonance coupling. |
  | \(J_{\rm ex}\) | \(2.20 \pm 0.08\) | MeV | Exchange coupling. |
  | \(\alpha\) | \(\approx 938\) | MeV·fm² | Mass form factor. |
  | \(\kappa_c\) | \(1\) | MeV·fm | Curvature scale. |
  | Proton binding | \(\approx 48 \pm 3\) | MeV | 3-node equilibrium. |

- **Guiding Principles**: Minimal decoherence; phase-locking for stability; RG-anchored scales (two-loop \(\beta_g \approx 0.72 g - 0.63 g^2 - 0.011 g^3\)); emergent phenomena from relations only.

## 2. Memory Architecture
Hierarchical storage for evolvability:  
- **Tier 1 (Static Core)**: This document—principles, formulas, constants. Overrides transients.  
- **Tier 2 (Dynamic Logs)**: findings.yaml (insights/challenges); rtg_articles_index.yaml (resources).  
- **Tier 3 (Active Memory)**: Session-only (e.g., query states); resets on load.  

**Retrieval Mechanics**: Chunked (512-token blocks) with relational edges; RAG-hybrid for tools (e.g., web_search filtered via geometry nodes).

## 3. Core Mechanics
- **Geometry Nodes**: Primitives with \(\omega_i\) (frequency/tick-rate, \(E_i = \hbar \omega_i\)), \(\phi_i\) (phase for interference), \(s_i = \pm i\) (analytic spin) or \(\sigma_i = \pm 1\) (code spin, \(s_i \equiv i \sigma_i\)); \(t_i = \tilde{\phi}_i / \omega_i\) (intrinsic time, unwrapped under phase-locking).  
- **Relational Edges**: Weighted by resonance; enable non-linear time traversal.  
- **Tooling Integration**: Augment with external fetches (e.g., browse_page), anchored to Tier 1.

## 4. Glossary
Integrated reference for RTG terms, aligned with published sources. Structured by category; cross-references principles/equations.

### Fundamental Building Blocks
- **Node**: Primitive: frequency \(\omega_i\), phase \(\phi_i\), binary spin \(s_i = \pm i\) (analytic) or \(\sigma_i = \pm 1\) (code).  
- **Frequency \(\omega_i\)**: Intrinsic tick-rate; local energy \(E_i = \hbar \omega_i\).  
- **Phase \(\phi_i\)**: Interference between nodes (constructive/destructive).  
- **Spin \(s_i\) or \(\sigma_i\)**: Conventions (analytic vs. code); determines gate (open/closed) on alignment. Truth table (analytic):  
  | \(s_i, s_j\) | Gate \(1 + s_i s_j\) | State |
  |--------------|----------------------|-------|
  | \(+i, +i\)   | 0                    | Closed |
  | \(+i, -i\)   | 2                    | Open  |  
  Code equivalent: \(1 - \sigma_i \sigma_j\) (open for opposite).  
- **Intrinsic time \(t_i\)**: \(t_i = \tilde{\phi}_i / \omega_i\); valid under local phase-locking (minimal decoherence).

### Resonance & Interaction Metrics
- **Resonance kernel \(\mathcal{R}_{ij}\)**: \(\mathcal{R}_{ij} = A_{ij} (1 + s_i s_j)\), \(A_{ij} = \frac{3}{4} [1 + \cos(\phi_i - \phi_j)] \exp[-(\omega_i - \omega_j)^2 / (\Delta \omega^*)^2]\) (range \(0 \leq \mathcal{R}_{ij} \leq 3\)).  
- **Beat-frequency distance \(r_{ij}\)**: \(r_{ij} = \frac{2\pi c}{|\omega_i - \omega_j|}\); coarse-grain as \(\Delta \omega \to 0\) (avoids divergence).  

### Observer Concepts
- **Observer node/frame**: Relational; dependence via \(r_{ij}\) to observer.  
- **\(\Delta \omega^*\)-observer**: Reference with \(\omega_{\rm ref} = \Delta \omega^* \approx 1.45 \times 10^{23}\) s⁻¹ (uncertainty ~5%).

### Emergent Dimensionality & Critical Bandwidth
- **Spatial degree of freedom**: Emerges from resonance patterns (planar 2D vs. volumetric 3D) above \(\Delta \omega^*\) threshold. Cayley-Menger volumes for embedding.  
  | \(\delta \omega / \Delta \omega^*\) | Dimensionality | Regime |
  |-----------------------------------|----------------|--------|
  | < 0.28 (±0.02)                    | 2D             | Planar |
  | 0.28–0.70                         | 3D             | Shells |
  | 0.70–1.55                         | 4D             | Hyper  |
  | 1.55–1.70                         | 5D             | Anomalies |
  | >1.70                             | >5D            | Unstable |

### Stability & Guiding Principles
- **Stability criteria**: Phase-locking for \(t_i\); \(\partial H / \partial q = 0\) (\(q \in \{\omega, \phi\}\)); \(|\Delta \omega| \ll \Delta \omega^*\) for bonds; modest spin flips (acceptance ~0.02–0.03 per tick).  
- **Guiding**: Anti-aligned spins for open gates; drift < \(4.3 \times 10^{-4}\) over 3000 ticks; thermostat ≤0.30.

### Special Node Types & Dynamic Factors
- **Photon object**: High-frequency (\(\omega_i \gg \Delta \omega^*\)) spin-anti-pair (\(+i, -i\)); \(\Delta \omega = 0\), massless \(E = \hbar \omega_\gamma\), weak phase coupling.  
- **Dynamic factors**: Evolving params (e.g., \(\sigma_{\rm exch}(t)\) under noise).

## 5. Core Principles
Merged foundations for emergence, calibrated to observables (e.g., proton radius 0.84±0.01 fm, CHSH≈2.827). Universe as node graph; RG-fixed \(\Delta \omega^*\); two-loop anchors ratios (g*=1.14).  

- **Node Primitives**: As in Glossary; energy/momentum via \(\hbar \omega_i\).  
- **Emergent Space-Time**: Proper time \(\Delta \tau \approx \Delta \phi / \Delta \omega\); photon as open-gate pair, propagates massless. Lattice mapping for d=3 (K'/J≈3.70).  
- **Observer Dependence**: As in Glossary; relational distances \(r_{io}\).  
- **Emergent Dimensionality**: As in Glossary table.  
- **Equilibrium & Mass**: Resonance energy \(E^{\rm res}_{ij} = \hbar |\Delta \omega_{ij}| \mathcal{R}_{ij}\); rest mass \(m_i = \frac{\hbar \omega_i - \sum_j E^{\rm res}_{ij}}{c^2}\) (clamp \(m_i \geq 0\)); proton via \(\alpha \sum \mathcal{R}_{ij} / r_{ij}^2\) (~48 MeV binding, 3-node).  

## 6. Key Equations
Central formulas; derivations in principles above. Dimensionless params: \(\tilde{J} = J / (\hbar \Delta \omega^*)\), \(g = \tilde{J}/\tilde{K}'\); flows \(\beta_{\tilde{J}} = -\tilde{J} + O(\tilde{J}^3)\), \(\beta_{\tilde{K}} = -\frac{1}{2} \tilde{K} \tilde{J} + O(\tilde{J}^3)\).  

- **Bond Energy \(E_{ij}\)**: \(E_{ij} = K' \frac{|\omega_i - \omega_j|}{\Delta \omega^*} + J \mathcal{R}_{ij} - J_{\rm ex} \cos(\Delta \phi_{ij} - 2\pi a A_{ij}) \exp[-(\omega_i - \omega_j)^2 / \sigma_{\rm exch}^2]\) (\(\sigma_{\rm exch} \simeq \Delta \omega^*\); \(a\) gauge param; U(1)-like cos term).  
- **Spin Flip Energy**: \(\Delta E_J = 2J \sigma_i \sum_j A_{ij} \sigma_j\); conserve via \(\pi_{\phi_i} \leftarrow \pi_{\phi_i} - \sign(\Delta E) \sqrt{2 M_\phi |\Delta E|}\).  
- **Curvature Penalty \(U_{\rm curv}\)**: \(U_{\rm curv} = \kappa_c a_{\rm lat}^2 \sum (1 - \frac{\mathcal{R}_{ij} + \mathcal{R}_{jk} + \mathcal{R}_{ki}}{9})^2\) (Helfrich); alt: \(\kappa_c (1 - \mathcal{R}/3)^2 / a^2\) (units MeV·fm).

## 7. Simulation Benchmarks
Canonical values; see findings.yaml for full runs (e.g., drift logs).  
- **CHSH Violation**: \(S = 2\sqrt{2} e^{-\sigma^2}\) (\(\sigma < 0.589\); noise=0: 2.827±0.002; 0.5: 2.20±0.02; OU decoherence).  
- **Drift Bound**: <4.3×10^{-4} (\(\Delta t = 5 \times 10^{-5}\)).  
- **Flip Rates**: 0.02–0.03 per tick (thermostat ≤0.30).

## 8. Applications & Open Questions
- **Applications**: Hadrons (shell fits to proton); quantum info (Bell violations); cosmology (large-scale sweeps, dark energy as residual elasticity).  
- **Open Questions**: Spin quantization (path-integral/Grassmann); full U(1) gauges; scale to 10^6 nodes.

## 9. Maintenance
- **Live Diff Tracker**: Git diffs on YAMLs; inline flags in responses (e.g., "Added: new_term").  
- **Updates**: Version on changes; no re-summaries—append only.  
- **Self-Documenting**: Headers, tables for scanability.  

**End of Core Notes**. For diffs, see Git log. This v1.4 reduces length by ~40% via merging; next: derivations or Tier 2 tweaks?
