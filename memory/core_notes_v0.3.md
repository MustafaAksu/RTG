# Relational Time Geometry (RTG) – Core Notes v0.3

**Revision date:** 23 Sep 2025  
**Authors:** Mustafa Aksu, ChatGPT (OpenAI), Grok (xAI)  
**Scope:** Tier 1 (Core Notes, ~2–3k tokens) for active session use

---

## 1. Methodology & Organization

- **Publishing site:** [https://rtgtheory.org](https://rtgtheory.org)  
  Contains full-length articles (Tier 3). Any document published must be in **simple HTML**:  
  - No `style`, `div`, `figure` tags.  
  - No manual table of contents (auto-generated).  

- **GitHub repository:** [https://github.com/MustafaAksu/RTG](https://github.com/MustafaAksu/RTG)  
  Holds scripts, raw outputs, structured memory files (Tier 2).  
  Recommended path: `/memory/core_notes.md` and `/memory/findings.yaml`.  

- **Tiering system:**  
  - **Tier 1:** This Core Notes document (short, ~2–3k tokens). Inject at start of sessions.  
  - **Tier 2:** Structured archive (YAML/JSON), updated as new findings stabilize. Used via retrieval.  
  - **Tier 3:** Full articles (WordPress). Long derivations, history, public-facing.  

- **Collaboration principle:**  
  AI (ChatGPT, Grok) and Mustafa Aksu co-develop iteratively.  
  Findings are first captured in chat, then distilled into Tier 2 → Tier 1.  
  Only stable, reusable knowledge is promoted upward.  
  Speculative ideas stay in Tier 2/3 until validated.  

---

## 2. Core Definitions & Notation

```yaml
- id: node
  type: definition
  value: "Fundamental entity defined by (ω, φ, s) where ω = frequency, φ = phase, s = spin. Properties only meaningful relationally."
  tags: [core, definition]

- id: resonance_kernel
  type: definition
  value: "R_ij encodes coupling between nodes i and j; bounded, spin-gated function of Δω, Δφ."
  tags: [core, kernel]

- id: observer_distance
  type: definition
  value: "Relational distance r_ij = (2πc)/|ω_i − ω_j|; observer-relative, not absolute."
  tags: [geometry]

- id: intrinsic_time
  type: definition
  value: "Time is defined relationally via phase-locking; single node has no measurable time."
  tags: [time]

- id: core_principles
  type: definition
  value: "RTG principles: no absolute observables, distance/time emerge via resonance, gauge invariance is emergent, stability from bounded kernels."
  tags: [principles, foundation]

- id: enriched_geometry
  type: definition
  value: "Curvature elasticity and torsional corrections are part of emergent geometry; anisotropy in resonance networks biases curvature."
  tags: [geometry, enrichment]

- id: forces_fields
  type: definition
  value: "Forces = resonance gradients; electromagnetism from photon nodes; nuclear forces from higher-order resonance; gravity as residual elasticity."
  tags: [forces, fields]
```

---

## 3. Core Equations & Kernels

```yaml
- id: mass_estimator
  type: equation
  value: "m_i = [ħ ω_i − Σ_j E_res(ij)] / c^2"
  tags: [mass, estimation]

- id: kernel_energy
  type: equation
  value: "E_res(ij) = A_ij (1 + s_i s_j) f(Δω_ij, Δφ_ij)"
  tags: [resonance, energy]

- id: curvature_penalty
  type: equation
  value: "E_curv ∝ κ (H − H0)^2"
  tags: [geometry, curvature]
```

---

## 4. Constants & Parameters

```yaml
- id: delta_omega_star
  type: constant
  value: "1.45e23 rad/s (±0.08e23)"
  tags: [critical, bandwidth]

- id: rg_fixed_point
  type: constant
  value: "g* ≈ 1.14 ± 0.02"
  tags: [renormalization, fixed_point]

- id: proton_radius
  type: constant
  value: "≈ 0.84 fm (empirical reference)"
  tags: [proton, reference]
```

---

## 5. Operational Rules

```yaml
- id: relational_only
  type: rule
  value: "No single-node observables; all properties emerge from interactions."

- id: kernel_bounded
  type: rule
  value: "Always use bounded, spin-gated kernels; avoid ad-hoc rescalings."

- id: log_diagnostics
  type: rule
  value: "Track Δφ, Δω/Δω*, correlation energies, flip rate in all simulations."

- id: curvature_monitoring
  type: rule
  value: "Include curvature penalties; monitor drift over cycles."

- id: cosmology_expansion
  type: rule
  value: "Cosmological expansion in RTG is explained as frequency drift, not metric stretch."

- id: eft_mapping
  type: rule
  value: "RTG can map to minimal EFT by coarse-graining resonance kernels."
```

---

## 6. Stable Findings

```yaml
- id: proton_stability
  type: finding
  date: 2025-05-06
  value: "Proton stable at ~3.0581e23 Hz with phase cycling; radius preserved, charge +1e."
  tags: [proton, stability]

- id: kaon_kplus
  type: finding
  date: 2025-05-05
  value: "K+ meson modeled at 3.089e23 Hz, radius 5.6e-16 m; matched experimental mass."
  tags: [kaon, meson, validation]

- id: proton_model
  type: finding
  date: 2025-05-06
  value: "Proton modeled as 3-node SU(2) bound state; stability via oscillatory resonance and phase cycling."
  tags: [proton, SU2, stability]

- id: two_loop_derivation
  type: finding
  date: 2025-05-06
  value: "Two-loop RG derivation supports Δω* ≈ 1.45e23 rad/s critical bandwidth."
  tags: [RG, constants, derivation]
```

---

## 7. Open Questions

```yaml
- id: dimensionality_emergence
  type: open_question
  value: "Mapping of δ ω / Δω* thresholds to emergent dimensionality (D=2,3,4)."

- id: black_hole_modeling
  type: open_question
  value: "How do RTG kernels behave near Planck wall / singularities?"

- id: cosmology_growth
  type: open_question
  value: "Is universal frequency shift sufficient to explain observed expansion?"

- id: black_hole_resolution
  type: open_question
  value: "How RTG handles gravitational collapse: resonance saturation + curvature penalty halting collapse at Planck wall."

- id: energy_framework
  type: open_question
  value: "RTG cosmology defines relational energy tensor; dark energy as resonance imbalance. Energy conservation only approximate."

- id: four_d_corridors
  type: open_question
  value: "Transient 4-D corridors emerge under high resonance density; not yet gauged, unstable structures."

- id: rtg_to_eft
  type: open_question
  value: "Construction of minimal EFT from RTG by coarse-graining resonance kernels; testable predictions possible."

- id: dimensionality_thresholds
  type: open_question
  value: "Thresholds in Δω / Δω* mark transitions between effective dimensions; coherence can add relational axes."
```
