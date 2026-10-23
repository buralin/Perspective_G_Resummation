# erpa_ccd: extended (ERPA) channel equations as a CC-type amplitude equation

Companion code of `notes/erpa_ccd.tex`.  Builds the particle-hole, crossed
particle-hole and particle-particle ERPA matrices on a CASSCF vacuum
(formulas of `notes/erpa_channels.tex`, verified in `numerics/erpa_blocks/`),
assembles the metric-dressed channel Riccati residuals, combines them in the
fermionic doubles space following the common-amplitude construction of
`notes/Pa2CCD.tex`, and evaluates the correlation energy E_c = 1/4 sum B t.

* `erpa_ccd.py` - everything: spin-orbital integrals, HF and CASSCF vacua
  (full-space 1-/2-RDM from PySCF `make_rdm12s`, with antisymmetry/trace/energy
  checks), ERPA arrays, amplitude space (`max_active` restricts the number of
  active indices per double), channel residual pieces, DIIS solver, energy,
  pure-ERPA ring eigen-solution.
* `nonmosaic_ref.py`, `test_hf_limit.py` - independent non-mosaic CCD residual
  and the HF-limit identity (residuals agree to 1e-13; 2nd order = MP2).
* `class_exact.py`, `class_curve.py` - brute-force (full determinant space)
  exact overlaps/Hamiltonian blocks of internally contracted doubles classes
  on the CASSCF vacuum: exact-metric second order (full H in the doubles
  block) and class-CI energies; also the exact vs quasi-boson doubles norms.
* `n2_curve.py <basis> <max_active>` - N2 dissociation curve with CASSCF(6,6):
  FCI (STO-3G), SC-NEVPT2 (class resolved) and the extended channel variants
  (2nd order, ring+crossed ring, ladder, all).
* `cas22_test.py` - N2/STO-3G with the sigma CAS(2,2): exact reference for the
  classes with <= 1 active index vs the quasi-boson channel equations.
* `probe_631g_classes.py` - class-resolved 2nd order vs SC-NEVPT2 in 6-31G.
* `make_plots.py` - figures; `results/*.json` - the data behind the tables.

Run with the PySCF conda environment (`miniforge3/envs/PySCF/bin/python`).

* `n2_631g_hf.py <R1,R2,...> <tag>` - the same four variants on a plain RHF
  determinant (no active space) in 6-31G; `plot_totals.py` draws the
  total-energy curves for the RHF and the CASSCF(6,6) vacuum
  (`results/n2_631g_hf.json`, `results/n2_631g_ma1.json`).

Convergence diagnostics (Sec. "Why the equations stop converging" of the note):
* `diag_stability.py` - RPAx/TDA stability (min eig of A, A-B, A+B, lowest omega) of
  the RHF reference and of the external ERPA manifold on the CASSCF(6,6) vacuum, 6-31G.
* `diag_linear.py` - lowest eigenvalues of the linearized equations (Jacobian at t=0,
  preconditioned by the second-order denominators) for the four variants.
* `diag_newton.py <R> <basis>` - Newton-Krylov attempt and Jacobian spectra for the
  non-converging variants on the RHF reference.
* `diag_mosaic.py <basis>` - non-mosaic CCD vs full CCD (mosaic terms added in
  `nonmosaic_ref.py`) vs PySCF CCSD along the RHF N2 curve.

* `active_space_scan.py <basis> <R1,R2> <nel/ncas,...>` - dependence on the active-space size: external
  correlation and its NEVPT2 class split, the one-active-index channel variants, external ERPA
  stability and the quasi-boson norm ratios (`results/active_space_scan_*.json`).
