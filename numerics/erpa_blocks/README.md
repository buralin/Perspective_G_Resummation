# erpa_blocks: numerical verification for `notes/erpa_channels.tex`

Brute-force checks of the extended-RPA (ERPA) matrix elements written in
`notes/erpa_channels.tex`: the particle-hole and particle-particle double
commutators and metrics for an arbitrary vacuum in terms of the 1- and 2-RDM,
and their resolution into core/active/virtual blocks for a CAS-type vacuum.
Everything is done with explicit second-quantized operators in a small Fock
space (dense matrices, at most 8 spin orbitals), so no approximation enters.

* `fockspace.py` - annihilation operators, random integrals with the
  eightfold symmetry, Hamiltonian, 1-/2-RDM of a state vector.
* `gen_terms.py` - generates the raw term lists of the ph (20 terms) and pp
  (72 terms) double commutators with `mrgw_hermitian/wick.py`.
* `verify_general.py` - checks the compact closed forms (Gamma form, cumulant
  form, symmetrized and unsymmetrized, the Lagrangian/Brillouin identity, the
  metrics) for a random non-eigenstate vacuum (7 spin orbitals, 3 electrons).
* `blocks.py` - resolves the general formulas into index-type blocks for a
  CAS-type product vacuum (2 core, 4 active with 2 electrons, 2 virtual spin
  orbitals), checks every block numerically against the brute-force reference
  and writes the LaTeX of all blocks to `blocks_out.tex`.
* `format_blocks.py` - the same with line breaks and labels, as pasted into
  the note (`blocks_formatted.tex`).

Run with a Python that has numpy (e.g. the miniforge base environment):

    python verify_general.py
    python blocks.py
    python format_blocks.py

All reported deviations should be of order 1e-15.
