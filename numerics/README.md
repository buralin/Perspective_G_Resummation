# numerics: Hermitian pole-space MR-GW with mixed inactive-active couplings

Pilot implementation of the common Hermitian matrix of Eq. (7) in
`notes/comparison.tex`,

```
      | F_ii + Sigma_stat,ii    K_B^{ia}        A_i     |
  H = | K_B^{ai}                E              M^+ A_a  |     G(z) = (1  M  0) (z - H)^{-1} (1  M  0)^+
      | A_i^+                   A_a^+ M        E_GW     |
```

which embeds, in one Hermitian matrix,

* the exact Dyall (CAS) reference propagator: inactive orbital poles
  (Dyall Fock energies) and the active N±1 poles with their Dyson amplitudes `M`,
* the one-shot MR-GW self-energy of Wang, Fang, and Li (arXiv:2604.16013)
  as a diagonal pole bath `E_GW` coupled through the amplitudes `A`,
* the mixed inactive-active couplings `K_B` of Objective II in
  `notes/mr-hedin.tex`, either bare (`vbar_R`) or statically screened with the
  parent-consistent kernel `v_R - [W_D(0)]^x`, plus the static first-order
  self-energy `Sigma_1^{11}`.

Setting `K_B = 0` recovers MR-GW exactly; dropping the bath and using the bare
kernel recovers the block-matrix Green's function of the preliminary section of
`mr-hedin.tex`.

## Layout

| file | content |
|---|---|
| `mrgw_hermitian/dyall.py` | molecule, RHF, CASSCF (PySCF); canonicalization of core/virtual orbitals with the Dyall Fock operator; spin-orbital integrals; Dyall partition `H = H_D + V_R`; dense diagonalization of the active N_a, N_a±1 sectors (PySCF `direct_spin1`); charged poles, Dyson amplitudes, one-body density, `Sigma_1^{11}`, three-operator amplitudes (PySCF `fci.addons`) |
| `mrgw_hermitian/mrrpa.py` | four-channel MR-RPA of the residual interaction (Wang–Fang–Li Secs. S2.A–S2.D), screened couplings `M_{pr,I}`, static `W_D(0)`, MR-GW pole bath (their Eq. S52) |
| `mrgw_hermitian/hermitian.py` | assembly of `K_top`, the bath coupling and `T`; matrix-free `matvec`; dense diagonalization; Green's function via the bath Schur complement; spectral function; spectral moments; guess vectors. Also the (non-PSD) Hall-form insertion of the first-order mixed blocks for comparison |
| `mrgw_hermitian/davidson.py` | block Davidson with root following (maximum-overlap selection of the Ritz vectors) |
| `mrgw_hermitian/fci_reference.py` | exact full-CI Green's function; embedding of active-space CI vectors in the full orbital space; two independent checks of the coupling matrix |
| `run_checks.py` | consistency checks on H4/STO-6G |
| `example_h4.py` | the H4 example (Fig. 6 setting of Wang–Fang–Li) |

## Requirements

NumPy, SciPy, PySCF (tested with 2.12.1) and, for the plot, matplotlib.  On this
machine the miniforge base environment has all of them:

```
~/miniforge3/bin/python run_checks.py
~/miniforge3/bin/python example_h4.py            # STO-6G, R = 1.0 A, CAS(2,2)
~/miniforge3/bin/python example_h4.py --basis 6-31g
```

## Conventions

* Spin orbitals `p = 2*p_spatial + sigma` (`sigma = 0` alpha, `1` beta).
* `V[p,r,q,s] = v_{pr,qs} = <pq|rs> = (pr|qs)`, `vbar_{pr,qs} = v_{pr,qs} - v_{ps,qr}`
  (Wang–Fang–Li convention).  `V_R` is `V` with the all-active elements removed.
* Retarded Green's function throughout: removal poles carry the signed energy
  `kappa = -(E^{N-1} - E^N_0)`, attachment poles `kappa = E^{N+1} - E^N_0`.
* Active-pole vectors `d_alpha` and the connected three-operator tensors
  `C^{D,±}` are computed from the same eigenvectors, so state phases cancel.

## What the checks verify (`run_checks.py`)

1. `h_eff` equals PySCF's `h1e_for_cas`; `E_0(active) + E_core` equals the CASSCF energy.
2. Sum rules `d d^T = 1` and `T_D T_D^T = 1` (zeroth moment).
3. Block structure of `Sigma_1^{11}` (zero active-active, core-core, virtual-virtual blocks).
4. MR-RPA: symmetric A/B, positive excitation energies, `(X+Y)^T (X-Y) = 1`, pair symmetry of `W_D(0)`.
5. The Hermitian matrix with `K_B = 0` reproduces `[G_D^{-1} - Sigma^{MR-GW}(z)]^{-1}` exactly.
6. The same-sector blocks of `K_top` (bare kernel) equal the Hamiltonian matrix elements between the
   orthonormal charged Dyall states `a_a^+|Psi_0>`, `|Phi_I>|Xi_mu^{N+1}>` (and, with a minus sign,
   `a_i|Psi_0>`, `|Phi_I>|Xi_mu^{N-1}>`), computed in the full determinant space.
7. `dG/dlambda` of the full-CI Green's function of `H_D + lambda V_R` (finite differences) equals
   `T_D R_D K^(1) R_D T_D^+`, which validates the complete first-order structure including the
   cross-sector couplings (core with N+1 states, virtual with N-1 states).
8. The screened mixed extension is PSD, has `M0 = 1`, and its first moment with the bare kernel equals
   `h + vbar gamma_D` (first-order exactness of `M1`).
9. Davidson root following reproduces the dense max-overlap eigenvalues.

## Example output (H4, R = 1.0 A, STO-6G, CAS(2,2); energies in eV relative to the N-electron ground state)

Roots obtained by Davidson root following (identical to the dense max-overlap eigenvalues):

| method | principal IP (HOMO-like) | principal EA (LUMO-like) | lowest-sigma IP | virtual EA |
|---|---|---|---|---|
| CAS (Dyall reference) | -11.134 | 8.571 | -16.997 | 23.372 |
| MR-GW | -10.878 | 8.524 | -15.938 | 26.343 |
| MR-GW + mixed (bare) | -10.863 | 8.504 | -14.973 | 26.752 |
| MR-GW + mixed (screened) | -10.861 | 8.505 | -14.944 | 26.765 |
| full CI (most intense poles) | -11.086 | 8.786 | -16.027 | 23.930 |

All Hermitian variants have positive spectral weights and `M0 = 1`; the Hall-form
insertion of the first-order mixed blocks (last panel of the figure) reaches
`A(omega) = -4.3 1/eV`, the non-PSD behaviour reported in Fig. 6 of Wang, Fang, and Li.
`h4_sto-6g_spectral_functions.png` and `h4_6-31g_spectral_functions.png` are the plots
produced by `example_h4.py` (the 6-31G run takes about 10 s).

## Ozone check against Wang, Fang, and Li (`check_ozone.py`)

`check_ozone.py` rebuilds the O3 setting of the paper (CCCBDB geometry R = 1.278 A,
theta = 116.8 deg, 6-31G, CASCI with RHF orbitals) and compares with their Tables S3-S5:

```
~/miniforge3/bin/python check_ozone.py --cas 6,4          # 2 s, bath of 4.4e4 poles
~/miniforge3/bin/python check_ozone.py --cas 8,5          # 5 s, bath of 1.0e5 poles
~/miniforge3/bin/python check_ozone.py --cas 6,4 --plot   # + spectral functions, ~30 s
```

The RHF orbital energies agree with their Table S3 to 1e-5 Ha (same C2v labels), the
leading CASCI(6,4) coefficients are +0.936/-0.349 as in the paper, and the CASCI cation
energies match their CAS rows.  The three lowest vertical IPs are obtained by Davidson
root following from the CASCI cation states (12 iterations, < 1 s for a 1e5-dimensional
matrix).  With `K_B = 0` the Hermitian matrix is exactly the MR-GW Dyson equation of the
paper, and its roots reproduce their MR-GW values:

| 6-31G, IPs in eV | 2A1 | 2B2 | 2A2 |
|---|---|---|---|
| CAS(6,4), paper / this work | 14.38 / 14.38 | 14.67 / 14.67 | 15.12 / 15.12 |
| MR-GW(6,4), paper / this work | 12.65 / 12.65 | 12.94 / 12.94 | 14.56 / 14.56 |
| MR-GW(6,4) + mixed (bare) | 12.68 | 12.90 | 14.57 |
| MR-GW(6,4) + mixed (screened) | 12.68 | 13.21 | 14.99 |
| CAS(8,5), paper / this work | 13.59 / 13.59 | 13.92 / 13.93 | 13.10 / 13.10 |
| MR-GW(8,5), paper / this work | 11.58 / 11.58 | 12.09 / 12.09 | 12.94 / 12.94 |
| MR-GW(8,5) + mixed (bare) | 11.79 | 12.08 | 12.94 |
| MR-GW(8,5) + mixed (screened) | 12.01 | 12.27 | 13.32 |
| DMRG (paper) | 12.34 | 12.59 | 13.34 |
| Exp. (paper) | 12.73 | 13.00 | 13.54 |

Deviations between this implementation and the paper's MR-GW numbers are at most 0.005 eV
(their values are read off a frequency grid).  The mixed couplings leave the state ordering
unchanged.  With the bare kernel they shift the IPs by at most 0.2 eV; the statically
screened kernel shifts the 2B2 and 2A2 peaks upward by 0.3-0.4 eV, which brings all three
CAS(8,5) values closer to DMRG (errors -0.33, -0.32, -0.02 eV instead of -0.76, -0.50,
-0.40 eV for MR-GW) but moves the CAS(6,4) values further away.  The quasiparticle weights
are 0.70-0.88.

CAS(8,6) is refused by default: its bath would have about 4e6 poles, beyond the dense
storage of `T_D^+ A` used here (a factorized application of `A` would be needed, as for
the paper's own dense pilot implementation).
