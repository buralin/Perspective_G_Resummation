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
`mr-hedin.tex` (its Eq. (7)).  With `HermitianGF(..., bath_hamiltonian='top')` the bath states
`|n> x |1_I>` are built for all top states `n` and the top-space Hamiltonian `K` acts inside the
one-boson sector (`B_I = K + sigma Omega_I`, couplings `g_I = T_D^+ M_I T_D`); this is the strict
fermion-plus-boson supermatrix of Eq. (7) combined with the MR-RPA bosons (Sec. II of
`notes/mr-hedin.pdf`, Eq. (44)), and without cross-sector couplings its bath Schur complement is the
MR-GW self-energy built from the Eq. (7) propagator (Eq. (45)).  The default `'diagonal'` bath keeps
the zeroth-order energies `kappa_n +- Omega_I` (MR-GW form).

Every variant can be built on three active-space references: the exact CAS states, or
extended-Koopmans (EKT) charged states without or with the 2h1p/2p1h sector combined with an
extended-RPA (ERPA) active response (Sec. IV of `notes/mr-hedin.pdf`); see
"Supermatrix variants with the three references" below.

## Layout

| file | content |
|---|---|
| `mrgw_hermitian/dyall.py` | molecule, RHF, CASSCF (PySCF); canonicalization of core/virtual orbitals with the Dyall Fock operator; spin-orbital integrals; Dyall partition `H = H_D + V_R`; dense diagonalization of the active N_a, N_a±1 sectors (PySCF `direct_spin1`); charged poles, Dyson amplitudes, one-body density, `Sigma_1^{11}`, three-operator amplitudes (PySCF `fci.addons`) |
| `mrgw_hermitian/mrrpa.py` | four-channel MR-RPA of the residual interaction (Wang–Fang–Li Secs. S2.A–S2.D), screened couplings `M_{pr,I}`, static `W_D(0)`, MR-GW pole bath (their Eq. S52) |
| `mrgw_hermitian/hermitian.py` | assembly of `K_top`, the bath coupling and `T`; diagonal or dressed bath; matrix-free `matvec`; dense diagonalization; Green's function via the bath Schur complement; spectral function; spectral moments; guess vectors. Also the (non-PSD) Hall-form insertion of the first-order mixed blocks for comparison |
| `mrgw_hermitian/ekt_erpa.py` | `EKTERPAReference`: EKT charged states (primary manifold `{a_x}`, `{a_x^+}`, or extended by the 2h1p/2p1h operators) and ERPA neutral excitations of the active space; drop-in replacement for `DyallReference` |
| `mrgw_hermitian/wick.py` | symbolic normal-ordering engine: reduces expectation values of operator products (commutators with the active Hamiltonian) to `einsum` contractions of integrals with spin-orbital RDMs; LaTeX output of the term lists |
| `mrgw_hermitian/rdm.py` | spin-orbital 1- to 4-RDMs of the CAS wave function, Grassmann (shuffle) product, cumulant decomposition and cumulant-truncated reconstruction of the 3- and 4-RDM |
| `mrgw_hermitian/davidson.py` | block Davidson with root following (maximum-overlap selection of the Ritz vectors) |
| `mrgw_hermitian/fci_reference.py` | exact full-CI Green's function; embedding of active-space CI vectors in the full orbital space; checks of the coupling matrix and of the superoperator propagator |
| `run_checks.py` | 77 consistency checks on H4/STO-6G |
| `cumulant_comparison.py` | RDM realization of the extended EKT/ERPA reference versus the CI-vector realization, and the supermatrix variants with the 3- and 4-body cumulants dropped (H4 CAS(4,4), ozone) |
| `equations.tex` / `equations.pdf` | all implemented equations with index conventions, thresholds and the identities behind each check, plus an equation-to-function index (reimplementation guide) |
| `example_h4.py` | the H4 example (Fig. 6 setting of Wang–Fang–Li): eight variants, Davidson root following, pole tables, sum rules, spectral-function plots |
| `check_ozone.py` | ozone setting of Wang–Fang–Li: MR-GW, mixed variants, EKT/ERPA variants vs their Tables S3–S5 |
| `ginit_prelim.py` | the ten supermatrix variants of Tables I and II of `notes/mr-hedin.pdf` (Eq. (7) without bath, MR-GW, diagonal bath with bare/screened `K_B`, dressed bath with bare/screened `K_B`, with and without cross-sector couplings) for the exact CAS reference and for the two EKT/ERPA references, H4 and ozone |
| `results/` | outputs of all reference runs listed below (the numbers quoted here and in the notes) |

## Requirements

NumPy, SciPy, PySCF (tested with 2.12.1) and, for the plots, matplotlib.  On this
machine the miniforge base environment has all of them.  The reference runs
(outputs in `results/`):

```
~/miniforge3/bin/python run_checks.py                                       > results/run_checks.txt
~/miniforge3/bin/python example_h4.py                                       > results/example_h4_sto-6g_cas2_2.txt
~/miniforge3/bin/python example_h4.py --basis 6-31g                         > results/example_h4_6-31g_cas2_2.txt
~/miniforge3/bin/python example_h4.py --basis 6-31g --ncas 4 --nelecas 4    > results/example_h4_6-31g_cas4_4.txt
~/miniforge3/bin/python check_ozone.py --cas 6,4 --plot                     > results/check_ozone_6_4.txt
~/miniforge3/bin/python check_ozone.py --cas 8,5                            > results/check_ozone_8_5.txt
~/miniforge3/bin/python ginit_prelim.py                                     > results/ginit_prelim_h4.txt
~/miniforge3/bin/python ginit_prelim.py --o3                                > results/ginit_prelim_o3.txt
~/miniforge3/bin/python cumulant_comparison.py                              > results/cumulant_comparison_h4.txt
~/miniforge3/bin/python cumulant_comparison.py --o3                         > results/cumulant_comparison_o3.txt
```

`ginit_prelim.py` accepts `--reference cas|ekt|ekt2` (default: all three), `--response exact`
(EKT poles with the exact CAS excitations in the active-active channel) and, with `--o3`,
`--cas 6,4` / `--cas 8,5`.

## Conventions

* Spin orbitals `p = 2*p_spatial + sigma` (`sigma = 0` alpha, `1` beta).
* `V[p,r,q,s] = v_{pr,qs} = <pq|rs> = (pr|qs)`, `vbar_{pr,qs} = v_{pr,qs} - v_{ps,qr}`
  (Wang–Fang–Li convention).  `V_R` is `V` with the all-active elements removed.
* Retarded Green's function throughout: removal poles carry the signed energy
  `kappa = -(E^{N-1} - E^N_0)`, attachment poles `kappa = E^{N+1} - E^N_0`.
* Active-pole vectors `d_alpha` and the connected three-operator tensors
  `C^{D,±}` are computed from the same eigenvectors, so state phases cancel.

## What the checks verify (`run_checks.py`, 77 checks)

1. `h_eff` equals PySCF's `h1e_for_cas`; `E_0(active) + E_core` equals the CASSCF energy.
2. Sum rules `d d^T = 1` and `T_D T_D^T = 1` (zeroth moment).
3. Block structure of `Sigma_1^{11}` (zero active-active, core-core, virtual-virtual blocks).
4. MR-RPA: symmetric A/B, positive excitation energies, `(X+Y)^T (X-Y) = 1`, pair symmetry of `W_D(0)`.
5. The Hermitian matrix with `K_B = 0` reproduces `[G_D^{-1} - Sigma^{MR-GW}(z)]^{-1}` exactly.
6. The same-sector blocks of `K_top` (bare kernel) equal the Hamiltonian matrix elements between the
   orthonormal charged Dyall states `a_a^+|Psi_0>`, `|Phi_I>|Xi_mu^{N+1}>` (and, with a minus sign,
   `a_i|Psi_0>`, `|Phi_I>|Xi_mu^{N-1}>`), computed in the full determinant space.
7. The superoperator form of the retarded propagator, `(a_p^+|(z - H_super)^-1|a_q^+)` with the
   anticommutator binary product, equals the Lehmann representation (Eq. (18) of the notes).
8. `dG/dlambda` of the full-CI Green's function of `H_D + lambda V_R` (finite differences) equals
   `T_D R_D K^(1) R_D T_D^+`, which validates the complete first-order structure including the
   cross-sector couplings (core with N+1 states, virtual with N-1 states).
9. The screened mixed extension is PSD, has `M0 = 1`, and its first moment with the bare kernel equals
   `h + vbar gamma_D` (first-order exactness of `M1`).
10. Davidson root following reproduces the dense max-overlap eigenvalues.
11. Dressed bath: equals the diagonal bath for diagonal `K`; without cross-sector couplings equals the
    Dyson equation with the GW self-energy built from the Eq. (7) propagator; PSD, `M0 = 1`, Davidson vs dense.
12. EKT (primary): `M0 = 1`, `M1` of the active propagator exact, poles exact for CAS(2,2);
    EKT (extended): `M0`–`M3` exact, poles exact; ERPA zeroth and energy-weighted sum rules;
    EKT/ERPA Hermitian matrix: linearization identity, PSD, `M0 = 1`, Davidson vs dense.
13. Supermatrix variants with the EKT references (both manifolds): the first moment of the bare no-bath
    variant equals the exact-reference value in all blocks; the same-sector elements of `K` equal the
    Hamiltonian matrix elements between the embedded EKT states and `a_P^(+)|Psi_0>`; the dressed-bath
    identity holds; dressed bare, dressed screened and diagonal bare variants are PSD with `M0 = 1`
    and Davidson agrees with dense.
14. Spin-orbital RDMs and cumulants: `D_1 = gamma`, partial traces `D_k -> (N-k+1) D_{k-1}`, the trace
    relation of the 2-cumulant, all cumulants vanish for a single determinant, all cross-subsystem cumulants
    (2-, 3- and 4-body) vanish for a product state of two independent subsystems, and the cumulant
    reconstruction is consistent.  The RDM realization of the EKT (primary and extended) and of the ERPA
    reproduces the CI-vector realization in the matrices, eigenvalues and `G(z)` of the dressed screened
    variant; the extended EKT with the 4-cumulant dropped keeps `M0` exact.

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
`h4_sto-6g_cas2_2_spectral_functions.png`, `h4_6-31g_cas2_2_spectral_functions.png` and
`h4_6-31g_cas4_4_spectral_functions.png` are the plots produced by `example_h4.py` (one panel per
variant, including the EKT/ERPA ones; the 6-31G CAS(4,4) run takes about three minutes because
all eight matrices, up to 9400 rows, are also diagonalized densely).

## Ozone check against Wang, Fang, and Li (`check_ozone.py`)

`check_ozone.py` rebuilds the O3 setting of the paper (CCCBDB geometry R = 1.278 A,
theta = 116.8 deg, 6-31G, CASCI with RHF orbitals) and compares with their Tables S3-S5:

```
~/miniforge3/bin/python check_ozone.py --cas 6,4          # 5 s, bath of 4.4e4 poles
~/miniforge3/bin/python check_ozone.py --cas 8,5          # 16 s, bath of 1.0e5 poles
~/miniforge3/bin/python check_ozone.py --cas 6,4 --plot   # + spectral functions (exact-reference variants,
                                                          #   experimental and DMRG IPs as reference lines), ~30 s
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

## EKT/ERPA references (`mrgw_hermitian/ekt_erpa.py`)

`EKTERPAReference` is a drop-in replacement for `DyallReference` in which

* the active N±1 states are extended-Koopmans (EKT) states, `|alpha^-> = sum_q c_q a_q |Xi_0>`
  and `|alpha^+> = sum_q c_q a_q^+ |Xi_0>`, from the generalized eigenproblems
  `A^- c = omega gamma c` and `A^+ c = omega (1 - gamma^T) c` with
  `A^-_pq = <Xi_0|a_p^+ (H_act - E_0) a_q|Xi_0>`, `A^+_pq = <Xi_0|a_p (H_act - E_0) a_q^+|Xi_0>`
  (functions of the 1- and 2-RDMs; evaluated here with CI vectors).  The EKT poles exhaust the
  zeroth and first spectral moments of the active Green's function exactly;
* the active neutral excitations that enter the reference polarizability are extended-RPA (ERPA)
  solutions of the equation of motion `<0|[E_qp,[H,O^+]]|0> = omega <0|[E_qp,O^+]|0>` in the
  single-excitation manifold `E_pq = a_p^+ a_q` on the correlated CAS reference, solved in the
  natural-spin-orbital basis where the metric is `n_q - n_p`.  With the Dyall Hamiltonian the
  core->active and active->virtual channels of the reference response reduce exactly to the EKT
  problems shifted by the orbital energies, so only the active-active channel uses the ERPA
  equations.  The ERPA response obeys the zeroth and energy-weighted sum rules of the exact
  active-space response within the singles manifold (checked to 1e-16).  With
  `response='exact'` the exact CAS excitations are kept instead, which isolates the effect of
  the EKT charged manifold.

Extended charged manifold (`EKTERPAReference(ref, charged_manifold='extended')`).  The
primary operators are supplemented by the 2h1p operators `a_y^+ a_w a_z` (removal) and the
2p1h operators `a_y a_w^+ a_z^+` (attachment) of the active space, and the generalized
eigenproblem is solved after canonical orthogonalization of the strongly linearly dependent
Gram matrix.  Because `(H - E_0) a_z |Xi_0> = [H, a_z] |Xi_0>` lies in this span, the extended
manifold conserves the spectral moments M0-M3 of the active Green's function exactly (checked),
and it reproduces the exact poles whenever it spans the N±1 sector; its rank is bounded by
about `n_act^3` instead of the sector dimension, so it becomes an approximation only for active
spaces of roughly eight or more orbitals.

MR-RPA screening of the residual interaction, the MR-GW bath, the mixed couplings and the
dressed bath are then built exactly as before (`MRRPA(ekt)`, `HermitianGF(ekt, ...)`).  Two
identities hold for both manifolds (checked): the first moment of the bare no-bath variant is
identical to the exact-reference value in all blocks, because the sum over EKT states is the
projector on a manifold that contains `a_x^(+)|Xi_0>`; and the same-sector elements of `K` are
exact Hamiltonian matrix elements between the embedded EKT states.  The working equations are
collected in Sec. IV of `notes/mr-hedin.pdf`.

H4 in 6-31G with CAS(4,4) (`example_h4.py --basis 6-31g --ncas 4 --nelecas 4`) is the smallest
setting in which the primary manifold is incomplete (rank 4 of 24 per sector) while the extended
one is complete (24 of 24):

| H4, 6-31G, CAS(4,4), eV | principal IP | principal EA | virtual EA |
|---|---|---|---|
| MR-GW | 12.058 | 3.416 | 26.432 |
| EKT/ERPA MR-GW (primary) | 12.587 | 3.756 | 26.632 |
| EKT(2h1p)/ERPA MR-GW | 12.057 | 3.417 | 26.489 |
| full CI | 11.999 | 2.993 | 22.500 |

The primary manifold also produces a spurious intense removal peak at 18.2 eV (the exact
reference has a 0.76-weight peak at 16.7 eV); the extended manifold removes both defects, and the
residual differences to MR-GW (1 meV for the principal peaks) come from the singles ERPA response.

For H4/CAS(2,2) the primary manifold spans the whole N±1 space, so the EKT poles are
exact; the ERPA reproduces the two single excitations exactly and misses the double one, and the
EKT/ERPA MR-GW roots agree with MR-GW to 1 meV.  For ozone the ERPA excitation energies of the
active space are essentially exact (the lowest four agree with the CASCI values to 1 meV), but
the primary-manifold EKT cation energies are far too high, because the CASCI cations are strongly
relaxed multiconfigurational states that the manifold `{a_x |Xi_0>}` cannot represent.  The
extended manifold is complete for both active spaces (rank 24 of 24 and 50 of 50 in the removal
sectors, from 80 and 168 manifold vectors) and recovers the CASCI cation energies to 1e-12 Ha:

| O3, 6-31G, IPs in eV | 2A1 | 2B2 | 2A2 |
|---|---|---|---|
| CASCI(6,4) | 14.38 | 14.67 | 15.12 |
| EKT(6,4), primary | 15.29 | 16.02 | 15.21 |
| EKT(6,4), 1h+2h1p | 14.38 | 14.67 | 15.12 |
| MR-GW(6,4) | 12.65 | 12.94 | 14.56 |
| EKT/ERPA MR-GW(6,4) | 13.37 | 13.94 | 14.64 |
| EKT(2h1p)/ERPA MR-GW(6,4) | 12.66 | 12.95 | 14.57 |
| MR-GW(6,4) + mixed (screened) | 12.68 | 13.21 | 14.99 |
| EKT(2h1p)/ERPA MR-GW(6,4) + mixed (screened) | 12.68 | 13.22 | 15.00 |
| CASCI(8,5) | 13.59 | 13.93 | 13.10 |
| EKT(8,5), primary | 15.28 | 16.30 | 15.27 |
| EKT(8,5), 1h+2h1p | 13.59 | 13.93 | 13.10 |
| MR-GW(8,5) | 11.58 | 12.09 | 12.94 |
| EKT/ERPA MR-GW(8,5) | 12.88 | 13.77 | 15.01 |
| EKT(2h1p)/ERPA MR-GW(8,5) | 11.68 | 12.13 | 12.97 |
| MR-GW(8,5) + mixed (screened) | 12.01 | 12.27 | 13.32 |
| EKT(2h1p)/ERPA MR-GW(8,5) + mixed (screened) | 12.08 | 12.31 | 13.36 |
| DMRG (paper) | 12.34 | 12.59 | 13.34 |

With the primary manifold the screening lowers the EKT energies by about 2 eV and restores the
ordering 2A1 < 2B2 < 2A2, but the ionization energies remain 0.7-2 eV above MR-GW and DMRG.  With
the extended manifold the charged poles are exact, and the remaining difference to MR-GW isolates
the effect of replacing the exact active response by the singles ERPA: at most 0.01 eV for
CAS(6,4) and 0.10 eV for CAS(8,5).  The weak point of the primary-manifold variant is therefore the
EKT description of the charged active states, which the 2h1p/2p1h extension removes, while the
ERPA is an adequate replacement for the neutral response in these active spaces.

## Supermatrix variants with the three references (`ginit_prelim.py`)

`ginit_prelim.py` builds, for each reference (exact CAS states; EKT primary + ERPA; EKT extended
+ ERPA), the ten matrices of Tables I and II of `notes/mr-hedin.pdf` and follows the HOMO-like
removal and LUMO-like attachment roots (H4) or the three lowest cation states of each irrep
(ozone).  Principal IP/EA in eV for H4, 6-31G, CAS(4,4) (full CI 11.999/2.993):

| variant | exact CAS | EKT, primary | EKT, 1h+2h1p |
|---|---|---|---|
| Eq. (7), bare K_B, no bath | 12.176/3.638 | 12.714/3.908 | 12.176/3.638 |
| MR-GW (K_B = 0, diag. bath) | 12.058/3.416 | 12.587/3.756 | 12.057/3.417 |
| Eq. (7) + diag. bath, bare | 12.052/3.267 | 12.581/3.539 | 12.052/3.268 |
| Eq. (7) + diag. bath, screened | 12.082/3.778 | 12.620/4.071 | 12.081/3.779 |
| Eq. (7) + dressed bath, bare | 12.043/3.295 | 12.571/3.567 | 12.042/3.296 |
| Eq. (7) + dressed bath, screened | 12.065/3.807 | 12.602/4.100 | 12.064/3.808 |
| dressed bath, bare, no cross-sector | 12.043/3.294 | 12.571/3.567 | 12.042/3.295 |
| dressed bath, screened, no cross-sector | 12.043/3.805 | 12.572/4.099 | 12.043/3.807 |

The top block has 104 rows for the exact and the extended-manifold reference (96 active poles)
but 24 rows for the primary manifold (16 poles), and the bath shrinks accordingly (552 instead of
about 11000 rows for the dressed bath).  The extended manifold reproduces the exact-reference
numbers to 1-2 meV in every variant (the residual is the ERPA response), whereas the primary
manifold is 0.5 eV too high for the IP and 0.3-0.4 eV too high for the EA in every variant, i.e.
the deficiency of the primary EKT states is not repaired by the screening, the mixed couplings or
the dressed bath.

For ozone (`ginit_prelim.py --o3`, Table II of the notes, `results/ginit_prelim_o3.txt`) the same
holds: with the extended manifold every variant reproduces the exact-reference IPs to 0.01 eV for
CAS(6,4) and to 0.03-0.10 eV for CAS(8,5) (the residual is the ERPA response), e.g. dressed bath
with the screened kernel 12.63/13.15/14.94 vs 12.63/13.14/14.93 eV for CAS(6,4), whereas the primary
manifold stays 0.7-2 eV too high in every variant (13.09/14.15/15.02 eV).  The variant-to-variant
pattern (screened kernel +0.3-0.4 eV for 2B2/2A2, dressed vs diagonal bath within 0.05 eV) is the
same for all three references.  The primary-manifold matrices are much smaller: top block 62/64
instead of 102/154 rows, dressed bath 1.5e4 instead of 5.2e4/1.5e5 rows for CAS(6,4)/(8,5).

## Density-matrix realization and cumulant truncation (`wick.py`, `rdm.py`, `cumulant_comparison.py`)

`EKTERPAReference(ref, charged_manifold=..., realization='rdm', cumulant_drop=())` evaluates every EKT
and ERPA matrix element as a contraction of the active integrals with the spin-orbital reduced density
matrices `D_k[p1..pk, q1..qk] = <p1^+ .. pk^+ qk .. q1>` of the CAS wave function, `k <= 4`.  The
reduction is done symbolically by a small normal-ordering engine (`wick.py`): the commutators of the
active Hamiltonian with the manifold operators (`a_y`, `a_y^+ a_w a_z`, `a_y^+`, `a_y a_w^+ a_z^+`,
`a_p^+ a_q`) are expanded, the bra operator is anticommuted into normal order, and each normal-ordered
string of `k` creators and `k` annihilators becomes `D_k`.  The resulting RDM orders are

| block | metric | Hamiltonian |
|---|---|---|
| 1h–1h (primary EKT) | 1-RDM | 2-RDM (3 terms) |
| 1h–2h1p | 2-RDM | 3-RDM (11 terms) |
| 2h1p–2h1p | 3-RDM | 4-RDM (28 terms) |
| ERPA (double commutator) | 1-RDM | 2-RDM (20 terms) |

and the mixed-coupling tensors need the 3-RDM for the 2h1p/2p1h rows.  The complete term lists are in
`rdm_terms.tex` (generated by `wick.to_latex`, typeset in `equations.pdf`).  With exact RDMs the RDM and
CI-vector realizations agree to 1e-12 in all matrices and in the Green's function of every variant.

`rdm.py` computes the RDMs as Gram matrices of the vectors `a_qk .. a_q1 |Xi_0>` (sorted tuples, filled
by antisymmetry; the spin-orbital 4-RDM of the (8,5) space has 1e8 elements), and decomposes them with
the Grassmann (shuffle) product into cumulants,

```
D_2 = g^g/2! + L_2
D_3 = g^g^g/3! + g^L_2 + L_3
D_4 = g^g^g^g/4! + (g^g)^L_2/2! + L_2^L_2/2! + g^L_3 + L_4
```

(each distinct index assignment once; verified by the vanishing of all cumulants for a determinant and of
all cross-subsystem cumulants for a product state).  `cumulant_drop=(4,)` rebuilds `D_4` with `L_4 = 0`
(only the 2h1p–2h1p Hamiltonian block changes), `cumulant_drop=(3, 4)` also rebuilds `D_3` with `L_3 = 0`
(the 2h1p metric, the 1h–2h1p block and the mixed-coupling tensors change as well).

Results for H4, 6-31G, CAS(4,4) (`cumulant_comparison.py`; principal IP/EA in eV, extended manifold,
ERPA response; full CI 11.999/2.993):

| variant | exact RDMs (= CI) | L4 = 0 | L3 = L4 = 0, tau = 5e-2 |
|---|---|---|---|
| Eq. (7), bare K_B, no bath | 12.176/3.638 | 12.175/3.642 | 12.663/4.037 |
| MR-GW (K_B = 0, diag. bath) | 12.057/3.417 | 12.056/3.421 | 12.547/3.847 |
| Eq. (7) + diag. bath, screened | 12.081/3.779 | 12.080/3.783 | 12.571/4.191 |
| Eq. (7) + dressed bath, bare | 12.042/3.296 | 12.041/3.299 | 12.532/3.704 |
| Eq. (7) + dressed bath, screened | 12.064/3.808 | 12.062/3.812 | 12.554/4.220 |

The relative cumulant norms are `|L_k|/|D_k| = 0.14, 0.04, 0.015` for `k = 2, 3, 4`.  Dropping `L_4`
keeps the number of poles (96), the metric and the low-lying poles (the two lowest IPs/EAs move by 2-8 meV,
every variant by at most 4 meV), while the highest poles of the manifold move by up to 30 eV and
`M_2`, `M_3` of the active propagator deviate by 5e-4 Ha.  Dropping `L_3` as well is qualitatively
different: the reconstructed 3-RDM is not N-representable, the Gram matrix of the 2h1p/2p1h sector has
134 negative eigenvalues, the manifold acquires 226 instead of 96 poles, 70 of them on the wrong side of
the Fermi level, and the MR-RPA built from them is unstable (164 complex or negative-norm modes), so the
supermatrix variants cannot be formed (the script skips them).  Keeping only the metric directions above
`5e-2` of the largest eigenvalue (`lin_tol=5e-2`) gives a stable manifold of 48 poles, but all IPs/EAs are
0.4-0.5 eV too high, worse than the primary manifold with the exact 2-RDM.  The 3-body cumulant is thus
required for the 2h1p/2p1h sector, the 4-body cumulant can be dropped at the meV level for the low-lying
poles.

Ozone (`cumulant_comparison.py --o3`, Table III of the notes; IPs 2A1/2B2/2A2 in eV, extended manifold,
ERPA response):

| variant | CAS | exact RDMs (= CI) | L4 = 0 | L3 = L4 = 0, tau = 5e-2 |
|---|---|---|---|---|
| MR-GW (K_B = 0, diag. bath) | (6,4) | 12.66/12.95/14.57 | 12.66/12.95/14.57 | 12.75/13.07/-- |
| Eq. (7) + dressed bath, screened | (6,4) | 12.63/13.15/14.94 | 12.63/13.15/14.94 | 12.69/13.27/-- |
| MR-GW (K_B = 0, diag. bath) | (8,5) | 11.68/12.13/12.97 | 11.68/12.13/12.88 | 12.30/--/13.70 |
| Eq. (7) + dressed bath, screened | (8,5) | 12.05/12.26/13.32 | 12.06/12.26/13.23 | 12.73/--/14.06 |

The cumulant norms are `|L_k|/|D_k| = 0.15, 0.04, 0.02` for (6,4) and `0.11, 0.025, 0.010` for (8,5).
With `L_4 = 0` the (6,4) results change by at most 2 meV; for (8,5) the 2A1 and 2B2 states change by
2 meV but the 2A2 state by 0.10 eV (the EKT pole itself moves from 13.10 to 13.01 eV).  The raw
`L_3 = L_4 = 0` reference is unstable (144/238 negative metric eigenvalues, 68/148 wrong-sign poles,
1021/2736 lost MR-RPA modes for (6,4)/(8,5)); the regularized one raises the cation energies by
0.1-0.7 eV and one of the three cation states loses its residue entirely (dash: 2A2 for (6,4), 2B2 for
(8,5)).  Note that two attachment poles of the exact (6,4) active Hamiltonian lie below the neutral
ground state (CASCI in RHF orbitals); the script reports such wrong-sign poles but only lost MR-RPA
modes disqualify a reference.  The (8,5) run takes about 10 minutes, dominated by the cumulant
reconstruction of the 1e8-element 4-RDM.
