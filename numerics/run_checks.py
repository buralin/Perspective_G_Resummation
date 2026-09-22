#!/usr/bin/env python
"""
Consistency checks of the Hermitian MR-GW construction on linear H4
(STO-6G, CAS(2,2)), where everything can be compared with exact results.

Run:  python run_checks.py [--basis sto-6g] [--r 1.0]
"""
import os
import sys
import argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mrgw_hermitian import (build_h4, run_casscf, DyallReference, MRRPA,
                            EKTERPAReference, HermitianGF, davidson_root_following,
                            FCIReference, check_same_sector_couplings,
                            check_first_order_gf, HARTREE2EV)

EV = HARTREE2EV
status = []


def report(name, value, tol):
    ok = value < tol
    status.append((name, ok))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: {value:.3e} (tol {tol:.0e})")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--basis', default='sto-6g')
    ap.add_argument('--r', type=float, default=1.0)
    ap.add_argument('--ncas', type=int, default=2)
    ap.add_argument('--nelecas', type=int, default=2)
    args = ap.parse_args()

    mol = build_h4(args.r, args.basis)
    mf, mc = run_casscf(mol, args.ncas, args.nelecas)
    print(f"RHF energy    = {mf.e_tot:.10f}")
    print(f"CASSCF energy = {mc.e_tot:.10f}")
    ref = DyallReference(mc)

    # 1. active Dyall one-body operator vs PySCF's h1e_for_cas
    h1eff, ecore = mc.get_h1eff(ref.mo_coeff)
    report("h_eff vs mc.get_h1eff", np.abs(h1eff - ref.h_eff).max(), 1e-10)
    # 2. CAS ground state consistency
    report("E0(active) + E_core vs CASSCF energy", abs(ref.E0 + ecore - mc.e_tot), 1e-8)
    report("1 - |<Xi0|mc.ci>|", 1.0 - ref.overlap_with_mc_ci, 1e-8)
    # 3. sum rules
    report("active sum rule |d d^T - 1|", np.abs(ref.d_act @ ref.d_act.T - np.eye(ref.nact_so)).max(), 1e-10)
    T_D, K_D = ref.pole_representation()
    report("Dyall zeroth moment |T_D T_D^T - 1|", np.abs(T_D @ T_D.T - np.eye(ref.nso)).max(), 1e-10)
    # 4. structure of Sigma_1^{11} (Wang, Fang, Li, Eqs. S8-S13)
    S = ref.Sigma_stat
    ix = np.ix_
    report("Sigma_stat active-active block", np.abs(S[ix(ref.act_so, ref.act_so)]).max(), 1e-10)
    report("Sigma_stat core-core block", np.abs(S[ix(ref.core_so, ref.core_so)]).max(), 1e-8)
    report("Sigma_stat virtual-virtual block", np.abs(S[ix(ref.virt_so, ref.virt_so)]).max(), 1e-8)
    report("Sigma_stat Hermiticity", np.abs(S - S.T).max(), 1e-10)
    print(f"       max |Sigma_stat| core-virtual block (0 for converged CASSCF orbitals): "
          f"{np.abs(S[ix(ref.core_so, ref.virt_so)]).max():.2e}")
    print(f"       max |Sigma_stat| inactive-active block: "
          f"{np.abs(S[ix(ref.inact_so, ref.act_so)]).max():.2e}")

    # 5. MR-RPA
    rpa = MRRPA(ref)
    report("MR-RPA A/B asymmetry", rpa.asymmetry, 1e-10)
    report("MR-RPA normalization |R^T S - 1|", rpa.norm_error, 1e-8)
    report("MR-RPA positivity (-min Omega)", -rpa.Omega.min(), 0.0 + 1e-12)
    report("W_D(0) symmetry |W0 - W0^(pair swap)|", np.abs(rpa.W0 - rpa.W0.transpose(2, 3, 0, 1)).max(), 1e-10)

    # 6. exact linearization of the MR-GW Dyson equation
    gf_mrgw = HermitianGF(ref, rpa, mixed='none', bath=True)
    err = 0.0
    for z in (-0.8 + 0.05j, -0.3 + 0.05j, 0.2 + 0.05j, 0.7 + 0.05j):
        G_lin = gf_mrgw.greens_function(z)
        G_D = ref.dyall_greens_function(z)
        G_dyson = np.linalg.inv(np.linalg.inv(G_D) - rpa.self_energy(z))
        err = max(err, np.abs(G_lin - G_dyson).max())
    report("Hermitian linearization == MR-GW Dyson equation", err, 1e-9)

    # 7. same-sector couplings vs exact Hamiltonian matrix elements
    dev = check_same_sector_couplings(ref)
    report("same-sector K_top vs <charged Dyall states|H|...> (active-active)", dev['active-active'], 1e-9)
    report("same-sector K_top vs <charged Dyall states|H|...> (inactive-inactive)", dev['inactive-inactive'], 1e-9)
    report("same-sector K_top vs <charged Dyall states|H|...> (inactive-active)",
           dev['inactive-active (same sector, common sign per pole)'], 1e-9)

    # 8. first-order Green's function including cross-sector couplings
    print("First-order Green's function check (finite differences on full CI):")
    res = check_first_order_gf(ref, lam=1e-3)
    report("G_D from FCI(H_D) vs pole representation", max(r[1] for r in res), 1e-8)
    report("dG/dlambda: finite difference vs Hermitian K^(1)", max(r[2] / r[3] for r in res), 1e-5)

    # 9. PSD / moments of the screened mixed extension
    gf_mix = HermitianGF(ref, rpa, mixed='screened', bath=True)
    M0, M1 = gf_mix.moments()
    report("mixed-MR-GW zeroth moment |M0 - 1|", np.abs(M0 - np.eye(ref.nso)).max(), 1e-10)
    E, Z, w = gf_mix.poles()
    report("mixed-MR-GW spectral weights >= 0 (-min w)", -w.min(), 1e-12)
    fci = FCIReference(ref)
    _, M1_fci = fci.moments()
    _, M1_D = FCIReference(ref, ref.h1_D, ref.eri_D, verbose=False).moments()
    M1_exact_dyall_density = ref.h_so + np.einsum('pqrs,rs->pq', ref.Vbar, ref.gamma_so)
    print("First spectral moment diagnostics (max abs deviation from full CI, Ha):")
    for name, g in (('CAS/Dyall', HermitianGF(ref, None, mixed='none', bath=False)),
                    ('MR-GW', gf_mrgw),
                    ('MR-GW + mixed (bare)', HermitianGF(ref, rpa, mixed='bare', bath=True)),
                    ('MR-GW + mixed (screened)', gf_mix)):
        print(f"  {name:26s}: {np.abs(g.moments()[1] - M1_fci).max():.3e}")
    print(f"  {'h + vbar*gamma_D':26s}: {np.abs(M1_exact_dyall_density - M1_fci).max():.3e}"
          "  (first-order-exact reference: M1 with the Dyall density)")
    report("M1(MR-GW + mixed, bare) == h + vbar gamma_D (first-order exactness)",
           np.abs(HermitianGF(ref, rpa, mixed='bare', bath=True).moments()[1]
                  - M1_exact_dyall_density).max(), 1e-9)

    # 10. Davidson with root following vs dense diagonalization
    print("Davidson root following on the screened mixed MR-GW matrix:")
    homo = ref.act_so[0]
    lumo = ref.act_so[-2]
    guesses = np.array([gf_mix.orbital_guess(homo, 'remove'),
                        gf_mix.orbital_guess(lumo, 'attach'),
                        gf_mix.unit_vector(gf_mix.index_inactive(ref.core_so[0])),
                        gf_mix.unit_vector(gf_mix.index_inactive(ref.virt_so[0]))]).T
    theta, X, info = davidson_root_following(gf_mix.matvec, gf_mix.diagonal(), guesses,
                                             tol=1e-8, unit=EV, unit_name='(eV)')
    Ed, U = gf_mix.eig()
    ov = np.abs(guesses.T @ U)
    dense_sel = Ed[np.argmax(ov, axis=1)]
    report("Davidson roots vs dense max-overlap eigenvalues", np.abs(theta - dense_sel).max(), 1e-7)
    report("Davidson residual norms", info['residual_norms'].max(), 1e-7)

    # 11. EKT / ERPA variant
    print("EKT/ERPA variant (extended Koopmans poles, extended-RPA active response):")
    ekt = EKTERPAReference(ref)
    moments = ekt.active_moments(order=1)
    report("EKT zeroth moment |sum d d^T - 1|", np.abs(moments[0][0] - np.eye(ref.nact_so)).max(), 1e-10)
    report("EKT first moment == exact active first moment", np.abs(moments[1][0] - moments[1][1]).max(), 1e-9)
    if ref.ncas == 2 and ref.nelecas == (1, 1):
        report("EKT poles == exact poles (complete primary manifold for CAS(2,2))",
               np.abs(np.sort(ekt.kappa) - np.sort(ref.kappa)).max(), 1e-9)
    ekt2 = EKTERPAReference(ref, charged_manifold='extended')
    for k, (Mk, Mkx) in enumerate(ekt2.active_moments(order=3)):
        report(f"extended (1h+2h1p) manifold: moment M{k} == exact", np.abs(Mk - Mkx).max(), 1e-8)
    dev = ekt2.charged_pole_deviation()
    report("extended manifold poles == exact poles (complete for this active space)",
           np.inf if dev is None else dev, 1e-8)
    e0, e1, scale = ekt.response_sum_rules()
    report("ERPA zeroth sum rule vs exact active response", e0, 1e-9)
    report("ERPA energy-weighted sum rule vs exact active response", e1, 1e-9)
    rpa_ekt = MRRPA(ekt)
    err = 0.0
    gf_ekt0 = HermitianGF(ekt, rpa_ekt, mixed='none', bath=True)
    for z in (-0.8 + 0.05j, -0.3 + 0.05j, 0.2 + 0.05j, 0.7 + 0.05j):
        G_dyson = np.linalg.inv(np.linalg.inv(ekt.dyall_greens_function(z)) - rpa_ekt.self_energy(z))
        err = max(err, np.abs(gf_ekt0.greens_function(z) - G_dyson).max())
    report("EKT/ERPA Hermitian linearization == Dyson equation", err, 1e-9)
    gf_ekt = HermitianGF(ekt, rpa_ekt, mixed='screened', bath=True)
    M0h, _ = gf_ekt.moments()
    report("EKT/ERPA mixed-MR-GW zeroth moment |M0 - 1|", np.abs(M0h - np.eye(ref.nso)).max(), 1e-10)
    _, _, w = gf_ekt.poles()
    report("EKT/ERPA mixed-MR-GW spectral weights >= 0 (-min w)", -w.min(), 1e-12)
    guesses = np.array([gf_ekt.orbital_guess(homo, 'remove'), gf_ekt.orbital_guess(lumo, 'attach')]).T
    theta, X, info = davidson_root_following(gf_ekt.matvec, gf_ekt.diagonal(), guesses,
                                             tol=1e-8, verbose=False)
    Ed, U = gf_ekt.eig()
    dense_sel = Ed[np.argmax(np.abs(guesses.T @ U), axis=1)]
    report("EKT/ERPA Davidson roots vs dense", np.abs(theta - dense_sel).max(), 1e-7)

    print()
    nfail = sum(1 for _, ok in status if not ok)
    print(f"{len(status) - nfail} checks passed, {nfail} failed")
    return nfail


if __name__ == '__main__':
    sys.exit(main())
