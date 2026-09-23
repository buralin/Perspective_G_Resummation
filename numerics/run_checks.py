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
                            check_first_order_gf, check_superoperator_propagator,
                            HARTREE2EV)

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

    # 7b. superoperator form of the retarded propagator (Eq. (18) of mr-hedin.pdf, Sec. II)
    print("Superoperator propagator (a_p^+|(z - H_super)^-1|a_q^+) vs Lehmann representation:")
    report("superoperator propagator == retarded Lehmann Green's function",
           check_superoperator_propagator(ref), 1e-10)

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
    dense_sel = gf_mix.dense_roots(guesses)
    report("Davidson roots vs dense max-overlap eigenvalues", np.abs(theta - dense_sel).max(), 1e-7)
    report("Davidson residual norms", info['residual_norms'].max(), 1e-7)

    # 10b. bath with the top-space Hamiltonian inside the one-boson sector
    print("Dressed bath (top-space Hamiltonian in the one-boson sector):")
    zs = (-0.8 + 0.05j, -0.3 + 0.05j, 0.2 + 0.05j, 0.7 + 0.05j)
    g_diag = HermitianGF(ref, rpa, mixed='none', bath=True, include_static=False)
    g_top = HermitianGF(ref, rpa, mixed='none', bath=True, include_static=False,
                        bath_hamiltonian='top')
    report("dressed bath == diagonal bath when K_top is diagonal",
           max(np.abs(g_diag.greens_function(z) - g_top.greens_function(z)).max() for z in zs), 1e-10)
    g7 = HermitianGF(ref, None, mixed='bare', bath=False, cross_sector=False)
    g_top7 = HermitianGF(ref, rpa, mixed='bare', bath=True, cross_sector=False,
                         bath_hamiltonian='top')
    err = 0.0
    for z in zs:
        G_ref = np.linalg.inv(np.linalg.inv(g7.greens_function(z)) - g_top7.dressed_bath_self_energy(z))
        err = max(err, np.abs(g_top7.greens_function(z) - G_ref).max())
    report("dressed bath == GW self-energy built from the Eq. (7) propagator (no cross-sector)", err, 1e-9)
    g_top_full = HermitianGF(ref, rpa, mixed='bare', bath=True, bath_hamiltonian='top')
    _, _, w = g_top_full.poles()
    M0, _ = g_top_full.moments()
    report("dressed-bath variant: zeroth moment |M0 - 1|", np.abs(M0 - np.eye(ref.nso)).max(), 1e-10)
    report("dressed-bath variant: spectral weights >= 0 (-min w)", -w.min(), 1e-12)
    theta, X, info = davidson_root_following(g_top_full.matvec, g_top_full.diagonal(), guesses,
                                             tol=1e-8, verbose=False)
    report("dressed-bath variant: Davidson roots vs dense",
           np.abs(theta - g_top_full.dense_roots(guesses)).max(), 1e-7)

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
    dense_sel = gf_ekt.dense_roots(guesses)
    report("EKT/ERPA Davidson roots vs dense", np.abs(theta - dense_sel).max(), 1e-7)

    # 12. supermatrix variants with the EKT references (Sec. IV of the notes)
    print("Supermatrix variants with the EKT references:")
    g_ex = HermitianGF(ref, None, mixed='bare', bath=False)
    for label, r in (('primary', ekt), ('extended', ekt2)):
        g = HermitianGF(r, None, mixed='bare', bath=False)
        report(f"EKT ({label}): M1 of Eq. (7) (bare K_B, no bath) == exact-reference value",
               np.abs(g.moments()[1] - g_ex.moments()[1]).max(), 1e-9)
        dev = check_same_sector_couplings(r, verbose=False)
        report(f"EKT ({label}): same-sector K_top vs <embedded EKT states|H|...> (active-active)",
               dev['active-active'], 1e-9)
        report(f"EKT ({label}): same-sector K_top vs <embedded EKT states|H|...> (inactive-active)",
               dev['inactive-active (same sector, common sign per pole)'], 1e-9)
    for label, r, p in (('primary', ekt, rpa_ekt), ('extended', ekt2, MRRPA(ekt2, verbose=False))):
        g7 = HermitianGF(r, None, mixed='bare', bath=False, cross_sector=False)
        g_top7 = HermitianGF(r, p, mixed='bare', bath=True, cross_sector=False, bath_hamiltonian='top')
        err = 0.0
        for z in zs:
            G_ref = np.linalg.inv(np.linalg.inv(g7.greens_function(z)) - g_top7.dressed_bath_self_energy(z))
            err = max(err, np.abs(g_top7.greens_function(z) - G_ref).max())
        report(f"EKT ({label}): dressed bath == GW self-energy from the Eq. (7) propagator (no cross-sector)",
               err, 1e-9)
        for kw, name in ((dict(mixed='bare', bath=True, bath_hamiltonian='top'), 'dressed, bare'),
                         (dict(mixed='screened', bath=True, bath_hamiltonian='top'), 'dressed, screened'),
                         (dict(mixed='bare', bath=True), 'diag., bare')):
            g = HermitianGF(r, p, **kw)
            M0, _ = g.moments()
            _, _, w = g.poles()
            report(f"EKT ({label}), {name}: |M0 - 1| and -min weight",
                   max(np.abs(M0 - np.eye(ref.nso)).max(), -w.min()), 1e-10)
            gs = np.array([g.orbital_guess(homo, 'remove'), g.orbital_guess(lumo, 'attach')]).T
            theta, X, info = davidson_root_following(g.matvec, g.diagonal(), gs, tol=1e-8, verbose=False)
            report(f"EKT ({label}), {name}: Davidson roots vs dense", np.abs(theta - g.dense_roots(gs)).max(), 1e-7)

    # 13. density-matrix realization of the EKT/ERPA reference and cumulants
    print("Spin-orbital RDMs, cumulants and the RDM realization of EKT/ERPA:")
    from mrgw_hermitian import rdm as rdmmod
    from pyscf.fci import cistring
    D = ref.rdms(4)
    Nel = sum(ref.nelecas)
    report("D_1 == gamma_act", np.abs(D[1] - ref.gamma_act).max(), 1e-12)
    report("partial traces D_k -> (N-k+1) D_{k-1}", max(
        np.abs(np.einsum('prqr->pq', D[2]) - (Nel - 1) * D[1]).max(),
        np.abs(np.einsum('pqrstr->pqst', D[3]) - (Nel - 2) * D[2]).max(),
        np.abs(np.einsum('pqrtuvwt->pqruvw', D[4]) - (Nel - 3) * D[3]).max()), 1e-12)
    L = rdmmod.cumulants(D)
    report("cumulant trace relation sum_r L2[p,r,q,r] = gamma^2 - gamma",
           np.abs(np.einsum('prqr->pq', L[2]) - (D[1] @ D[1] - D[1])).max(), 1e-12)
    Ld = rdmmod.cumulants(rdmmod.spin_orbital_rdms(rdmmod.determinant_vector(ref.ncas, ref.nelecas),
                                                   ref.ncas, ref.nelecas, order=4))
    report("all cumulants vanish for a single determinant", max(np.abs(v).max() for v in Ld.values()), 1e-14)
    # product of two independent two-orbital subsystems: cross-subsystem cumulants vanish
    c, d = np.array([0.9, -np.sqrt(1 - 0.81)]), np.array([0.8, -0.6])
    vprod = np.zeros((cistring.num_strings(4, 2), cistring.num_strings(4, 2)))
    for i in (0, 1):
        for j in (2, 3):
            st = (1 << i) | (1 << j)
            vprod[cistring.str2addr(4, 2, st), cistring.str2addr(4, 2, st)] = c[i] * d[j - 2]
    Lp = rdmmod.cumulants(rdmmod.spin_orbital_rdms(vprod, 4, (2, 2), order=4))
    maskA = np.zeros(8, bool)
    maskA[:4] = True
    cross = 0.0
    for k, T in Lp.items():
        mA = np.zeros(T.shape, bool)
        mB = np.zeros(T.shape, bool)
        for ax in range(2 * k):
            sh = [1] * (2 * k)
            sh[ax] = 8
            mA |= maskA.reshape(sh)
            mB |= (~maskA).reshape(sh)
        cross = max(cross, float(np.abs(T[mA & mB]).max()))
    report("cross-subsystem cumulants L2, L3, L4 vanish for a product state", cross, 1e-14)
    R = rdmmod.reconstruct(D, drop=(4,))
    report("reconstruction: D_4(L4 = 0) + L_4 == D_4", np.abs(R[4] + L[4] - D[4]).max(), 1e-12)
    for label, man in (('primary', 'primary'), ('extended', 'extended')):
        ci_r = ekt if man == 'primary' else ekt2
        rd_r = EKTERPAReference(ref, charged_manifold=man, realization='rdm', verbose=False)
        err = 0.0
        for key in ci_r.ekt_problems:
            A1, S1, e1, _ = ci_r.ekt_problems[key]
            A2, S2, e2, _ = rd_r.ekt_problems[key]
            err = max(err, np.abs(A1 - A2).max(), np.abs(S1 - S2).max(), np.abs(e1 - e2).max())
        report(f"EKT ({label}): RDM realization == CI realization (A, S, eigenvalues)", err, 1e-9)
        report(f"EKT ({label}): RDM realization == CI realization (ERPA energies)",
               np.abs(ci_r.erpa_omega - rd_r.erpa_omega).max(), 1e-10)
        rpa_rd = MRRPA(rd_r, verbose=False)
        rpa_ci = MRRPA(ci_r, verbose=False)
        g1 = HermitianGF(ci_r, rpa_ci, mixed='screened', bath=True, bath_hamiltonian='top')
        g2 = HermitianGF(rd_r, rpa_rd, mixed='screened', bath=True, bath_hamiltonian='top')
        report(f"EKT ({label}): RDM realization == CI realization (G(z), dressed screened)",
               max(np.abs(g1.greens_function(z) - g2.greens_function(z)).max() for z in zs), 1e-9)
    rd_t = EKTERPAReference(ref, charged_manifold='extended', realization='rdm', cumulant_drop=(4,),
                            verbose=False)
    report("extended EKT with L4 = 0: metric unchanged, M0 of the active propagator exact",
           np.abs(rd_t.active_moments(order=0)[0][0] - np.eye(ref.nact_so)).max(), 1e-10)

    print()
    nfail = sum(1 for _, ok in status if not ok)
    print(f"{len(status) - nfail} checks passed, {nfail} failed")
    return nfail


if __name__ == '__main__':
    sys.exit(main())
