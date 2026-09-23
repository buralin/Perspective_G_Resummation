#!/usr/bin/env python
"""
Density-matrix realization of the EKT/ERPA reference and cumulant truncation.

The extended-Koopmans (EKT) matrices of the 1h+2h1p / 1p+2p1h manifold and
the ERPA matrices are evaluated in two ways (Sec. IV of notes/mr-hedin.pdf):

    'ci'   by applying the operators and the active Hamiltonian to CI vectors;
    'rdm'  as contractions of the active integrals with the spin-orbital
           reduced density matrices D_1..D_4 of |Xi_0> (Wick engine).

With exact RDMs both realizations agree to machine precision.  The RDMs are
then decomposed into cumulants,

    D_3 = g^g^g/3! + g^L_2 + L_3,
    D_4 = g^g^g^g/4! + (g^g)^L_2/2 + L_2^L_2/2 + g^L_3 + L_4,

and the EKT(2h1p)/ERPA variants of the supermatrix are recomputed with
L_4 = 0 and with L_3 = L_4 = 0.  The 3-RDM enters the 2h1p metric, the
1h-2h1p Hamiltonian block and the mixed-coupling tensors; the 4-RDM enters
only the 2h1p-2h1p Hamiltonian block (and likewise for 2p1h).

Run:  python cumulant_comparison.py                 # H4, 6-31G, CAS(4,4)
      python cumulant_comparison.py --o3 [--cas 6,4 | --cas 8,5]
"""
import os
import sys
import time
import argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mrgw_hermitian import (build_h4, build_o3, run_casscf, run_casci, DyallReference,
                            EKTERPAReference, MRRPA, HermitianGF, davidson_root_following,
                            FCIReference, HARTREE2EV)
from mrgw_hermitian import rdm as rdmmod

EV = HARTREE2EV
STATES = ('A1', 'B2', 'A2')
WFL_MRGW = {'(6,4)': (12.65, 12.94, 14.56), '(8,5)': (11.58, 12.09, 12.94)}
WFL_DMRG = (12.34, 12.59, 13.34)

REALIZATIONS = [
    ('CI vectors', dict(realization='ci')),
    ('RDMs, exact', dict(realization='rdm')),
    ('RDMs, L4 = 0', dict(realization='rdm', cumulant_drop=(4,))),
    ('RDMs, L3 = L4 = 0', dict(realization='rdm', cumulant_drop=(3, 4))),
    # the L3 = 0 metric is indefinite; keeping only the metric directions with
    # eigenvalues above 5e-2 of the largest one is the regularization used to
    # obtain a stable (but strongly truncated) manifold
    ('L3 = L4 = 0, tau = 5e-2', dict(realization='rdm', cumulant_drop=(3, 4), lin_tol=5e-2)),
]
VARIANTS = [
    ('Eq. (7), bare K_B, no bath', dict(mixed='bare', bath=False)),
    ('MR-GW (K_B = 0, diag. bath)', dict(mixed='none', bath=True)),
    ('Eq. (7) + diag. bath, screened', dict(mixed='screened', bath=True)),
    ('Eq. (7) + dressed bath, bare', dict(mixed='bare', bath=True, bath_hamiltonian='top')),
    ('Eq. (7) + dressed bath, screened', dict(mixed='screened', bath=True, bath_hamiltonian='top')),
]


def roots(gf, guesses):
    theta, X, info = davidson_root_following(gf.matvec, gf.diagonal(), guesses, tol=1e-8, verbose=False)
    if not info['converged'].all():
        print(f"  warning: Davidson not converged, residuals {info['residual_norms']}")
    return theta, np.sum((gf.T @ X) ** 2, axis=0)


def build_references(ref):
    import warnings
    refs = []
    for label, kw in REALIZATIONS:
        t0 = time.time()
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            r = EKTERPAReference(ref, charged_manifold='extended', verbose=False, **kw)
            rpa = MRRPA(r, verbose=False)
        refs.append((label, r, rpa))
        print(f"  built {label} in {time.time() - t0:.1f} s")
    return refs


def cumulant_report(ref):
    t0 = time.time()
    D = ref.rdms(4)
    norms = rdmmod.cumulant_norms(D)
    print(f"spin-orbital RDMs D_1..D_4 of |Xi_0> ({ref.nact_so} active spin orbitals, "
          f"{time.time() - t0:.1f} s); Frobenius norms:")
    for k, (nd, nl) in norms.items():
        print(f"  k = {k}: |D_k| = {nd:9.4f}   |L_k| = {nl:9.4f}   |L_k|/|D_k| = {nl / nd:.3f}")
    return norms


def stable(r, rpa):
    """A reference is usable for the supermatrix if its MR-RPA has no lost
    (complex / negative-norm) modes.  Poles on the "wrong" side of the Fermi
    level are counted as a diagnostic only: for a CASCI reference in RHF
    orbitals the active Dyall Hamiltonian can have N_a+1 states below the
    neutral ground state (two for O3 CAS(6,4)), which is not an instability."""
    wrong = int(np.sum(r.kappa[r.pole_sign < 0] > 0) + np.sum(r.kappa[r.pole_sign > 0] < 0))
    return rpa.nmodes_lost == 0, wrong


def ekt_report(refs, exact):
    """Charged poles, ranks, metric and moments of the extended manifold."""
    print(f"\n{'realization':26s} {'poles':>6s} {'dropped':>8s} {'neg.metric':>11s} {'wrong sign':>11s} "
          f"{'RPA lost':>9s} {'max|dkappa|':>12s} {'|dM0..M3|':>10s}   lowest IPs / EAs (eV), physical sign only")
    for label, r, rpa in refs:
        dev = np.abs(np.sort(r.kappa) - np.sort(exact.kappa)).max() if r.npoles == exact.npoles else np.nan
        mom = max(np.abs(Mk - Mkx).max() for Mk, Mkx in r.active_moments(order=3))
        neg = sum(getattr(r, 'metric_negative', {}).values()) if hasattr(r, 'metric_negative') else 0
        ok, wrong = stable(r, rpa)
        ips = -r.kappa[(r.pole_sign < 0) & (r.kappa < 0)]
        eas = r.kappa[(r.pole_sign > 0) & (r.kappa > 0)]
        ips, eas = np.sort(ips)[:3], np.sort(eas)[:3]
        print(f"{label:26s} {r.npoles:6d} {r.ekt_dropped:8d} {neg:11d} {wrong:11d} {rpa.nmodes_lost:9d} "
              f"{dev:12.2e} {mom:10.2e}   "
              + ", ".join(f"{x * EV:7.3f}" for x in ips) + " / " + ", ".join(f"{x * EV:7.3f}" for x in eas))
    print("  (dropped: manifold directions removed by canonical orthogonalization; neg.metric: negative "
          "eigenvalues of the Gram matrix;\n   wrong sign: removal poles above / attachment poles below the "
          "Fermi level; RPA lost: complex or negative-norm MR-RPA modes;\n   max|dkappa|: deviation of the "
          "sorted poles from the exact ones when the numbers agree; |dM0..M3|: active spectral moments)")


def variant_table(refs, guess_fn, labels, extra=None):
    print(f"\n{'variant':36s} {'realization':26s}" + "".join(f"{l:>20s}" for l in labels) + "        n")
    for vname, kw in VARIANTS:
        for label, r, rpa in refs:
            ok, wrong = stable(r, rpa)
            if not ok:
                print(f"{vname:36s} {label:26s}   skipped: {rpa.nmodes_lost} MR-RPA modes lost, "
                      f"{wrong} wrong-sign poles (unstable reference)")
                continue
            need = kw['bath'] or kw['mixed'] == 'screened'
            gf = HermitianGF(r, rpa if need else None, **kw)
            theta, w = roots(gf, guess_fn(gf, r))
            print(f"{vname:36s} {label:26s}" + "".join(f"{abs(t) * EV:12.3f} ({ww:.3f})" for t, ww in zip(theta, w))
                  + f"   {gf.n:6d}")
        if extra is not None:
            print(f"{'':36s} {'reference':26s}" + "".join(f"{v:12.3f}        " for v in extra))
        print()


def run_h4(basis='6-31g', ncas=4, nelecas=4):
    print("=" * 100)
    print(f"H4, R = 1.0 A, {basis}, CASSCF({nelecas},{ncas}); extended EKT manifold, ERPA response")
    print("=" * 100)
    mol = build_h4(1.0, basis)
    mf, mc = run_casscf(mol, ncas, nelecas)
    ref = DyallReference(mc, verbose=False)
    cumulant_report(ref)
    refs = build_references(ref)
    prim_ci = EKTERPAReference(ref, charged_manifold='primary', verbose=False)
    prim_rd = EKTERPAReference(ref, charged_manifold='primary', realization='rdm', verbose=False)
    print(f"  primary manifold, CI vs RDM realization: max|kappa - kappa'| = "
          f"{np.abs(np.sort(prim_ci.kappa) - np.sort(prim_rd.kappa)).max():.1e} Ha "
          f"(1- and 2-RDM only, no cumulant approximation involved)")
    print(f"  ERPA, CI vs RDM realization: max|omega - omega'| = "
          f"{np.abs(prim_ci.erpa_omega - refs[1][1].erpa_omega).max():.1e} Ha")
    ekt_report(refs, ref)
    fci = FCIReference(ref, verbose=False)
    pp = fci.principal_poles()
    nocc = mol.nelectron // 2
    homo, lumo = 2 * (nocc - 1), 2 * nocc
    variant_table(refs, lambda gf, r: np.array([gf.orbital_guess(homo, 'remove'), gf.orbital_guess(lumo, 'attach')]).T,
                  ['IP (HOMO-like)', 'EA (LUMO-like)'], extra=(-pp['remove'][0][0] * EV, pp['attach'][0][0] * EV))
    print("  (last line: full CI; energies in eV as |root|, weights in parentheses)")


def lowest_cation_states(reference, sector):
    states = {}
    for ia, p in enumerate(reference.poles):
        if p.sign > 0 or p.sector != sector:
            continue
        nm = reference.irrep_name(reference.pole_symmetry(ia))
        if nm not in states or -p.kappa < -reference.poles[states[nm]].kappa:
            states[nm] = ia
    return states


def run_o3(cas):
    nelecas, ncas = cas
    label = f"({nelecas},{ncas})"
    print("=" * 100)
    print(f"O3, 6-31G, CASCI{label} in RHF orbitals; extended EKT manifold, ERPA response")
    print("=" * 100)
    mol = build_o3('6-31g')
    mf, mc = run_casci(mol, ncas, nelecas)
    ref = DyallReference(mc, verbose=False)
    cumulant_report(ref)
    refs = build_references(ref)
    ekt_report(refs, ref)
    na, nb = ref.nelecas
    sector = (na - 1, nb)

    def guess(gf, r):
        st = lowest_cation_states(r, sector)
        return np.array([gf.unit_vector(gf.index_active_pole(st[nm])) for nm in STATES]).T
    print("\nCation states (lowest per irrep) of the references, IPs in eV:")
    for lab, r, _ in refs:
        st = lowest_cation_states(r, sector)
        print(f"  {lab:26s}" + "".join(f"  2{nm}: {-r.poles[st[nm]].kappa * EV:7.3f}" for nm in STATES if nm in st))
    variant_table(refs, guess, ['2' + s for s in STATES], extra=WFL_DMRG)
    print(f"  (last line: DMRG of Wang, Fang, Li; their MR-GW{label}: "
          + "/".join(f"{v:.2f}" for v in WFL_MRGW[label]) + " eV)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--o3', action='store_true')
    ap.add_argument('--cas', action='append')
    args = ap.parse_args()
    if args.o3:
        for cas in (args.cas or ['6,4', '8,5']):
            run_o3(tuple(int(x) for x in cas.split(',')))
    else:
        run_h4()


if __name__ == '__main__':
    main()
