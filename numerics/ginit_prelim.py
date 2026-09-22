#!/usr/bin/env python
"""
Results for Eq. (7) of notes/mr-hedin.pdf (preliminary section): the initial
Green's function

    G_init(w) = diag(1, M) [ w - F_ii   -H_ia ; -H_ia^+   w - E^{N+-1} ]^{-1} diag(1, M^+)

i.e. the resolvent of the Hermitian block matrix built from the inactive
orbital energies, the exact active N+-1 energies and the first-order
inactive--active coupling H_ia, without any MR-GW bath.  In the pole-space
language of Objective II this is the top-space matrix

    K_top = K_D + T_D^+ Sigma_1^{11} T_D + K_B^{(1)}          (bare kernel, no bath),

and two versions are computed:

  * 'same-sector'  : H_ia only between a virtual orbital and an N+1 state or a
                     core orbital and an N-1 state (the literal superoperator
                     construction of the preliminary section);
  * 'all pairs'    : additionally the cross-sector entries of K_B (Objective II).

For comparison the CAS reference, MR-GW (bath, K_B = 0), MR-GW + mixed (bare)
and full CI (H4) or the DMRG/experimental values of Wang, Fang, Li (O3) are
listed.

Run:  python ginit_prelim.py            # H4 (STO-6G CAS(2,2), 6-31G CAS(2,2) and CAS(4,4))
      python ginit_prelim.py --o3       # ozone CAS(6,4) and CAS(8,5)
"""
import os
import sys
import argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mrgw_hermitian import (build_h4, build_o3, run_casscf, run_casci, DyallReference,
                            MRRPA, HermitianGF, davidson_root_following, FCIReference,
                            HARTREE2EV)

EV = HARTREE2EV
WFL_DMRG = {'(6,4)': (12.34, 12.59, 13.34), '(8,5)': (12.34, 12.59, 13.34)}
WFL_EXP = (12.73, 13.00, 13.54)
WFL_MRGW = {'(6,4)': (12.65, 12.94, 14.56), '(8,5)': (11.58, 12.09, 12.94)}
STATES = ('A1', 'B2', 'A2')


def roots(gf, guesses):
    """Davidson root following; for matrices small enough to diagonalize the
    roots are cross-checked against the dense max-overlap eigenvalues and
    the maximal deviation is returned as the third value (None otherwise)."""
    theta, X, info = davidson_root_following(gf.matvec, gf.diagonal(), guesses,
                                             tol=1e-8, verbose=False)
    if not info['converged'].all():
        print(f"  warning: Davidson not converged, residuals {info['residual_norms']}")
    w = np.sum((gf.T @ X) ** 2, axis=0)
    dev = None
    if gf.n <= 6000:
        dev = np.abs(gf.dense_roots(guesses) - theta).max()
    return theta, w, dev


def variants(ref, rpa):
    return [
        ('CAS (Dyall reference)', HermitianGF(ref, None, mixed='none', bath=False)),
        ('Eq. (7): G_init, same-sector H_ia', HermitianGF(ref, None, mixed='bare', bath=False,
                                                          cross_sector=False)),
        ('Eq. (7): G_init, all pairs (bare K_B)', HermitianGF(ref, None, mixed='bare', bath=False)),
        ('MR-GW (bath, K_B = 0)', HermitianGF(ref, rpa, mixed='none', bath=True)),
        ('Eq. (7) + bath (diag.) = MR-GW + mixed (bare)', HermitianGF(ref, rpa, mixed='bare', bath=True)),
        ('Eq. (7) x bath (dressed), bare', HermitianGF(ref, rpa, mixed='bare', bath=True,
                                                       bath_hamiltonian='top')),
        ('Eq. (7) x bath (dressed), screened', HermitianGF(ref, rpa, mixed='screened', bath=True,
                                                           bath_hamiltonian='top')),
    ]


def run_h4(basis, ncas, nelecas):
    print("=" * 90)
    print(f"H4, R = 1.0 A, {basis}, CASSCF({nelecas},{ncas})")
    print("=" * 90)
    mol = build_h4(1.0, basis)
    mf, mc = run_casscf(mol, ncas, nelecas)
    ref = DyallReference(mc, verbose=False)
    rpa = MRRPA(ref, verbose=False)
    fci = FCIReference(ref, verbose=False)
    nocc = mol.nelectron // 2
    homo, lumo = 2 * (nocc - 1), 2 * nocc
    labels = ['IP (HOMO-like)', 'EA (LUMO-like)']
    if ref.core_so:
        labels.append(f'IP ({ref.so_label(ref.core_so[0])})')
    if ref.virt_so:
        labels.append(f'EA ({ref.so_label(ref.virt_so[0])})')
    print(f"{'variant':44s}" + "".join(f"{l:>22s}" for l in labels))
    for name, gf in variants(ref, rpa):
        g = [gf.orbital_guess(homo, 'remove'), gf.orbital_guess(lumo, 'attach')]
        if ref.core_so:
            g.append(gf.unit_vector(gf.index_inactive(ref.core_so[0])))
        if ref.virt_so:
            g.append(gf.unit_vector(gf.index_inactive(ref.virt_so[0])))
        theta, w, dev = roots(gf, np.array(g).T)
        E, Z, wall = gf.poles()
        print(f"{name:44s}" + "".join(f"{abs(t) * EV:12.3f} ({ww:.3f})" for t, ww in zip(theta, w))
              + f"   min weight {wall.min():+.1e}"
              + (f"   |Davidson - dense| {dev * EV:.1e} eV" if dev is not None else ""))
    pp = fci.principal_poles()
    print(f"{'full CI (most intense poles)':44s}"
          f"{-pp['remove'][0][0] * EV:12.3f} ({pp['remove'][0][1]:.3f})"
          f"{pp['attach'][0][0] * EV:12.3f} ({pp['attach'][0][1]:.3f})")
    print("  (energies in eV as |root|; weight in parentheses; min weight over all roots "
          "shows positivity)")
    print()


def run_o3(cas):
    nelecas, ncas = cas
    label = f"({nelecas},{ncas})"
    print("=" * 90)
    print(f"O3, 6-31G, CASCI{label} in RHF orbitals")
    print("=" * 90)
    mol = build_o3('6-31g')
    mf, mc = run_casci(mol, ncas, nelecas)
    ref = DyallReference(mc, verbose=False)
    rpa = MRRPA(ref, verbose=False)
    na, nb = ref.nelecas
    sector = (na - 1, nb)
    states = {}
    for ia, p in enumerate(ref.poles):
        if p.sign > 0 or p.sector != sector:
            continue
        nm = ref.irrep_name(ref.pole_symmetry(ia))
        if nm not in states or -p.kappa < -ref.poles[states[nm]].kappa:
            states[nm] = ia
    print(f"{'variant':44s}" + "".join(f"{'2' + s:>14s}" for s in STATES) + "   ordering")
    for name, gf in variants(ref, rpa):
        g = np.array([gf.unit_vector(gf.index_active_pole(states[nm])) for nm in STATES]).T
        theta, w, dev = roots(gf, g)
        ips = -theta * EV
        order = ' < '.join('2' + STATES[k] for k in np.argsort(ips))
        print(f"{name:44s}" + "".join(f"{ip:8.2f} ({ww:.2f})" for ip, ww in zip(ips, w)) + f"   {order}"
              + (f"   |Davidson - dense| {dev * EV:.1e} eV" if dev is not None else ""))
    for nm, vals in (('MR-GW (paper)', WFL_MRGW[label]), ('DMRG (paper)', WFL_DMRG[label]),
                     ('Exp. (paper)', WFL_EXP)):
        order = ' < '.join('2' + STATES[k] for k in np.argsort(vals))
        print(f"{nm:44s}" + "".join(f"{v:8.2f}       " for v in vals) + f"   {order}")
    print()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--o3', action='store_true', help='ozone instead of H4')
    args = ap.parse_args()
    if args.o3:
        run_o3((6, 4))
        run_o3((8, 5))
    else:
        run_h4('sto-6g', 2, 2)
        run_h4('6-31g', 2, 2)
        run_h4('6-31g', 4, 4)


if __name__ == '__main__':
    main()
