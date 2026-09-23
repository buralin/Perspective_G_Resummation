#!/usr/bin/env python
"""
Supermatrix variants of Sec. II of notes/mr-hedin.pdf for three active-space
references.

Starting point is Eq. (7) of the preliminary section, the initial Green's
function

    G_init(w) = diag(1, M) [ w - F_ii   -H_ia ; -H_ia^+   w - E^{N+-1} ]^{-1} diag(1, M^+),

i.e. the resolvent of the Hermitian block matrix built from the inactive
orbital energies, the active N+-1 energies and the first-order
inactive--active coupling H_ia.  In the pole-space language of Sec. II this is
the top-space matrix

    K = K_D + T_D^+ Sigma_1^{11} T_D + K_B          (bare or screened kernel),

which is then combined with the MR-RPA bosons in the diagonal (MR-GW) form,
B_I -> diag(kappa_n + sigma_n Omega_I), or in the dressed form,
B_I = K + sigma Omega_I (Eq. (44) of the notes).  The rows of Tables I and II
of the notes are computed:

    reference poles (+ Sigma_1^11)                : K_B = 0, no bath
    Eq. (7): same-sector H_ia, no bath            : literal superoperator construction
    Eq. (7): all pairs (bare K_B), no bath        : + cross-sector entries (Objective II)
    MR-GW (K_B = 0, diag. bath)                   : Wang-Fang-Li method
    Eq. (7) + diag. bath, bare / screened         : MR-GW + mixed
    Eq. (7) + dressed bath, bare / screened       : Eq. (44)
    dressed bath, bare / screened, no cross-sector: Eq. (45)

for the active-space references

    cas   exact N_a +/- 1 eigenstates of the Dyall active Hamiltonian and
          exact N_a states in the active-active MR-RPA channel (Sec. II);
    ekt   extended-Koopmans (EKT) charged states in the primary manifold
          {a_x |Xi_0>}, {a_x^+ |Xi_0>} and extended-RPA (ERPA) active
          response (Sec. IV of the notes);
    ekt2  as ekt with the charged manifold extended by the 2h1p and 2p1h
          operators a_y^+ a_w a_z |Xi_0>, a_y a_w^+ a_z^+ |Xi_0>.

For comparison full CI (H4) or the DMRG/experimental values of Wang, Fang, Li
(O3) are listed.

Run:  python ginit_prelim.py [--reference all|cas|ekt|ekt2]    # H4: STO-6G CAS(2,2), 6-31G CAS(2,2), CAS(4,4)
      python ginit_prelim.py --o3 [--cas 6,4 --cas 8,5]        # ozone
      python ginit_prelim.py --response exact                  # EKT poles with the exact active response
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

EV = HARTREE2EV
WFL_DMRG = (12.34, 12.59, 13.34)
WFL_EXP = (12.73, 13.00, 13.54)
WFL_MRGW = {'(6,4)': (12.65, 12.94, 14.56), '(8,5)': (11.58, 12.09, 12.94)}
STATES = ('A1', 'B2', 'A2')
REFERENCE_LABELS = {'cas': 'exact CAS states (Sec. II)',
                    'ekt': 'EKT, primary manifold {a_x}, {a_x^+} + ERPA response (Sec. IV)',
                    'ekt2': 'EKT, extended manifold 1h+2h1p / 1p+2p1h + ERPA response (Sec. IV)'}


def roots(gf, guesses, dense_max=6000):
    """Davidson root following; for matrices small enough to diagonalize the
    roots are cross-checked against the dense max-overlap eigenvalues and
    the maximal deviation is returned as the third value (None otherwise)."""
    theta, X, info = davidson_root_following(gf.matvec, gf.diagonal(), guesses,
                                             tol=1e-8, verbose=False)
    if not info['converged'].all():
        print(f"  warning: Davidson not converged, residuals {info['residual_norms']}")
    w = np.sum((gf.T @ X) ** 2, axis=0)
    dev = None
    if gf.n <= dense_max:
        dev = np.abs(gf.dense_roots(guesses) - theta).max()
    return theta, w, dev


def build_references(ref, which, response='erpa', verbose=False):
    """(key, label, reference, MRRPA) for the requested references."""
    out = []
    for key in which:
        if key == 'cas':
            r = ref
        elif key == 'ekt':
            r = EKTERPAReference(ref, charged_manifold='primary', response=response, verbose=verbose)
        elif key == 'ekt2':
            r = EKTERPAReference(ref, charged_manifold='extended', response=response, verbose=verbose)
        else:
            raise ValueError(key)
        label = REFERENCE_LABELS[key]
        if key != 'cas' and response == 'exact':
            label = label.replace('ERPA response', 'exact active response')
        out.append((key, label, r, MRRPA(r, verbose=verbose)))
    return out


def variants(ref, rpa):
    top = dict(bath_hamiltonian='top')
    return [
        ('reference poles (+ Sigma_1)', HermitianGF(ref, None, mixed='none', bath=False)),
        ('Eq. (7): same-sector H_ia, no bath', HermitianGF(ref, None, mixed='bare', bath=False,
                                                            cross_sector=False)),
        ('Eq. (7): all pairs (bare K_B), no bath', HermitianGF(ref, None, mixed='bare', bath=False)),
        ('MR-GW (K_B = 0, diag. bath)', HermitianGF(ref, rpa, mixed='none', bath=True)),
        ('Eq. (7) + diag. bath, bare', HermitianGF(ref, rpa, mixed='bare', bath=True)),
        ('Eq. (7) + diag. bath, screened', HermitianGF(ref, rpa, mixed='screened', bath=True)),
        ('Eq. (7) + dressed bath, bare', HermitianGF(ref, rpa, mixed='bare', bath=True, **top)),
        ('Eq. (7) + dressed bath, screened', HermitianGF(ref, rpa, mixed='screened', bath=True, **top)),
        ('dressed bath, bare, no cross-sector [Eq. (45)]',
         HermitianGF(ref, rpa, mixed='bare', bath=True, cross_sector=False, **top)),
        ('dressed bath, screened, no cross-sector [Eq. (45)]',
         HermitianGF(ref, rpa, mixed='screened', bath=True, cross_sector=False, **top)),
    ]


def reference_summary(key, r):
    if key == 'cas':
        print(f"  reference: {r.npoles} charged poles ({np.sum(r.pole_sign > 0)} attachment, "
              f"{np.sum(r.pole_sign < 0)} removal), "
              f"{r.sectors[r.nelecas].nstates - 1} neutral excitations")
        return
    ranks = ", ".join(f"{sec} {'att' if sg > 0 else 'rem'}: rank {rk} of dim {dim} ({m} vectors)"
                      for (sg, sec), (rk, m, dim) in r.charged_rank.items())
    dev = r.charged_pole_deviation()
    print(f"  reference: {r.npoles} EKT poles [{ranks}]"
          + (f"; complete, max |kappa - kappa_exact| = {dev:.1e} Ha" if dev is not None else "; incomplete"))
    om, _ = r.neutral_transition_densities()
    om_exact, _ = r.exact_neutral_transition_densities()
    print(f"  active response: {om.size} excitations ({'ERPA' if r.response == 'erpa' else 'exact'}; "
          f"exact CAS: {om_exact.size}); lowest (eV): "
          + ", ".join(f"{w * EV:.3f}" for w in np.sort(om)[:3]) + " vs exact "
          + ", ".join(f"{w * EV:.3f}" for w in np.sort(om_exact)[:3]))


def run_h4(basis, ncas, nelecas, which, response):
    print("=" * 100)
    print(f"H4, R = 1.0 A, {basis}, CASSCF({nelecas},{ncas})")
    print("=" * 100)
    mol = build_h4(1.0, basis)
    mf, mc = run_casscf(mol, ncas, nelecas)
    ref = DyallReference(mc, verbose=False)
    fci = FCIReference(ref, verbose=False)
    nocc = mol.nelectron // 2
    homo, lumo = 2 * (nocc - 1), 2 * nocc
    labels = ['IP (HOMO-like)', 'EA (LUMO-like)']
    if ref.core_so:
        labels.append(f'IP ({ref.so_label(ref.core_so[0])})')
    if ref.virt_so:
        labels.append(f'EA ({ref.so_label(ref.virt_so[0])})')
    for key, rlabel, r, rpa in build_references(ref, which, response):
        print(f"\n--- {rlabel} ---")
        reference_summary(key, r)
        print(f"{'variant':52s}" + "".join(f"{l:>22s}" for l in labels))
        for name, gf in variants(r, rpa):
            t0 = time.time()
            g = [gf.orbital_guess(homo, 'remove'), gf.orbital_guess(lumo, 'attach')]
            if ref.core_so:
                g.append(gf.unit_vector(gf.index_inactive(ref.core_so[0])))
            if ref.virt_so:
                g.append(gf.unit_vector(gf.index_inactive(ref.virt_so[0])))
            theta, w, dev = roots(gf, np.array(g).T)
            M0, _ = gf.moments()
            line = (f"{name:52s}" + "".join(f"{abs(t) * EV:12.3f} ({ww:.3f})" for t, ww in zip(theta, w))
                    + f"   n = {gf.n:6d}  |M0 - 1| {np.abs(M0 - np.eye(ref.nso)).max():.0e}")
            if gf.n <= 6000:
                _, _, wall = gf.poles()
                line += f"  min weight {wall.min():+.0e}  |Davidson - dense| {dev * EV:.0e} eV"
            line += f"  {time.time() - t0:.1f} s"
            print(line)
    pp = fci.principal_poles()
    print(f"\n{'full CI (most intense poles)':52s}"
          f"{-pp['remove'][0][0] * EV:12.3f} ({pp['remove'][0][1]:.3f})"
          f"{pp['attach'][0][0] * EV:12.3f} ({pp['attach'][0][1]:.3f})")
    print("  (energies in eV as |root|; weight in parentheses; min weight over all roots "
          "shows positivity)")
    print()


def lowest_cation_states(reference, sector):
    states = {}
    for ia, p in enumerate(reference.poles):
        if p.sign > 0 or p.sector != sector:
            continue
        nm = reference.irrep_name(reference.pole_symmetry(ia))
        if nm not in states or -p.kappa < -reference.poles[states[nm]].kappa:
            states[nm] = ia
    return states


def run_o3(cas, which, response):
    nelecas, ncas = cas
    label = f"({nelecas},{ncas})"
    print("=" * 100)
    print(f"O3, 6-31G, CASCI{label} in RHF orbitals")
    print("=" * 100)
    mol = build_o3('6-31g')
    mf, mc = run_casci(mol, ncas, nelecas)
    ref = DyallReference(mc, verbose=False)
    na, nb = ref.nelecas
    sector = (na - 1, nb)
    for key, rlabel, r, rpa in build_references(ref, which, response):
        print(f"\n--- {rlabel} ---")
        reference_summary(key, r)
        states = lowest_cation_states(r, sector)
        order = [nm for nm in STATES if nm in states]
        print(f"{'variant':52s}" + "".join(f"{'2' + s:>14s}" for s in order) + "   ordering")
        for name, gf in variants(r, rpa):
            t0 = time.time()
            g = np.array([gf.unit_vector(gf.index_active_pole(states[nm])) for nm in order]).T
            theta, w, dev = roots(gf, g)
            ips = -theta * EV
            ordering = ' < '.join('2' + order[k] for k in np.argsort(ips))
            print(f"{name:52s}" + "".join(f"{ip:8.2f} ({ww:.2f})" for ip, ww in zip(ips, w))
                  + f"   {ordering}   n = {gf.n:7d}"
                  + (f"   |Davidson - dense| {dev * EV:.0e} eV" if dev is not None else "")
                  + f"  {time.time() - t0:.1f} s")
    print()
    for nm, vals in (('MR-GW (paper)', WFL_MRGW[label]), ('DMRG (paper)', WFL_DMRG),
                     ('Exp. (paper)', WFL_EXP)):
        ordering = ' < '.join('2' + STATES[k] for k in np.argsort(vals))
        print(f"{nm:52s}" + "".join(f"{v:8.2f}       " for v in vals) + f"   {ordering}")
    print()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--o3', action='store_true', help='ozone instead of H4')
    ap.add_argument('--cas', action='append', help='ozone active space "nelec,norb" (repeatable; '
                    'default 6,4 and 8,5)')
    ap.add_argument('--reference', default='all', choices=['all', 'cas', 'ekt', 'ekt2'])
    ap.add_argument('--response', default='erpa', choices=['erpa', 'exact'],
                    help='active response used with the EKT references')
    args = ap.parse_args()
    which = ['cas', 'ekt', 'ekt2'] if args.reference == 'all' else [args.reference]
    if args.o3:
        for cas in (args.cas or ['6,4', '8,5']):
            run_o3(tuple(int(x) for x in cas.split(',')), which, args.response)
    else:
        run_h4('sto-6g', 2, 2, which, args.response)
        run_h4('6-31g', 2, 2, which, args.response)
        run_h4('6-31g', 4, 4, which, args.response)


if __name__ == '__main__':
    main()
