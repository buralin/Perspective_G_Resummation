"""Ground-state Rayleigh-Schroedinger (MP) series of Ne, the perturbative input of every ISR-ADC(n).

The ADC(n) matrix elements are truncated power series built from the RS ground state. If that series has
a singularity inside |lambda| <= 1, the ISR matrix elements, and with them ADC(n) for n -> infinity, cannot
converge at lambda = 1. This script prints E^(k), the ratio |E^(k)/E^(k-1)| and the norm series for the
same Hamiltonian, orbitals and frozen core as the oracle records (H(lambda) = F + lambda (H - F), canonical
RHF, D2h, 1s frozen), using the oracle's own ground sector and rspt.

--partition en replaces the MP zeroth-order Hamiltonian (sum of orbital energies) by the Epstein-Nesbet one,
the diagonal of H in the determinant basis of each sector, H(lambda) = H_0^EN + lambda (H - H_0^EN), so that
every first-order energy vanishes (Olsen, Jorgensen, Helgaker, Christiansen, J. Chem. Phys. 112, 9736 (2000)).

Optionally (--levels L) also the RS series of the Koopmans cation states (a_{p beta}|HF> of the L highest distinct
occupied levels) in the same orbitals and partitioning, E_{N-1}^(k), so that E_{N-1}^(k) - E_N^(k) is the IP series.

usage: python ne_ground_rs_series.py <system> [--nord N] [--levels L] [--partition mp|en] [--out <json>]
       system: Ne_ccpvdz_fc1, Ne_augccpvdz_fc1, Ar_ccpvdz_fc5, Ar_augccpvdz_fc5, Be_ccpvdz, Be_augccpvdz
WICKS_ROOT (default ~/Code/wicks) locates validation/lambda_oracle/sector_oracle.py.
"""
import argparse
import json
import os
import sys
import time

WICKS = os.environ.get('WICKS_ROOT', os.path.expanduser('~/Code/wicks'))
sys.path.insert(0, os.path.join(WICKS, 'validation', 'lambda_oracle'))
sys.path.insert(0, WICKS)
import sector_oracle as SO  # noqa: E402

# systems beyond sector_oracle's own tables: (atoms, basis, frozen spatial core orbitals, point group)
SO.SYSTEMS_EXTRA.update({
    'Ar_ccpvdz_fc5': ('atoms_basis_fc', ('Ar 0 0 0', 'cc-pvdz', 5, 'D2h')),
    'Be_ccpvdz': ('atoms_basis_fc', ('Be 0 0 0', 'cc-pvdz', 0, 'D2h')),
    'Be_augccpvdz': ('atoms_basis_fc', ('Be 0 0 0', 'aug-cc-pvdz', 0, 'D2h')),
})


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('system')
    ap.add_argument('--nord', type=int, default=24, help='highest energy order')
    ap.add_argument('--levels', type=int, default=0,
                    help='also the RS series E_{N-1}^(k) of the Koopmans cation state of the highest LEVELS '
                         'distinct occupied levels (2p, 2s for Ne)')
    ap.add_argument('--partition', choices=('mp', 'en'), default='mp',
                    help='zeroth-order Hamiltonian: mp = sum of orbital energies, en = diagonal of H')
    ap.add_argument('--out', default=None)
    args = ap.parse_args()
    t0 = time.time()
    mol, mf, ncore = SO.scf_system(args.system)
    h1, eri, eps, orbsym = SO.active_hamiltonian(mol, mf, ncore)
    no = mol.nelectron // 2 - ncore
    ground = SO.CISector(h1, eri, eps, orbsym, no, no, 0)
    if args.partition == 'en':
        ground.F = ground.hdiag.copy()      # rspt takes its H_0 from F: Epstein-Nesbet
    _, E, nser = SO.rspt(ground, ground.address(range(no), range(no)), args.nord)
    cations = []
    for p, members in SO.distinct_levels(eps[:no])[:args.levels]:
        irrep = int(orbsym[p])
        ion = SO.CISector(h1, eri, eps, orbsym, no, no - 1, irrep)
        if args.partition == 'en':
            ion.F = ion.hdiag.copy()
        koop = ion.address(range(no), [q for q in range(no) if q != p])
        _, Eion, _ = SO.rspt(ion, koop, args.nord)
        cations.append({'level': int(p), 'irrep': irrep, 'dim': int(ion.dim),
                        'E': {int(k): float(v) for k, v in Eion.items()}})
        r = [abs(Eion[k] / Eion[k - 1]) for k in range(3, args.nord + 1)]
        print(f'  cation level {p} (irrep {irrep}, dim {ion.dim}): |E(k)/E(k-1)| k=3..{args.nord}: '
              + ' '.join(f'{x:.3f}' for x in r), flush=True)
    ks = sorted(E)
    print(f'{args.system}: ground dim {ground.dim}, {time.time() - t0:.0f} s')
    print(' k        E^(k) (Ha)       |E^(k)/E^(k-1)|')
    for k in ks:
        r = abs(E[k] / E[k - 1]) if k >= 3 and E[k - 1] != 0 else float('nan')
        print(f'{k:2d} {E[k]: .12e} {r:8.4f}')
    out = args.out or os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ne_records',
                                   f'rs_ground_series_{args.system}_k{args.nord}.json')
    with open(out, 'w') as fh:
        json.dump({'system': args.system, 'partition': args.partition,
                   'what': f'ground-state RS ({args.partition.upper()}) energy coefficients E^(k), Ha; '
                   'norm series <Psi|Psi>_k', 'dim': int(ground.dim),
                   'E': {int(k): float(E[k]) for k in ks}, 'norm': [float(x) for x in nser],
                   'cations': cations,
                   'time_s': time.time() - t0}, fh, indent=1)
    print(f'-> {out}')


if __name__ == '__main__':
    main()
