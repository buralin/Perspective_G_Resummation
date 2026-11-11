"""Ground-state Rayleigh-Schroedinger (MP) series of Ne, the perturbative input of every ISR-ADC(n).

The ADC(n) matrix elements are truncated power series built from the RS ground state. If that series has
a singularity inside |lambda| <= 1, the ISR matrix elements, and with them ADC(n) for n -> infinity, cannot
converge at lambda = 1. This script prints E^(k), the ratio |E^(k)/E^(k-1)| and the norm series for the
same Hamiltonian, orbitals and frozen core as the oracle records (H(lambda) = F + lambda (H - F), canonical
RHF, D2h, 1s frozen), using the oracle's own ground sector and rspt.

usage: python ne_ground_rs_series.py <system> [--nord N] [--out <json>]
       system: Ne_ccpvdz_fc1 or Ne_augccpvdz_fc1
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


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('system')
    ap.add_argument('--nord', type=int, default=24, help='highest energy order')
    ap.add_argument('--out', default=None)
    args = ap.parse_args()
    t0 = time.time()
    mol, mf, ncore = SO.scf_system(args.system)
    h1, eri, eps, orbsym = SO.active_hamiltonian(mol, mf, ncore)
    no = mol.nelectron // 2 - ncore
    ground = SO.CISector(h1, eri, eps, orbsym, no, no, 0)
    _, E, nser = SO.rspt(ground, ground.address(range(no), range(no)), args.nord)
    ks = sorted(E)
    print(f'{args.system}: ground dim {ground.dim}, {time.time() - t0:.0f} s')
    print(' k        E^(k) (Ha)       |E^(k)/E^(k-1)|')
    for k in ks:
        r = abs(E[k] / E[k - 1]) if k >= 3 and E[k - 1] != 0 else float('nan')
        print(f'{k:2d} {E[k]: .12e} {r:8.4f}')
    out = args.out or os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ne_records',
                                   f'rs_ground_series_{args.system}_k{args.nord}.json')
    with open(out, 'w') as fh:
        json.dump({'system': args.system, 'what': 'ground-state RS (MP) energy coefficients E^(k), Ha; '
                   'norm series <Psi|Psi>_k', 'dim': int(ground.dim),
                   'E': {int(k): float(E[k]) for k in ks}, 'norm': [float(x) for x in nser],
                   'time_s': time.time() - t0}, fh, indent=1)
    print(f'-> {out}')


if __name__ == '__main__':
    main()
