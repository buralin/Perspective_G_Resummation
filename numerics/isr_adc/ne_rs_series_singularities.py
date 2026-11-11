"""Singularities of the Ne Rayleigh-Schroedinger series (MP or EN partitioning) from their coefficients.

For the ground-state energy E_N(lambda), and for each Koopmans cation state E_{N-1}(lambda) and the IP series
E_{N-1}(lambda) - E_N(lambda), this prints
- the ratios |c_k / c_{k-1}| at a few k,
- the root-test radius 1 / max_{k in [k1, kmax]} |c_k|^{1/k},
- the poles of near-diagonal Pade approximants [L/M] (L + M = kmax - j, j = 0..2) that are stable across them
  (within POLE_TOL) and not cancelled by a nearby zero (Froissart doublet): the nearest singularities,
- the nearest pole of the top diagonal approximant alone (a branch cut shows as poles that move between
  approximants, so it has no stable pole but a nearest one).
Coefficients come from ne_ground_rs_series.py records; the constant c_0 is dropped (it does not move poles).

usage: python ne_rs_series_singularities.py <record.json> [...]
"""
import json
import sys

import numpy as np

RATIO_KS = (10, 16, 20, 30)
ROOT_TEST_K1 = 20          # root test over k = ROOT_TEST_K1 .. kmax
MAX_RADIUS = 2.0           # poles beyond this |lambda| are not reported
POLE_TOL = 2e-2            # a pole is stable if every approximant has one within this distance
DOUBLET_TOL = 1e-10        # a pole with a numerator zero this close is a Froissart doublet (a weak
                           # physical singularity has a small residue, hence a close zero: keep this tight)


def pade(c, L, M):
    """Numerator and denominator coefficients (ascending) of the [L/M] Pade approximant of sum c_k x^k."""
    A = np.array([[c[L + i - j] if L + i - j >= 0 else 0.0 for j in range(1, M + 1)] for i in range(1, M + 1)])
    b = -np.array([c[L + i] for i in range(1, M + 1)])
    q = np.concatenate([[1.0], np.linalg.lstsq(A, b, rcond=None)[0]])
    p = np.array([sum(q[j] * c[i - j] for j in range(min(i, M) + 1)) for i in range(L + 1)])
    return p, q


def poles(c, L, M):
    """Poles of [L/M] inside MAX_RADIUS that are not cancelled by a zero."""
    p, q = pade(c, L, M)
    zq = np.roots(q[::-1])
    zp = np.roots(p[::-1]) if L > 0 else np.array([])
    keep = [z for z in zq if abs(z) < MAX_RADIUS and (len(zp) == 0 or np.min(np.abs(zp - z)) > DOUBLET_TOL)]
    return np.array(keep)


def stable_poles(c):
    """Poles present (within POLE_TOL) in every near-diagonal approximant using the top three orders."""
    kmax = len(c) - 1
    sets = []
    for j in range(3):
        n = kmax - j
        for L in (n // 2, n - n // 2):
            sets.append(poles(c, L, n - L))
    out = []
    for z in sorted(sets[0], key=abs):
        if all(len(s) and np.min(np.abs(s - z)) < POLE_TOL for s in sets[1:]):
            out.append(z)
    return out


def describe(name, coeff):
    """One line: ratios, root-test radius, stable poles."""
    ks = sorted(coeff)
    c = np.array([0.0] + [coeff[k] for k in ks if k >= 1])        # drop c_0
    kmax = len(c) - 1
    ratios = ' '.join(f'k{k}:{abs(c[k] / c[k - 1]):.3f}' for k in RATIO_KS if k <= kmax and c[k - 1] != 0)
    root = 1.0 / max(abs(c[k]) ** (1.0 / k) for k in range(ROOT_TEST_K1, kmax + 1))
    ps = stable_poles(c)
    top = poles(c, kmax // 2, kmax - kmax // 2)
    near = min(top, key=abs) if len(top) else None
    fmt = lambda z: f'{z.real:+.3f}{z.imag:+.3f}i' if abs(z.imag) > 1e-3 else f'{z.real:+.3f}'  # noqa: E731
    pz = ', '.join(fmt(z) for z in ps[:4])
    print(f'  {name:12s} |c_k/c_k-1| {ratios}; root-test radius {root:.3f}; stable Pade poles {pz or "none"}; '
          f'nearest [{kmax // 2}/{kmax - kmax // 2}] pole {fmt(near) if near is not None else "none"}')
    return dict(ratios={k: abs(c[k] / c[k - 1]) for k in RATIO_KS if k <= kmax and c[k - 1] != 0},
                root_test_radius=root, stable_poles=[[z.real, z.imag] for z in ps],
                nearest_top_pole=None if near is None else [near.real, near.imag])


def main():
    for path in sys.argv[1:]:
        with open(path) as fh:
            rec = json.load(fh)
        print(f'{rec["system"]} ({rec.get("partition", "mp").upper()}): {path}')
        E = {int(k): v for k, v in rec['E'].items()}
        describe('ground', E)
        for cat in rec.get('cations', []):
            label = {3: '2p', 0: '2s'}.get(cat['level'], f'level {cat["level"]}')
            Ec = {int(k): v for k, v in cat['E'].items()}
            describe(f'{label} cation', Ec)
            describe(f'{label} IP', {k: Ec[k] - E[k] for k in Ec})


if __name__ == '__main__':
    main()
