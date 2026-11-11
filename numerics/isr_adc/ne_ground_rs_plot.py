"""RS (MP) series of Ne (1s frozen) for the ground state (N) and the 2p, 2s Koopmans cation states (N-1).

(a) cc-pVDZ and (b) aug-cc-pVDZ: |E^(k)| against k for N, N-1 (2p) and N-1 (2s); filled markers E^(k) > 0,
open E^(k) < 0. (c) aug-cc-pVDZ: ratio |E^(k)/E^(k-1)| of the same three series (log axis), together with
the decay ratio of the non-Dyson ADC(n) error, |ADC(n) - FCI| / |ADC(n-1) - FCI|, for 2p and 2s against n.

Input (next to this script): ne_records/rs_series_N_Nm1_Ne_{ccpvdz,augccpvdz}_fc1_k30.json and
ne_records/sector_oracle_Ne_augccpvdz_fc1_n11.json.

usage: python ne_ground_rs_plot.py [--data <dir>] [--out <file>]
"""
import argparse
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
INK, MUTED, GRID = '#0b0b0b', '#52514e', '#d9d8d4'
SERIES = (('$N$', '#2a78d6'), ('$N-1$, 2p', '#eb6834'), ('$N-1$, 2s', '#1baa7d'))
KMIN = 2   # E^(0), E^(1) depend on how the frozen-core constant is split; only their sum is meaningful


def series(path):
    """[E_N, E_{N-1}(2p), E_{N-1}(2s)] as {k: E^(k)} dicts from one record."""
    with open(path) as fh:
        d = json.load(fh)
    out = [{int(k): v for k, v in d['E'].items()}]
    for c in sorted(d['cations'], key=lambda c: -c['level']):   # level 3 (2p) first, then level 0 (2s)
        out.append({int(k): v for k, v in c['E'].items()})
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--data', default=os.path.join(HERE, 'ne_records'))
    ap.add_argument('--out', default=os.path.join(HERE, 'ne_ground_rs_series.pdf'))
    args = ap.parse_args()
    plt.rcParams.update({'font.size': 8, 'axes.edgecolor': MUTED, 'axes.labelcolor': INK,
                         'xtick.color': MUTED, 'ytick.color': MUTED, 'text.color': INK})
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.6))
    for ax, (basis, system) in zip(axes[:2], (('cc-pVDZ', 'Ne_ccpvdz_fc1'), ('aug-cc-pVDZ', 'Ne_augccpvdz_fc1'))):
        for E, (label, color) in zip(series(os.path.join(args.data, f'rs_series_N_Nm1_{system}_k30.json')), SERIES):
            ks = np.array([k for k in sorted(E) if k >= KMIN])
            e = np.array([E[k] for k in ks])
            ax.plot(ks, np.abs(e), color=color, lw=1.0, label=label, zorder=1)
            ax.scatter(ks[e > 0], e[e > 0], color=color, s=9, zorder=2)
            ax.scatter(ks[e < 0], -e[e < 0], facecolor='white', edgecolor=color, s=9, lw=0.8, zorder=2)
            if system == 'Ne_augccpvdz_fc1':
                axes[2].plot(ks[1:], np.abs(e[1:] / e[:-1]), color=color, marker='o', ms=2.4, lw=1.0, label=label)
        ax.set_yscale('log')
        ax.set_xlabel('$k$')
        ax.text(0.98, 0.98, basis, transform=ax.transAxes, ha='right', va='top', fontsize=7.5)
    axes[0].set_ylabel(r'$|E^{(k)}|$ (Ha)')
    axes[0].legend(frameon=False, fontsize=6.5, loc='lower left')
    axes[1].text(0.06, 0.90, 'filled: $E^{(k)}>0$\nopen: $E^{(k)}<0$', transform=axes[1].transAxes, ha='left',
                 va='top', fontsize=6, color=MUTED)
    # ADC(n) error decay ratio, aug-cc-pVDZ
    with open(os.path.join(args.data, 'sector_oracle_Ne_augccpvdz_fc1_n11.json')) as fh:
        blocks = json.load(fh)['blocks']
    for b, (lab, mk) in zip(sorted(blocks, key=lambda b: -b['level']), (('2p', 's'), ('2s', '^'))):
        ns = sorted(int(n) for n in b['orders'])
        err = np.array([abs(b['orders'][str(n)]['ip_ev'] - b['fci_ip_ev']) for n in ns])
        axes[2].plot(ns[1:], err[1:] / err[:-1], color=MUTED, marker=mk, ms=3.0, mfc='white', lw=0.8, ls='--',
                     label=f'ADC($n$) error, {lab}')
    axes[2].axhline(1.0, color=MUTED, lw=0.8, zorder=0)
    axes[2].set_yscale('log')
    axes[2].set_ylim(0.2, 20)
    axes[2].set_xlim(2.5, 30.5)
    axes[2].set_xlabel('$k$ (MP) or $n$ (ADC)')
    axes[2].set_ylabel('ratio of successive terms')
    axes[2].text(0.98, 0.98, 'aug-cc-pVDZ', transform=axes[2].transAxes, ha='right', va='top', fontsize=7.5)
    axes[2].legend(frameon=False, fontsize=5.5, loc='lower right', ncol=2, columnspacing=0.8, handlelength=1.8)
    for ax, tag in zip(axes, ('(a)', '(b)', '(c)')):
        ax.grid(axis='y', color=GRID, lw=0.5)
        for side in ('top', 'right'):
            ax.spines[side].set_visible(False)
        ax.text(0.01, 1.01, tag, transform=ax.transAxes, fontsize=8, fontweight='bold', va='bottom')
    fig.tight_layout(pad=0.4, w_pad=1.0)
    fig.savefig(args.out)
    print(f'-> {args.out}')


if __name__ == '__main__':
    main()
