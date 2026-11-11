"""Ground-state MP series of Ne (1s frozen): (a) |E^(k)| against k, (b) ratio test |E^(k)/E^(k-1)|.

Filled markers: E^(k) > 0, open: E^(k) < 0. Panel (b): aug-cc-pVDZ only (the cc-pVDZ ratio is noisy), together with the decay ratio of the
non-Dyson ADC(n) error, |ADC(n) - FCI| / |ADC(n-1) - FCI|, for 2p and 2s against n.

Input: ne_records/rs_ground_series_Ne_{ccpvdz,augccpvdz}_fc1_k30.json and
ne_records/sector_oracle_Ne_augccpvdz_fc1_n11.json next to this script.

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
BASES = (('cc-pVDZ', 'Ne_ccpvdz_fc1', '#2a78d6'), ('aug-cc-pVDZ', 'Ne_augccpvdz_fc1', '#eb6834'))
KMIN = 2   # E^(0), E^(1) depend on how the frozen-core constant is split; only their sum is meaningful


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--data', default=os.path.join(HERE, 'ne_records'))
    ap.add_argument('--out', default=os.path.join(HERE, 'ne_ground_rs_series.pdf'))
    args = ap.parse_args()
    plt.rcParams.update({'font.size': 8, 'axes.edgecolor': MUTED, 'axes.labelcolor': INK,
                         'xtick.color': MUTED, 'ytick.color': MUTED, 'text.color': INK})
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.8, 2.8))
    for label, system, color in BASES:
        with open(os.path.join(args.data, f'rs_ground_series_{system}_k30.json')) as fh:
            E = {int(k): v for k, v in json.load(fh)['E'].items()}
        ks = np.array([k for k in sorted(E) if k >= KMIN])
        e = np.array([E[k] for k in ks])
        ax1.plot(ks, np.abs(e), color=color, lw=1.0, label=label, zorder=1)
        ax1.scatter(ks[e > 0], e[e > 0], color=color, s=12, zorder=2)
        ax1.scatter(ks[e < 0], -e[e < 0], facecolor='white', edgecolor=color, s=12, lw=0.9, zorder=2)
        if system == 'Ne_augccpvdz_fc1':   # the cc-pVDZ ratio is noisy (near-zero terms); (a) shows it converging
            ax2.plot(ks[1:], np.abs(e[1:] / e[:-1]), color=color, marker='o', ms=2.8, lw=1.0,
                     label=f'MP $E^{{(k)}}$, {label}')
    # ADC(n) error decay ratio, aug-cc-pVDZ
    with open(os.path.join(args.data, 'sector_oracle_Ne_augccpvdz_fc1_n11.json')) as fh:
        blocks = json.load(fh)['blocks']
    for b, (lab, mk) in zip(sorted(blocks, key=lambda b: -b['level']), (('2p', 's'), ('2s', '^'))):
        ns = sorted(int(n) for n in b['orders'])
        err = np.array([abs(b['orders'][str(n)]['ip_ev'] - b['fci_ip_ev']) for n in ns])
        ax2.plot(ns[1:], err[1:] / err[:-1], color=MUTED, marker=mk, ms=3.2, mfc='white', lw=0.8, ls='--',
                 label=f'ADC($n$) error, {lab}, aug-cc-pVDZ')
    ax1.set_yscale('log')
    ax1.set_xlabel('$k$')
    ax1.set_ylabel(r'$|E^{(k)}|$ (Ha)')
    ax1.legend(frameon=False, fontsize=7)
    ax1.text(0.98, 0.62, 'filled: $E^{(k)}>0$\nopen: $E^{(k)}<0$', transform=ax1.transAxes, ha='right',
             va='top', fontsize=6.5, color=MUTED)
    ax2.axhline(1.0, color=MUTED, lw=0.8, zorder=0)
    ax2.set_xlabel('$k$ (MP) or $n$ (ADC)')
    ax2.set_ylabel('ratio of successive terms')
    ax2.set_ylim(0, 1.3)
    ax2.set_xlim(2.5, 30.5)
    ax2.legend(frameon=False, fontsize=6.5, loc='lower right')
    for ax, tag in ((ax1, '(a)'), (ax2, '(b)')):
        ax.grid(axis='y', color=GRID, lw=0.5)
        for side in ('top', 'right'):
            ax.spines[side].set_visible(False)
        ax.text(0.01, 1.01, tag, transform=ax.transAxes, fontsize=8, fontweight='bold', va='bottom')
    fig.tight_layout(pad=0.4, w_pad=1.2)
    fig.savefig(args.out)
    print(f'-> {args.out}')


if __name__ == '__main__':
    main()
