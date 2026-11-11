"""Ne valence IPs against order: non-Dyson ADC(n), IP-EOM-CC and GF(n), each minus FCI, 2 x 2 panels.

Columns: cc-pVDZ and aug-cc-pVDZ (1s frozen); rows: 2p and 2s ionization. Every curve is the IP of the
same symmetry block minus its FCI IP, in eV, on a symmetric-log axis (linear within +-10 meV).
ADC(n): non-Dyson IP-ADC(n), n = 2..11. GF(n): the Dyson root of the bare self-energy series summed
through order n (MBGF(n) of Hirata et al.); orders without a real root are left out (gaps).
IP-EOM-CC: CCSD, CCSDT, CCSDTQ, CCSDTQP, CCSDTQPH at n = 3, 5, 7, 9, 11 (CC level k at n = 2k - 1).

Input: the six records in ne_records/ next to this script (copied from the oracle campaign,
OtherProjects/ISRvsDiagrams/supinf_adcn/oracle_sector_{ccpvdz,adz}/).

usage: python ne_convergence_2x2.py [--data <dir>] [--out <file>]
"""
import argparse
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_OUT = os.path.join(HERE, '..', '..', 'manuscript', 'figures', 'ne_adc_cc_gf.pdf')
INK, MUTED, GRID = '#0b0b0b', '#52514e', '#d9d8d4'
C_ADC, C_CC, C_GF = '#eb6834', '#2a78d6', '#52514e'
BASES = (('cc-pVDZ', 'Ne_ccpvdz_fc1'), ('aug-cc-pVDZ', 'Ne_augccpvdz_fc1'))
LEVELS = ((3, '2p'), (0, '2s'))
LINTHRESH = 1e-2


def block(blocks, level):
    """The entry of one symmetry block (oracle level) in a record's block list."""
    return next(b for b in blocks if b['level'] == level)


def read(data, system):
    """{level: (FCI IP, ADC {n: IP}, CC {n: IP}, GF {n: IP or nan})} of one basis."""
    with open(os.path.join(data, f'sector_oracle_{system}_n11.json')) as fh:
        oracle = json.load(fh)['blocks']
    with open(os.path.join(data, f'sector_cc_{system}_n23456.json')) as fh:
        cc = json.load(fh)['cc']
    with open(os.path.join(data, f'sector_gf_{system}_n11.json')) as fh:
        gf = json.load(fh)['blocks']
    out = {}
    for level, _ in LEVELS:
        b = block(oracle, level)
        adc = {int(n): o['ip_ev'] for n, o in b['orders'].items()}
        ccs = {2 * int(k) - 1: block(e['blocks'], level)['ip_ev'] for k, e in cc.items()}
        gfs = {int(n): (float('nan') if o['ip_ev'] is None else o['ip_ev'])
               for n, o in block(gf, level)['orders'].items()}
        out[level] = (b['fci_ip_ev'], adc, ccs, gfs)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--data', default=os.path.join(HERE, 'ne_records'))
    ap.add_argument('--out', default=DEFAULT_OUT)
    args = ap.parse_args()

    plt.rcParams.update({'font.size': 7, 'axes.edgecolor': MUTED, 'axes.labelcolor': INK,
                         'xtick.color': MUTED, 'ytick.color': MUTED, 'text.color': INK})
    fig, axes = plt.subplots(2, 2, figsize=(3.4, 3.2), sharex=True, sharey=True)
    for col, (basis, system) in enumerate(BASES):
        data = read(args.data, system)
        for row, (level, label) in enumerate(LEVELS):
            ax = axes[row, col]
            fci, adc, cc, gf = data[level]
            for series, color, marker, ls, name in ((gf, C_GF, '^', ':', 'GF($n$)'),
                                                    (adc, C_ADC, 'o', '-', 'ADC($n$)'),
                                                    (cc, C_CC, 's', '--', 'IP-EOM-CC')):
                ns = sorted(series)
                ax.plot(ns, [series[n] - fci for n in ns], color=color, marker=marker, ms=2.8, lw=0.9,
                        ls=ls, mfc='white' if marker != 'o' else color, mew=0.8, label=name)
            ax.axhline(0.0, color=MUTED, lw=0.6, zorder=0)
            ax.axhspan(-LINTHRESH, LINTHRESH, color=GRID, alpha=0.35, lw=0, zorder=0)
            ax.set_yscale('symlog', linthresh=LINTHRESH, linscale=0.6)
            ax.set_yticks([-1, -0.1, -0.01, 0, 0.01, 0.1, 1])
            ax.set_yticklabels(['$-1$', '$-0.1$', '', '0', '', '0.1', '1'])
            ax.set_ylim(-5, 5)
            ax.set_xticks(range(2, 12))
            ax.tick_params(length=2, pad=1.5)
            for side in ('top', 'right'):
                ax.spines[side].set_visible(False)
            ax.text(0.98, 0.98, label, transform=ax.transAxes, ha='right', va='top', fontsize=7)
            ax.text(0.02, 0.98, f'({"abcd"[2 * row + col]})', transform=ax.transAxes, va='top',
                    fontsize=7, fontweight='bold')
            if row == 0:
                ax.set_title(basis, fontsize=7, pad=2)
    for ax in axes[1]:
        ax.set_xlabel('$n$')
    for ax in axes[:, 0]:
        ax.set_ylabel('IP $-$ FCI (eV)')
    handles, names = axes[0, 0].get_legend_handles_labels()
    order = (1, 2, 0)   # ADC, CC, GF
    fig.legend([handles[i] for i in order], [names[i] for i in order], frameon=False, fontsize=6.5,
               loc='upper center', ncol=3, handlelength=2.2, columnspacing=1.2, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(pad=0.3, h_pad=0.4, w_pad=0.4, rect=(0, 0, 1, 0.93))
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    fig.savefig(args.out)
    print(f'-> {os.path.abspath(args.out)}')


if __name__ == '__main__':
    main()
