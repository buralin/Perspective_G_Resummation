import json, numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
qb = json.load(open('n2_sto3g_ma4.json')) if __import__('os').path.exists('n2_sto3g_ma4.json') else json.load(open('n2_sto3g.json'))
ex = json.load(open('n2_class_exact.json'))
R = [r['R'] for r in qb]
fig, ax = plt.subplots(1, 2, figsize=(10, 4.2))
mEh = 1000.0
ax[0].plot(R, [mEh * (r['E_FCI'] - r['E_CAS']) for r in qb], 'k-o', label='FCI $-$ CASSCF')
ax[0].plot(R, [mEh * r['E_NEVPT2'] for r in qb], 'b-s', label='SC-NEVPT2 (all classes)')
ax[0].plot(R, [mEh * r['E2_both'] for r in ex], 'g-^', label='exact-metric E$_2$, (ca,ca)+(ca,aa)')
ax[0].plot(R, [mEh * r['E2_caca'] for r in ex], 'g--', label='exact-metric E$_2$, (ca,ca) only')
ax[0].set_xlabel('R(N$-$N) / Å'); ax[0].set_ylabel('correlation energy beyond CASSCF(6,6) / mE$_h$')
ax[0].set_title('N$_2$ / STO-3G, CASSCF(6,6): references', fontsize=10); ax[0].legend(fontsize=8)
ax[1].plot(R, [mEh * (r['E_FCI'] - r['E_CAS']) for r in qb], 'k-o', label='FCI $-$ CASSCF')
ax[1].plot(R, [mEh * r['E_pt2'] for r in qb], 'r-v', label='quasi-boson 2nd order')
ax[1].plot(R, [mEh * r['E_full'] for r in qb], 'm-d', label='quasi-boson ring+crossed ring')
ax[1].plot(R, [mEh * r['E_erpa_ring_eigen'] for r in qb], 'c:', label='ERPA ring (eigen), 1/4 Tr(Bt)')
ax[1].set_yscale('symlog', linthresh=1.0)
ax[1].set_xlabel('R(N$-$N) / Å'); ax[1].set_ylabel('mE$_h$ (symlog)')
ax[1].set_title('quasi-boson equations, class (ca,ca) only', fontsize=10); ax[1].legend(fontsize=8)
plt.tight_layout(); plt.savefig('n2_sto3g.pdf'); plt.savefig('n2_sto3g.png', dpi=130)
print("wrote n2_sto3g.pdf")
import os
if os.path.exists('n2_631g_ma1.json'):
    d = json.load(open('n2_631g_ma1.json'))
    R = [r['R'] for r in d]
    fig, ax = plt.subplots(1, 1, figsize=(5.5, 4.2))
    nev3 = [mEh * (r['nev_Sijrs0'] + r['nev_Sijr+1'] + r['nev_Srsi-1']) for r in d]
    ax.plot(R, nev3, 'b-s', label='SC-NEVPT2: Sijrs(0)+Sijr(+1)+Srsi($-$1)')
    ax.plot(R, [mEh * r['E_pt2'] for r in d], 'r-v', label='extended 2nd order')
    ax.plot(R, [mEh * r['E_ring'] for r in d], 'g-^', label='ring + crossed ring')
    ax.plot(R, [mEh * r['E_ladder'] for r in d], 'c-<', label='ladder')
    ax.plot(R, [mEh * r['E_full'] for r in d], 'm-d', label='ring + crossed ring + ladder')
    ax.set_xlabel('R(N$-$N) / Å'); ax.set_ylabel('correlation energy / mE$_h$')
    ax.set_title('N$_2$ / 6-31G, CASSCF(6,6), doubles with $\\leq$1 active index', fontsize=10); ax.legend(fontsize=8)
    plt.tight_layout(); plt.savefig('n2_631g.pdf'); plt.savefig('n2_631g.png', dpi=130)
    print("wrote n2_631g.pdf")
