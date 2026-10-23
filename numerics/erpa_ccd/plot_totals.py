import json, os, numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
hf = json.load(open('n2_631g_hf.json'))
cas = json.load(open('n2_631g_ma1.json'))
fig, ax = plt.subplots(1, 2, figsize=(11, 4.4), sharey=False)
def curve(a, d, key, label, style, conv=None):
    R = [r['R'] for r in d if (conv is None or r.get(conv, True))]
    E = [r['E_HF' if 'E_HF' in r else 'E_CAS'] + (r[key] if key else 0.0) for r in d if (conv is None or r.get(conv, True))]
    a.plot(R, E, style, label=label)
curve(ax[0], hf, None, 'RHF', 'k-o')
curve(ax[0], hf, 'E_pt2', '2nd order (= MP2)', 'r-v')
curve(ax[0], hf, 'E_ring', 'ring + crossed ring', 'g-^', 'conv_ring')
curve(ax[0], hf, 'E_ladder', 'ladder', 'c-<', 'conv_ladder')
curve(ax[0], hf, 'E_full', 'ring + crossed ring + ladder', 'm-d', 'conv_full')
ax[0].set_title('N$_2$ / 6-31G, RHF determinant as vacuum', fontsize=10)
curve(ax[1], cas, None, 'CASSCF(6,6)', 'k-o')
curve(ax[1], cas, 'E_pt2', '2nd order', 'r-v')
curve(ax[1], cas, 'E_ring', 'ring + crossed ring', 'g-^', 'conv_ring')
curve(ax[1], cas, 'E_ladder', 'ladder', 'c-<', 'conv_ladder')
curve(ax[1], cas, 'E_full', 'ring + crossed ring + ladder', 'm-d', 'conv_full')
ax[1].set_title('N$_2$ / 6-31G, CASSCF(6,6) vacuum (doubles with $\\leq$1 active index)', fontsize=10)
for a in ax:
    a.set_xlabel('R(N$-$N) / Å'); a.set_ylabel('total energy / E$_h$'); a.legend(fontsize=8)
plt.tight_layout(); plt.savefig('n2_631g_totals.pdf'); plt.savefig('n2_631g_totals.png', dpi=130)
print("wrote n2_631g_totals.png")
