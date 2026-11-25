# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# %%
data_order = pd.read_csv("order_convergence/ip_vs_order.csv")
data_order_ne = data_order[data_order['molecule'] == 'Ne']
data_order_ne_ccpvdz = data_order_ne[data_order_ne['basis'] == 'cc-pVDZ']
data_order_ne_accpvdz = data_order_ne[data_order_ne['basis'] == 'aug-cc-pVDZ']
# %%
data_order_ne_accpvdz_ndyson_adc = data_order_ne_accpvdz[data_order_ne_accpvdz['method'] == 'ADC(n), non-Dyson']
data_order_ne_accpvdz_ndyson_gfn = data_order_ne_accpvdz[data_order_ne_accpvdz['method'] == 'GF(n)']
data_order_ne_ccpvdz_ipeomcc = data_order_ne_ccpvdz[data_order_ne_ccpvdz['method'] == 'IP-EOM-CC']
#%%
data_order_ne_ccpvdz_ipeomcc_2s = data_order_ne_ccpvdz_ipeomcc[data_order_ne_ccpvdz_ipeomcc['state'] == '2s']
data_order_ne_ccpvdz_ipeomcc_2p = data_order_ne_ccpvdz_ipeomcc[data_order_ne_ccpvdz_ipeomcc['state'] == '2p']
data_order_ne_accpvdz_ndyson_adc_2s = data_order_ne_accpvdz_ndyson_adc[data_order_ne_accpvdz_ndyson_adc['state'] == '2s']
data_order_ne_accpvdz_ndyson_adc_2p = data_order_ne_accpvdz_ndyson_adc[data_order_ne_accpvdz_ndyson_adc['state'] == '2p']

data_order_ne_accpvdz_ndyson_gfn_2s = data_order_ne_accpvdz_ndyson_gfn[data_order_ne_accpvdz_ndyson_gfn['state'] == '2s']
data_order_ne_accpvdz_ndyson_gfn_2p = data_order_ne_accpvdz_ndyson_gfn[data_order_ne_accpvdz_ndyson_gfn['state'] == '2p']

#%%
fig, (ax1,ax2) = plt.subplots(1, 2, figsize=(8, 4), sharey=True)

ax1.spines['left'].set_linewidth(1.5)
ax1.spines['bottom'].set_linewidth(1.5)
ax1.tick_params(axis='both', which='major', length=6, width=1.5, labelsize=14)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
#ax1.grid(True, linestyle='--', color='gray', alpha=0.4)
ax1.axhline(-0.1, color='gray', linewidth=1.5, linestyle='-', alpha=0.7)
ax1.axhline(0.0, color='black', linewidth=1.5, linestyle='-', alpha=0.7)
ax1.axhline(0.1, color='gray', linewidth=1.5, linestyle='-', alpha=0.7)

ax2.spines['left'].set_linewidth(1.5)
ax2.spines['bottom'].set_linewidth(1.5)
ax2.tick_params(axis='both', which='major', length=6, width=1.5, labelsize=14)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
#ax2.grid(True, linestyle='--', color='gray', alpha=0.4)
ax2.axhline(-0.1, color='gray', linewidth=1.5, linestyle='-', alpha=0.7)
ax2.axhline(0.0, color='black', linewidth=1.5, linestyle='-', alpha=0.7)
ax2.axhline(0.1, color='gray', linewidth=1.5, linestyle='-', alpha=0.7)

ax1.plot(data_order_ne_accpvdz_ndyson_gfn_2s['n'],data_order_ne_accpvdz_ndyson_gfn_2s['error_eV'],marker='s', label='GF(n)',linestyle='--',linewidth=2,markersize=6)
ref = data_order_ne_accpvdz_ndyson_adc_2s
ax1.plot(data_order_ne_accpvdz_ndyson_adc_2s['n'],data_order_ne_accpvdz_ndyson_adc_2s['error_eV'],marker='o', label='ADC(n)',linewidth=2,markersize=6)
ax1.plot(data_order_ne_ccpvdz_ipeomcc_2s['n'],data_order_ne_ccpvdz_ipeomcc_2s['error_eV'],marker='^', label='EOM-IP-CC',linewidth=2,markersize=9,color='tab:green',fillstyle='full',linestyle='-.')
ax1.set_xlabel('n')
ax1.set_ylabel('$\\Delta E$ / eV')
ax1.set_xticks(data_order_ne_accpvdz_ndyson_adc_2s['n'])
ax1.set_ylim([-1.25,1.25])
ax1.set_title('a) 2s', fontsize=14)
ax1.legend(frameon=False)

ax2.plot(data_order_ne_accpvdz_ndyson_gfn_2p['n'],data_order_ne_accpvdz_ndyson_gfn_2p['error_eV'],marker='s', label='GF(n)',linestyle='--',linewidth=2,markersize=6)
ax2.plot(data_order_ne_accpvdz_ndyson_adc_2p['n'],data_order_ne_accpvdz_ndyson_adc_2p['error_eV'],marker='o', label='ADC(n)',linewidth=2,markersize=6)
ax2.plot(data_order_ne_ccpvdz_ipeomcc_2p['n'],data_order_ne_ccpvdz_ipeomcc_2p['error_eV'],marker='^', label='EOM-IP-CC',linewidth=2,markersize=9,color='tab:green',fillstyle='full',linestyle='-.')
ax2.set_xlabel('n')
ax2.set_xticks(data_order_ne_accpvdz_ndyson_adc_2p['n'])
ax2.set_ylim([-1.25,1.25])
ax2.set_title('b) 2p', fontsize=14)

plt.tight_layout() 
plt.savefig('error_vs_n.pdf')
plt.show()

# %%

data_order_ne_accpvdz
# %%
