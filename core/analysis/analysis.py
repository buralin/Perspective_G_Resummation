# %% 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import numpy as np
# %%
data_ref = pd.read_csv("../core_gw_results/results.csv")
print(data_ref['qsGW_SRG_QP_failed_final'].value_counts())
print(data_ref['qsGW_SRG_QP_failed_final'].value_counts())
diff_gw_pbeh = (data_ref['G0W0_PBEh_eV'] - data_ref['CVS_FCI_eV'])
# %%
fig, ax1 = plt.subplots(1, 1, figsize=(8, 4), sharey=True)
# data
error = [
    data_ref['G0W0_HF_error_eV'],
    diff_gw_pbeh,
    data_ref['evgw_error_eV'],
    data_ref['qsGW_SRG_error_eV']
]
positions = [1, 2, 3, 4]
models = ["$G_0W_0$@HF", "$G_0W_0$@PBE($\\alpha$)","ev$GW$@HF", "SRG-qs$GW$"]
maes = [np.mean(np.abs(err)) for err in error]
print(maes)
def plot_violin_with_mae(ax, error, positions, models, maes):
    # violin plots
    violins = ax.violinplot(
        error,
        positions=positions,
        widths=0.6,
        showmeans=False,
        showmedians=False,
        showextrema=False,
    )
    for body in violins["bodies"]:
        body.set_facecolor("lightgray")
        body.set_edgecolor("black")
        body.set_alpha(1)

    # 4. Overlay Jittered Scatter Points
    for idx, (pos, errors) in enumerate(zip(positions, error)):
        rng = np.random.default_rng(42)
        jitter = rng.uniform(-0.02, 0.02, size=len(errors))
        ax.scatter(
            np.full_like(errors, pos, dtype=float) + jitter,
            errors,
            color="gray",
            alpha=0.7,
            s=22,
            zorder=3,
        )

    # 5. Plot MAE Markers & Labels
    ax.plot(
        positions,
        maes,
        color="black",
        marker="_",
        markersize=50,
        label="MAE",
        linestyle="None",
        markeredgewidth=5
    )
    ax.set_xticks(positions)
    ax.set_xticklabels(models)

ax1.spines['left'].set_linewidth(2.5)
ax1.tick_params(axis='both', which='major', length=6, width=1.5, labelsize=14)

ax1.spines['top'].set_visible(False)
ax1.spines['bottom'].set_visible(False)
ax1.spines['right'].set_visible(False)


ax1.grid(True, linestyle='--', color='gray', alpha=0.7)

plot_violin_with_mae(ax1, error, positions, models, maes)

ax1.set_ylabel('$\\Delta E$ / eV')
ax1.axhline(0, color='tab:red', linewidth=2, linestyle='-', alpha=0.7)
plt.tight_layout()
#plt.savefig("violin_plot_core.pdf")
plt.show()
# %%
