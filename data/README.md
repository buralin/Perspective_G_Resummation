# Data behind the figures

Plain CSV files with the numbers of our own calculations: the data of the manuscript figures, the perturbation
series of the ground state, and the SupInf benchmarks. Each folder has its own README with the columns, settings and
sources.

Energies are in eV unless a column says Ha. Errors are always method minus reference: positive means the method gives
too large an ionization energy, negative too small.

| Folder | Content | Manuscript figure |
|---|---|---|
| `order_convergence/` | Ionization energies against the order: ADC(n) (non-Dyson, Dyson, EN), GF(n), IP-UCC(n), IP-EOM-CC; Ne, Ar, Be, H4 | Fig. `fig:NeConvergence`, Fig. `fig:NeMP` (c) |
| `ground_state_correlation/` | Rayleigh–Schrödinger series of the ground state and of the Koopmans cation states of Ne, Ar, Be | Fig. `fig:NeMP` (a), (b) |
| `adc_benchmarks_aug-cc-pvdz/` | SupInf ionization energies, ADC(2)–(5) and CC2–CCSDT, aug-cc-pVDZ | Fig. `fig:ADCviolin` |
| `benchmarks_aug-cc-pvtz/` | SupInf ionization energies, aug-cc-pVTZ: Faddeev-ADC family, UCC (Dyson and non-Dyson), EOM-CC | none |

The core-level GW violin of the manuscript uses the data in `core/core_gw_results/`. The table-of-contents
placeholder and `screening.png` contain no calculated data. The supplementary material has no
figures.
