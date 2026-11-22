# ADC(n) and coupled-cluster ionization energies of the SupInf set, aug-cc-pVDZ

These are the data of the manuscript figure `fig:ADCviolin` and of the comparisons quoted around it.

Settings: SupInf states, aug-cc-pVDZ, frozen core. The reference is the selected-CI (sCI) value of the SupInf set
(Marie and Loos, J. Chem. Theory Comput. 20, 4751 (2024)). Energies are in eV, and errors are method minus sCI.
Source: our table `OtherProjects/ISRvsDiagrams/supinf_adcn/supinf_adcn_fc_adz_convergence.csv`.

## `supinf_ips_adc_cc.csv`

Methods (`family`, `method`, `n`):
- Dyson diagrammatic ADC(2)–(5);
- Dyson two-sector ISR-ADC;
- non-Dyson ADC;
- CC2, CCSD, CC3 and CCSDT, taken from the SupInf reference data.

| Column | Meaning |
|---|---|
| `molecule`, `state` | the SupInf state; `state` counts the ionization energies of the molecule from the lowest |
| `sCI_eV` | reference |
| `family` | Dyson ADC, Dyson two-sector ISR-ADC, non-Dyson ADC, or CC |
| `method`, `n` | ADC(n) with n = 2..5, or the CC level (CC2, CCSD, CC3, CCSDT at n = 2..5) |
| `ip_eV`, `error_eV` | ionization energy and its error |
| `in_figure` | 1 for the 52 states of 22 molecules and the eight methods (Dyson ADC(2)–(5), CC2–CCSDT) shown in the figure |

Rows with `in_figure` = 0 give the comparisons quoted in the text, for example ADC(5) in all three constructions on 50
states.
