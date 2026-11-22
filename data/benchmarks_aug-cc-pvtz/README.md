# SupInf ionization energies, aug-cc-pVTZ

58 SupInf states, aug-cc-pVTZ, frozen core as in the SupInf set: He core for B–Ne, Ne core for Na–Ar, Li and Be
all-electron. The reference `sCI_eV` is the selected-CI value of the SupInf set (Marie and Loos, J. Chem. Theory
Comput. 20, 4751 (2024)). `state` counts the ionization energies of a molecule from the lowest; our values are paired
with the sCI states in energy order. Energies are in eV, and errors are method minus sCI.

## Files

| File | Content |
|---|---|
| `all_methods_long.csv` | Every value of the four files below in one long table, one row per state and method, each value once. Columns: `system`, `state`, `sCI_eV`, `method`, `static`, `ip_eV`, `error_eV`, `Z`, `flag`, `source`. |
| `faddeev_adc_mp2r_ips.csv` | The 21 methods of the Faddeev-ADC notes (`notes/3particleG.tex`, Table IV and the raw-data table):<br>- ADC(3), CCD-ADC(3), EN-ADC(3)<br>- Faddeev-ADC(3) with RPA, T1, T0, CCD and dRPA pair channels<br>- FLEX, PSD1, PSD2, GF2, GW<br>- EOM-CCD, and EOM-IP-CCSD from the SupInf data<br>For FLEX and PSD, each state uses the root with the largest pole strength within 8 eV of the same orbital's ADC(3) value. An empty cell means no value, or that the method is not defined for that molecule. |
| `faddeev_adc4_ips.csv` | Faddeev-ADC(4) (CCD pair channel), ADC(4), ADC(3) and CCD-Faddeev-ADC(3), each with its pole strength Z (Table VI of the notes). |
| `third_order_mp2u_ips.csv` | The third-order-complete methods with the MP2-unrelaxed static self-energy (Table V of the notes). |
| `ucc_eom_ips.csv` | - IP-UCCSD[M=2] with 1h+2h1p and with +3h2p (without the Dyson form)<br>- Dyson-UCCSD[M=2] with 2h1p+2p1h and with +3h2p+3p2h<br>- IP-EOM-CCSD+3h2p: CCSD ground state, ion space 1h+2h1p+3h2p<br>- EOM-IP-CCSD and EOM-IP-CC3 from the SupInf data |

## Columns `static` and `flag`

`static` names the static (frequency-independent) self-energy:

| Label | Meaning |
|---|---|
| `MP2r` | MP2 density, relaxed |
| `MP2u` | MP2 density, unrelaxed |
| `MP3u` | MP3 density, unrelaxed |
| `MP2r/CCD`, `MP2u/CCD` | the MP2 functional evaluated on the density of the CCD amplitudes |
| `MP2r/EN`, `MP2u/EN` | the MP2 functional evaluated on the density of the Epstein–Nesbet-dressed amplitudes |
| `--` | none, or not applicable (coupled cluster, UCC, GW) |

`flag` marks a root with a small pole strength Z, which may be satellite-like rather than a clean quasiparticle line:

| Flag | Meaning |
|---|---|
| `WEAK` | Z < 0.5 |
| `LOW` | Z < 0.3 |
| `Z<0.5`, `Z<0.3` | the same, from the Faddeev-ADC(4) table |

Flags and Z come from the quasiparticle audit of the notes and are attached only where the audited root is the value
in the file.

## Sources, settings and runs

- **Faddeev-ADC family:**
  - MBPTcode 9e03589, density fitting with aug-cc-pVTZ-RI.
  - Results in `OtherProjects/FaddeevADC/results/`: `supinf_atz_fc_roots/`, `supinf_atz_fc_adc4_ccdstatic/` and `supinf_atz_fc_mp2u_ccdstatic/`.
  - The Snellius and ADA campaigns are listed in `OtherProjects/FaddeevADC/HANDOFF_2026-11-12_supinf_atz_tables.md`.
- **UCC and EOM-CC:** exact integrals, RHF, all in `OtherProjects/FaddeevADC/ip_qucc/`.
  - IP-UCCSD[M=2]: `results/atz/states.csv`.
  - Dyson-UCCSD[M=2] 2h1p+2p1h: wicks af1b552, ADA job 1233456.
  - Dyson-UCCSD[M=2] +3h2p+3p2h: the 3p2h particles are restricted to 70 % of the MP2 virtual natural orbitals; wicks 29c4e10, 51ef428, 70a0b30 (ADA) and 6fadbb1 (Snellius), all the same equations.
  - IP-EOM-CCSD+3h2p: wicks dfaff58, ADA job 1233044.
  - The state selection is that of `results/atz/note/stats.md`.
