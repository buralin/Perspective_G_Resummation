# Order convergence of ionization energies

Ionization energies against the order n of the method, compared with FCI in the same basis. The values come from
the determinant oracle, and the records carry the code hashes. Records are in three places:
- `numerics/isr_adc/ne_records/`: Ne, Ar and Be;
- `notes/ucc_dyson/records/`: Ne/6-31G and H4;
- `OtherProjects/ISRvsDiagrams/supinf_adcn/oracle_sector_*`: Dyson ADC(n) for Ne in cc-pVDZ and aug-cc-pVDZ.

Where the same series appears in several records, the values agree to 1e-6 eV. The manuscript
figures `fig:NeConvergence` and `fig:NeMP` (c) use the Ne cc-pVDZ and aug-cc-pVDZ part. Energies are in eV, and errors
are method minus FCI. RHF orbitals throughout.

## Systems

| Molecule | Basis | Frozen core | States |
|---|---|---|---|
| Ne | cc-pVDZ, aug-cc-pVDZ, 6-31G | 1s | 2p, 2s |
| Ar | cc-pVDZ, aug-cc-pVDZ | Ne core | 3p, 3s |
| Be | cc-pVDZ, aug-cc-pVDZ | none | 2s, 1s |
| H4 linear chain, spacing 1.0 Å | STO-3G | none | HOMO, HOMO-1 |

## Methods

Not every method exists for every system and order; the file has a row only where a value was computed.

| `method` | Meaning |
|---|---|
| `ADC(n), non-Dyson` | Møller–Plesset partitioning |
| `EN-ADC(n), non-Dyson` | Epstein–Nesbet partitioning |
| `Dyson ADC(n), diagrammatic` | union space decoupled as in Tarantelli and Cederbaum |
| `Dyson ADC(n), union-space ISR` | union-space intermediate-state representation |
| `Dyson ADC(n), two-sector ISR` | two-sector intermediate-state representation |
| `GF(n)` | bare self-energy through order n, solved with the Dyson equation; an empty `ip_eV` means that order has no real root |
| `IP-UCC(n)` | perturbative IP-UCC through order n |
| `IP-EOM-CC` | CCSD, CCSDT, ..., placed at n = 3, 5, 7, ... and named in `cc_level` |

## Files

| File | Content |
|---|---|
| `ip_vs_order.csv` | Columns: `molecule`, `basis`, `frozen_core`, `state`, `method`, `n`, `cc_level`, `ip_eV`, `fci_ip_eV`, `error_eV`, `Z` (pole strength where stored), `in_figure`. `in_figure` = 1 marks the rows of `fig:NeConvergence`. |
| `ne_adc_error_ratio.csv` | Ne, aug-cc-pVDZ, non-Dyson ADC(n): the absolute error against FCI (`abs_error_eV`) and its ratio to the error of order n-1 (`ratio_to_previous`), for 2p and 2s, as in `fig:NeMP` (c). |
