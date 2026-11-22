# Perturbation series of the ground state and the Koopmans cation states

These are Rayleigh–Schrödinger perturbation series at full-CI accuracy, through order k = 30, from the determinant
oracle (records in `numerics/isr_adc/ne_records/`). The manuscript figure `fig:NeMP` (a), (b) shows the Ne,
Møller–Plesset part.

## What is included

- **Atoms and bases:** Ne (1s frozen), Ar (Ne core frozen) and Be (all electrons), each in cc-pVDZ and aug-cc-pVDZ, RHF. These are all the systems with such series.
- **Partitionings:**
  - Møller–Plesset (`MP`): the zeroth-order Hamiltonian is the Fock operator.
  - Epstein–Nesbet (`EN`): the zeroth-order Hamiltonian is the diagonal of H.
- **Series:**
  - the ground state (`N`);
  - the two highest Koopmans cation states (`N-1, 2p` and `N-1, 2s` for Ne, `N-1, 3p` and `N-1, 3s` for Ar, `N-1, 2s` and `N-1, 1s` for Be).

The energy through order K is the sum of the coefficients up to K. E^(0) and E^(1) depend on how the frozen-core
constant is split between them; only their sum is meaningful, which is why the figure starts at k = 2.

## Files

| File | Content |
|---|---|
| `rs_series_coefficients.csv` | Energy coefficients E^(k) in Ha. Columns: `atom`, `basis`, `partitioning`, `series`, `k`, `E_k_Ha`. |
| `rs_ground_state_norm.csv` | The order-k contribution to the norm of the ground state in intermediate normalization (`norm_k`, with `norm_0` = 1). Columns: `atom`, `basis`, `partitioning`, `k`, `norm_k`. |
| `ne_mp_coefficients_fig2.csv` | Exactly the rows plotted in `fig:NeMP` (a), (b): Ne, MP, k = 2..30. Columns: `basis`, `series`, `k`, `E_k_Ha`. |
