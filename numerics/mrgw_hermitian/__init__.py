"""
mrgw_hermitian
==============

Pilot implementation of the common Hermitian pole-space matrix of
Eq. (7) in notes/comparison.tex,

        | F_ii + Sigma_stat,ii    K_B^{ia}        A_i      |
    H = | K_B^{ai}                E              M^+ A_a   |
        | A_i^+                   A_a^+ M        E_GW      |

    G(z) = (1  M  0) (z - H)^{-1} (1  M  0)^+ ,

which combines (i) the exact Dyall (CAS) reference propagator, (ii) the
one-shot MR-GW self-energy of Wang, Fang, and Li (arXiv:2604.16013) as a
Hermitian pole bath, and (iii) the (statically screened) inactive--active
mixed couplings restored in Objective II of notes/mr-hedin.tex.

All molecular quantities (SCF, CASSCF, integrals, active-space CI sectors,
transition amplitudes) are obtained with PySCF routines.
"""
from .dyall import (DyallReference, build_h4, build_o3, run_casscf, run_casci,
                    HARTREE2EV, apply_cre, apply_des, dense_ci_hamiltonian)
from .mrrpa import MRRPA
from .ekt_erpa import EKTERPAReference
from .hermitian import HermitianGF, hall_insertion_gf
from .davidson import davidson_root_following
from .fci_reference import (FCIReference, embed_active_vector,
                            check_same_sector_couplings, check_first_order_gf)

__all__ = [
    'DyallReference', 'build_h4', 'build_o3', 'run_casscf', 'run_casci', 'HARTREE2EV',
    'apply_cre', 'apply_des', 'dense_ci_hamiltonian',
    'MRRPA', 'EKTERPAReference', 'HermitianGF', 'hall_insertion_gf', 'davidson_root_following',
    'FCIReference', 'embed_active_vector',
    'check_same_sector_couplings', 'check_first_order_gf',
]
