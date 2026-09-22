#!/usr/bin/env python
"""
Example: linear H4 (R = 1 Angstrom), CAS(2,2) with the CASSCF HOMO and LUMO,
the setting of Fig. 6 in Wang, Fang, and Li (arXiv:2604.16013).

The script builds the common Hermitian matrix of Eq. (7) in
notes/comparison.tex in four variants

    CAS (Dyall reference)      : K_B = 0, no bath
    MR-GW                      : K_B = 0, MR-GW bath          (Wang-Fang-Li)
    MR-GW + mixed (bare)       : K_B with I = vbar_R,  bath
    MR-GW + mixed (screened)   : K_B with I = v_R - W_D(0)^x, bath   (Objective II)

solves for the principal ionization and attachment roots with a Davidson
solver using root following, compares with dense diagonalization and with
full CI, and plots the spectral functions (including the non-PSD Hall-form
insertion of the mixed blocks for comparison).

Run:  python example_h4.py [--basis sto-6g] [--r 1.0] [--eta 0.1] [--no-plot]
"""
import os
import sys
import argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mrgw_hermitian import (build_h4, run_casscf, DyallReference, MRRPA,
                            EKTERPAReference, HermitianGF, hall_insertion_gf,
                            davidson_root_following, FCIReference, HARTREE2EV)

EV = HARTREE2EV


def principal_poles(gf, nmax=4, wmin=1e-3):
    E, Z, w = gf.poles()
    out = {}
    for name, sel in (('remove', E < 0), ('attach', E > 0)):
        idx = np.where(sel & (w > wmin))[0]
        idx = idx[np.argsort(-w[idx])][:nmax]
        out[name] = [(E[i], w[i]) for i in idx]
    return out


def fmt_poles(poles):
    return ", ".join(f"{e * EV:8.3f} eV ({w:.3f})" for e, w in poles)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--basis', default='sto-6g')
    ap.add_argument('--r', type=float, default=1.0)
    ap.add_argument('--ncas', type=int, default=2)
    ap.add_argument('--nelecas', type=int, default=2)
    ap.add_argument('--eta', type=float, default=0.1, help='broadening in eV')
    ap.add_argument('--no-plot', action='store_true')
    ap.add_argument('--no-fci', action='store_true')
    ap.add_argument('--tol', type=float, default=1e-8, help='Davidson residual tolerance')
    args = ap.parse_args()

    print("=" * 78)
    print(f"Linear H4, R = {args.r} A, basis {args.basis}, CAS({args.nelecas},{args.ncas})")
    print("=" * 78)
    mol = build_h4(args.r, args.basis)
    mf, mc = run_casscf(mol, args.ncas, args.nelecas)
    print(f"RHF energy    = {mf.e_tot:.10f} Ha")
    print(f"CASSCF energy = {mc.e_tot:.10f} Ha")
    ref = DyallReference(mc)
    rpa = MRRPA(ref)

    # EKT charged poles + ERPA active response instead of the exact CAS states
    ekt = EKTERPAReference(ref)
    rpa_ekt = MRRPA(ekt)

    variants = [
        ('CAS (Dyall reference)', HermitianGF(ref, None, mixed='none', bath=False)),
        ('MR-GW', HermitianGF(ref, rpa, mixed='none', bath=True)),
        ('MR-GW + mixed (bare)', HermitianGF(ref, rpa, mixed='bare', bath=True)),
        ('MR-GW + mixed (screened)', HermitianGF(ref, rpa, mixed='screened', bath=True)),
        ('EKT/ERPA MR-GW', HermitianGF(ekt, rpa_ekt, mixed='none', bath=True)),
        ('EKT/ERPA MR-GW + mixed (screened)', HermitianGF(ekt, rpa_ekt, mixed='screened', bath=True)),
    ]
    fci = None if args.no_fci else FCIReference(ref)

    # ------------------------------------------------------------------
    # Davidson with root following
    # ------------------------------------------------------------------
    homo = ref.act_so[0]           # lowest active spatial orbital, alpha spin
    lumo = ref.act_so[-2]          # highest active spatial orbital, alpha spin
    core = ref.core_so[0]
    virt = ref.virt_so[0]
    targets = [('principal IP  (HOMO-like, removal)', lambda g: g.orbital_guess(homo, 'remove')),
               ('principal EA  (LUMO-like, attachment)', lambda g: g.orbital_guess(lumo, 'attach')),
               (f'core IP       ({ref.so_label(core)})', lambda g: g.unit_vector(g.index_inactive(core))),
               (f'virtual EA    ({ref.so_label(virt)})', lambda g: g.unit_vector(g.index_inactive(virt)))]

    print()
    print("Davidson root following (targets: HOMO removal, LUMO attachment, core removal, virtual attachment)")
    print("-" * 78)
    dav_results = {}
    for name, gf in variants:
        print(f"{name}: matrix dimension {gf.n} (top {gf.ntop}, bath {gf.nB})")
        guesses = np.array([t[1](gf) for t in targets]).T
        theta, X, info = davidson_root_following(gf.matvec, gf.diagonal(), guesses,
                                                 tol=args.tol, unit=EV, unit_name='(eV)',
                                                 verbose=False)
        Ed, U = gf.eig()
        dense = Ed[np.argmax(np.abs(guesses.T @ U), axis=1)]
        weights = np.sum((gf.T @ X) ** 2, axis=0)
        print(f"  converged in {info['iterations']} iterations, max residual {info['residual_norms'].max():.1e}")
        for k, (tname, _) in enumerate(targets):
            print(f"  {tname:40s} Davidson {theta[k] * EV:9.3f} eV   dense {dense[k] * EV:9.3f} eV"
                  f"   weight {weights[k]:.3f}   overlap with guess {np.abs(guesses[:, k] @ X[:, k]):.3f}")
        dav_results[name] = (theta, weights)

    print()
    print("Active-space excitation energies entering the reference polarizability (eV)")
    print("-" * 78)
    om_exact, _ = ref.neutral_transition_densities()
    print("  exact CAS states : " + ", ".join(f"{w * EV:7.3f}" for w in np.sort(om_exact)))
    print("  ERPA (singles)   : " + ", ".join(f"{w * EV:7.3f}" for w in np.sort(ekt.erpa_omega)))
    print("  EKT charged poles: " + ", ".join(f"{k * EV:7.3f}" for k in np.sort(ekt.kappa))
          + "   (exact: " + ", ".join(f"{k * EV:7.3f}" for k in np.sort(ref.kappa)) + ")")

    # ------------------------------------------------------------------
    # pole tables
    # ------------------------------------------------------------------
    print()
    print("Most intense poles (energy relative to the N-electron ground state; weight = Tr Z Z^T)")
    print("-" * 78)
    for name, gf in variants:
        pp = principal_poles(gf)
        print(f"{name}")
        print(f"  removal    : {fmt_poles(pp['remove'])}")
        print(f"  attachment : {fmt_poles(pp['attach'])}")
    if fci is not None:
        pp = fci.principal_poles()
        print("FCI")
        print(f"  removal    : {fmt_poles(pp['remove'])}")
        print(f"  attachment : {fmt_poles(pp['attach'])}")

    # ------------------------------------------------------------------
    # sum rules
    # ------------------------------------------------------------------
    print()
    print("Sum rules and first-moment diagnostic (M1 = int omega A(omega) d omega)")
    print("-" * 78)
    M1_ref = None
    if fci is not None:
        M1_ref = fci.moments()[1]
    IA = np.ix_(ref.inact_so, ref.act_so)
    for name, gf in variants:
        M0, M1 = gf.moments()
        E, Z, w = gf.poles()
        line = (f"{name:34s} |M0 - 1| = {np.abs(M0 - np.eye(ref.nso)).max():.1e}   "
                f"min weight = {w.min():+.1e}")
        if M1_ref is not None:
            line += (f"   |M1 - M1(FCI)|: full {np.abs(M1 - M1_ref).max():.2e}, "
                     f"inactive-active block {np.abs((M1 - M1_ref)[IA]).max():.2e} Ha")
        print(line)
    if M1_ref is not None:
        print("  (the inactive-active block of M1 is where the mixed couplings enter; "
              "the bare kernel makes it exact to first order in V_R)")

    # ------------------------------------------------------------------
    # spectral functions
    # ------------------------------------------------------------------
    if args.no_plot:
        return
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, skipping the plot")
        return
    eta = args.eta / EV
    allK = np.concatenate([g.poles()[0] for _, g in variants[:2]])
    wmin, wmax = -30.0, 25.0
    omegas = np.linspace(wmin, wmax, 2200) / EV
    fig, axes = plt.subplots(len(variants) + 1, 1, figsize=(7.5, 2.1 * (len(variants) + 1)),
                             sharex=True)
    for ax, (name, gf) in zip(axes, variants):
        A = gf.spectral_function(omegas, eta)
        ax.plot(omegas * EV, A / EV, color='C0', lw=1.4, label=name)
        if fci is not None:
            ax.plot(omegas * EV, fci.spectral_function(omegas, eta) / EV, color='k', lw=0.9, ls='--', label='FCI')
        ax.set_ylabel(r'$A(\omega)$ (1/eV)')
        ax.legend(loc='upper left', fontsize=8, frameon=False)
    # Hall-form insertion of the bare mixed blocks (not PSD)
    ax = axes[-1]
    A_hall = np.array([-np.trace(hall_insertion_gf(ref, rpa, w + 1j * eta, mixed='bare')).imag / np.pi
                       for w in omegas])
    ax.plot(omegas * EV, A_hall / EV, color='C3', lw=1.4,
            label=r'Hall-form insertion of $\Sigma^{12/21}_1$ (MR-GW + $\Sigma_1^{12/21}$, not PSD)')
    if fci is not None:
        ax.plot(omegas * EV, fci.spectral_function(omegas, eta) / EV, color='k', lw=0.9, ls='--', label='FCI')
    ax.axhline(0.0, color='gray', lw=0.5)
    ax.set_ylabel(r'$A(\omega)$ (1/eV)')
    ax.legend(loc='upper left', fontsize=8, frameon=False)
    ax.set_xlabel(r'$\omega$ (eV)')
    fig.suptitle(f'Linear H$_4$, R = {args.r} $\\AA$, {args.basis}, CAS({args.nelecas},{args.ncas}), '
                 f'$\\eta$ = {args.eta} eV')
    fig.tight_layout()
    fname = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         f"h4_{args.basis.replace('*', 's')}_spectral_functions.png")
    fig.savefig(fname, dpi=150)
    print(f"\nspectral functions written to {fname}")
    if A_hall.min() < -1e-6:
        print(f"note: the Hall-form insertion has negative spectral weight, min A = {A_hall.min() / EV:.3e} 1/eV")


if __name__ == '__main__':
    main()
