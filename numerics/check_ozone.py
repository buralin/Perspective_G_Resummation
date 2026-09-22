#!/usr/bin/env python
"""
Ozone check against Wang, Fang, and Li, arXiv:2604.16013 (Fig. 4, Tables S3-S5).

Setting of the paper: O3 at the CCCBDB experimental geometry (R = 1.278 A,
theta = 116.8 deg), 6-31G basis, CASCI with RHF orbitals,
CAS(6,4) = {4b2, 6a1, 1a2, 2b1} (default) or CAS(8,5) = {1b1, 4b2, 6a1, 1a2, 2b1}.

The three lowest vertical ionization energies (2A1, 2B2, 2A2) are obtained by
Davidson root following, starting from the CASCI cation states, for

    MR-GW                     (K_B = 0; this is the Wang-Fang-Li method)
    MR-GW + mixed (bare)      (K_B with the bare kernel vbar_R)
    MR-GW + mixed (screened)  (K_B with v_R - W_D(0)^x, Objective II)

and compared with the CASCI, MR-GW, DMRG and experimental values of the paper.

Run:  python check_ozone.py [--cas 6,4 | --cas 8,5] [--plot] [--nomega 160]
"""
import os
import sys
import time
import argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mrgw_hermitian import (build_o3, run_casci, DyallReference, MRRPA,
                            HermitianGF, davidson_root_following, HARTREE2EV)

EV = HARTREE2EV

# Wang, Fang, Li, Table S3 (RHF/6-31G orbital energies, Hartree)
WFL_ORBITAL_ENERGIES = {'5a1': -0.83815, '3b2': -0.79362, '1b1': -0.79183,
                        '4b2': -0.57902, '6a1': -0.56193, '1a2': -0.49558,
                        '2b1': -0.06794, '7a1': 0.26189, '5b2': 0.36524}
# Wang, Fang, Li, Table S5 (vertical IPs in eV for 2A1, 2B2, 2A2; 6-31G)
WFL_IP = {
    'KT': (15.29, 15.76, 13.49),
    'GW@RHF': (13.25, 13.52, 13.20),
    'ADC(2)': (10.69, 10.79, 13.05),
    'ADC(3)': (12.61, 12.79, 12.49),
    'CAS(6,4)': (14.38, 14.67, 15.12),
    'MR-GW(6,4)': (12.65, 12.94, 14.56),
    'CAS(8,5)': (13.59, 13.92, 13.10),
    'MR-GW(8,5)': (11.58, 12.09, 12.94),
    'CAS(8,6)': (13.78, 14.10, 13.28),
    'MR-GW(8,6)': (11.93, 12.33, 13.09),
    'EOM-CCSD': (12.42, 12.64, 13.38),
    'DMRG': (12.34, 12.59, 13.34),
    'Exp.': (12.73, 13.00, 13.54),
}
STATES = ('A1', 'B2', 'A2')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cas', default='6,4', help='"nelec,norb", e.g. 6,4 or 8,5')
    ap.add_argument('--basis', default='6-31g')
    ap.add_argument('--tol', type=float, default=1e-7, help='Davidson residual tolerance')
    ap.add_argument('--plot', action='store_true', help='spectral functions in the IP window')
    ap.add_argument('--nomega', type=int, default=160)
    ap.add_argument('--force', action='store_true', help='run even if the bath is very large')
    args = ap.parse_args()
    nelecas, ncas = (int(x) for x in args.cas.split(','))
    cas_label = f"({nelecas},{ncas})"

    print("=" * 84)
    print(f"Ozone, CCCBDB geometry, {args.basis}, CASCI{cas_label} with RHF orbitals "
          "(Wang, Fang, Li, arXiv:2604.16013)")
    print("=" * 84)
    t0 = time.time()
    mol = build_o3(args.basis)
    mf, mc = run_casci(mol, ncas, nelecas)
    print(f"RHF energy    = {mf.e_tot:.8f} Ha")
    print(f"CASCI energy  = {mc.e_tot:.8f} Ha")

    # ---- RHF orbitals vs Table S3 ----
    from pyscf import symm
    orbsym = mf.mo_coeff.orbsym
    names = [symm.irrep_id2name(mol.groupname, s) for s in orbsym]
    counter = {}
    labels = []
    for nm in names:
        counter[nm] = counter.get(nm, 0) + 1
        labels.append(f"{counter[nm]}{nm.lower()}")
    print("\nRHF orbital energies (Ha) vs Wang-Fang-Li Table S3:")
    dev = 0.0
    for i in range(6, 15):
        ref_e = WFL_ORBITAL_ENERGIES.get(labels[i])
        dev = max(dev, abs(mf.mo_energy[i] - ref_e)) if ref_e is not None else dev
        print(f"  {i:2d} {labels[i]:4s} {mf.mo_energy[i]:+.5f}   paper {ref_e if ref_e is not None else '  n/a'}"
              + ("   <- active" if mc.ncore <= i < mc.ncore + mc.ncas else ""))
    print(f"  max deviation {dev:.1e} Ha")
    active_labels = [labels[i] for i in range(mc.ncore, mc.ncore + mc.ncas)]
    print(f"  active orbitals: {', '.join(active_labels)}")

    # ---- CASCI wave function ----
    ci = np.asarray(mc.ci)
    idx = np.argsort(-np.abs(ci).ravel())[:3]
    print("leading CASCI coefficients: " + ", ".join(f"{ci.ravel()[k]:+.4f}" for k in idx)
          + "   (paper, CAS(6,4): +0.936, -0.349)")

    # ---- Dyall reference and MR-RPA ----
    print()
    ref = DyallReference(mc)
    est_bath = 0
    if not args.force:
        nI_att = np.sum(ref.pole_sign > 0)
        nI_rem = np.sum(ref.pole_sign < 0)
        nchan = (len(ref.core_so) * len(ref.virt_so) // 2 + len(ref.core_so) * nI_att // 2
                 + len(ref.virt_so) * nI_rem // 2 + ref.sectors[ref.nelecas].nstates)
        est_bath = nchan * (len(ref.inact_so) + ref.npoles) // 2
        if est_bath > 3_000_000:
            print(f"estimated bath size {est_bath:.2e} is too large for the dense bath storage of "
                  "this pilot code; use --force to try anyway")
            return
    t1 = time.time()
    rpa = MRRPA(ref)
    print(f"  MR-RPA time {time.time() - t1:.1f} s")

    # ---- CASCI cation states ----
    na, nb = ref.nelecas
    sector = (na - 1, nb)
    cas_states = {}
    print(f"\nCASCI{cas_label} cation states in sector {sector} (lowest of each irrep):")
    for ia, p in enumerate(ref.poles):
        if p.sign > 0 or p.sector != sector:
            continue
        nm = ref.irrep_name(ref.pole_symmetry(ia))
        if nm not in cas_states or -p.kappa < -ref.poles[cas_states[nm]].kappa:
            cas_states[nm] = ia
    for nm in STATES:
        if nm in cas_states:
            ia = cas_states[nm]
            print(f"  2{nm}: IP = {-ref.poles[ia].kappa * EV:7.3f} eV   "
                  f"(paper CAS{cas_label}: {WFL_IP.get('CAS' + cas_label, ('n/a',) * 3)[STATES.index(nm)]})")
    missing = [nm for nm in STATES if nm not in cas_states]
    if missing:
        print(f"  warning: no CASCI cation state of symmetry {missing} in this sector")

    # ---- Hermitian variants and Davidson root following ----
    variants = [
        ('MR-GW', dict(mixed='none', bath=True)),
        ('MR-GW + mixed (bare)', dict(mixed='bare', bath=True)),
        ('MR-GW + mixed (screened)', dict(mixed='screened', bath=True)),
    ]
    results = {'CAS': {nm: -ref.poles[cas_states[nm]].kappa * EV for nm in STATES if nm in cas_states}}
    weights = {}
    gfs = {}
    for name, kw in variants:
        t2 = time.time()
        gf = HermitianGF(ref, rpa, **kw)
        gfs[name] = gf
        print(f"\n{name}: matrix dimension {gf.n} (top {gf.ntop}, bath {gf.nB}); build {time.time() - t2:.1f} s")
        order = [nm for nm in STATES if nm in cas_states]
        guesses = np.array([gf.unit_vector(gf.index_active_pole(cas_states[nm])) for nm in order]).T
        theta, X, info = davidson_root_following(gf.matvec, gf.diagonal(), guesses, tol=args.tol,
                                                 unit=EV, unit_name='(eV)', verbose=True)
        w = np.sum((gf.T @ X) ** 2, axis=0)
        results[name] = {nm: -theta[k] * EV for k, nm in enumerate(order)}
        weights[name] = {nm: w[k] for k, nm in enumerate(order)}
        print(f"  {info['iterations']} iterations, converged: {info['converged'].all()}, "
              f"time {time.time() - t2:.1f} s")
        for k, nm in enumerate(order):
            print(f"  2{nm}: IP = {-theta[k] * EV:7.3f} eV, weight {w[k]:.3f}, "
                  f"overlap with CAS state {np.abs(guesses[:, k] @ X[:, k]):.3f}")

    # ---- comparison table ----
    print("\n" + "=" * 84)
    print(f"Vertical ionization energies of O3 (eV), 6-31G, CAS{cas_label}")
    print("=" * 84)
    rows = [('KT', WFL_IP['KT']), ('GW@RHF (paper)', WFL_IP['GW@RHF'])]
    key = 'CAS' + cas_label
    if key in WFL_IP:
        rows.append((f'CAS{cas_label} (paper)', WFL_IP[key]))
    rows.append((f'CAS{cas_label} (this work)', tuple(results['CAS'].get(nm, np.nan) for nm in STATES)))
    key = 'MR-GW' + cas_label
    if key in WFL_IP:
        rows.append((f'MR-GW{cas_label} (paper)', WFL_IP[key]))
    for name, _ in variants:
        rows.append((f'{name} (this work)', tuple(results[name].get(nm, np.nan) for nm in STATES)))
    rows += [('EOM-CCSD (paper)', WFL_IP['EOM-CCSD']), ('DMRG (paper)', WFL_IP['DMRG']),
             ('Exp. (paper)', WFL_IP['Exp.'])]
    print(f"{'method':36s} {'2A1':>8s} {'2B2':>8s} {'2A2':>8s}   ordering")
    for name, vals in rows:
        order = ' < '.join('2' + STATES[k] for k in np.argsort(vals))
        print(f"{name:36s} " + " ".join(f"{v:8.2f}" for v in vals) + f"   {order}")
    if key in WFL_IP:
        d = [results['MR-GW'][nm] - WFL_IP[key][k] for k, nm in enumerate(STATES) if nm in results['MR-GW']]
        print(f"\nMR-GW (this work) - MR-GW (paper): " + ", ".join(f"{x:+.3f}" for x in d) + " eV")
    print(f"total time {time.time() - t0:.1f} s")

    # ---- optional spectral functions in the IP window ----
    if not args.plot:
        return
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, skipping the plot")
        return
    eta = 0.1 / EV
    omegas = np.linspace(-20.0, -8.0, args.nomega) / EV
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    gf_cas = HermitianGF(ref, None, mixed='none', bath=False)
    ax.plot(omegas * EV, gf_cas.spectral_function(omegas, eta) / EV, color='gray', lw=1.0,
            label=f'CASCI{cas_label}')
    for k, (name, _) in enumerate(variants):
        t3 = time.time()
        A = gfs[name].spectral_function_resolvent(omegas, eta)
        print(f"spectral function {name}: {time.time() - t3:.1f} s")
        ax.plot(omegas * EV, A / EV, lw=1.3, label=name)
    for k, nm in enumerate(STATES):
        ax.axvline(-WFL_IP['Exp.'][k], color='k', ls=':', lw=0.8)
    ax.set_xlabel(r'$\omega$ (eV)')
    ax.set_ylabel(r'$A(\omega)$ (1/eV)')
    ax.set_title(f'O$_3$, {args.basis}, CAS{cas_label}; dotted: experimental IPs')
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fname = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         f"o3_cas{nelecas}_{ncas}_spectral_functions.png")
    fig.savefig(fname, dpi=150)
    print(f"spectral functions written to {fname}")


if __name__ == '__main__':
    main()
