import sys, json, re, io, numpy as np
from pyscf import gto, scf, mcscf, fci, mrpt
from erpa_ccd import *
basis = sys.argv[1] if len(sys.argv) > 1 else 'sto-3g'
max_active = int(sys.argv[2]) if len(sys.argv) > 2 else 4
do_fci = (basis == 'sto-3g')
Rs = [0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.6, 1.8, 2.0, 2.4, 3.0]
results = []
mo_prev = None
for R in Rs:
    mol = gto.M(atom='N 0 0 0; N 0 0 %f' % R, basis=basis, unit='angstrom', verbose=0, symmetry=False)
    mf = scf.RHF(mol).run()
    mc = mcscf.CASSCF(mf, 6, 6); mc.natorb = True; mc.max_cycle_macro = 200
    if mo_prev is not None:
        mo0 = mcscf.project_init_guess(mc, mo_prev, prev_mol=prev_mol)
        mc.kernel(mo0)
    else:
        mc.kernel()
    if not mc.converged:
        print("CASSCF not converged at R=%.2f" % R)
    mo_prev, prev_mol = mc.mo_coeff, mol
    rec = dict(R=R, E_HF=mf.e_tot, E_CAS=mc.e_tot)
    if do_fci:
        rec['E_FCI'] = fci.FCI(mf).kernel()[0]
    # NEVPT2 with class resolution
    pt = mrpt.NEVPT(mc); pt.verbose = 4; buf = io.StringIO(); pt.stdout = buf
    e_pt = pt.kernel()
    rec['E_NEVPT2'] = e_pt
    for line in buf.getvalue().splitlines():
        m = re.match(r"\s*(S\w+)\s+\(([^)]*)\)('?)\s*,\s*E = ([-\d.]+)", line)
        if m:
            rec['nev_' + m.group(1) + m.group(2) + m.group(3)] = float(m.group(4))
    # extended channels
    h, v, w = so_integrals(mol, mc.mo_coeff)
    g, G, C, A, V = cas_vacuum(mc)
    rec['E_rdm_check'] = energy_from_rdms(h, v, g, G, mol.energy_nuc()) - mc.e_tot
    rec['nat_occ'] = list(np.round(np.diag(g)[A][0::2], 4))
    X = ERPACCD(h, w, g, G, C, A, V, verbose=0, max_active=max_active)
    rec['n_amp'] = int(X.mask4.sum()); rec['n_P'] = int(X.Pm.sum())
    for name, (ring, ladder) in [('pt2', (0, 0)), ('ring', (1, 0)), ('ladder', (0, 1)), ('full', (1, 1))]:
        t, E, ok, hist = X.solve(ring=ring, ladder=ladder, tol=1e-8, maxiter=300)
        rec['E_' + name] = E; rec['conv_' + name] = bool(ok); rec['it_' + name] = len(hist)
    t4a, t4full, om = X.erpa_ring_eigen()
    rec['E_erpa_ring_eigen'] = X.energy(t4a); rec['omega_min'] = float(om[0]) if len(om) else None
    results.append(rec)
    print("R=%.2f  E_CAS=%.6f  FCI-CAS=%s  NEVPT2=%.5f  pt2=%.5f ring=%.5f(%s) ladder=%.5f full=%.5f(%s) eig=%.5f  rdmchk=%.1e  nocc=%s"
          % (R, mc.e_tot, ("%.5f" % (rec['E_FCI'] - mc.e_tot)) if do_fci else 'n/a', e_pt, rec['E_pt2'], rec['E_ring'], rec['conv_ring'],
             rec['E_ladder'], rec['E_full'], rec['conv_full'], rec['E_erpa_ring_eigen'], rec['E_rdm_check'], rec['nat_occ']), flush=True)
json.dump(results, open('n2_%s_ma%d.json' % (basis.replace('-', ''), max_active), 'w'), indent=1)
