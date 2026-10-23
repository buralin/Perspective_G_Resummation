"""N2/STO-3G with CAS(2,2): exact reference (brute force) for the classes with <=1 active index
vs the quasi-boson extended channel equations restricted to the same classes."""
import json, itertools, io, re, numpy as np
from pyscf import gto, scf, mcscf, fci, mrpt
from class_exact import *
from erpa_ccd import *
out = []
mo_prev = None
for R in [0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.6, 1.8, 2.0]:
    mol = gto.M(atom='N 0 0 0; N 0 0 %f' % R, basis='sto-3g', unit='angstrom', verbose=0)
    mf = scf.RHF(mol).run()
    mc = mcscf.CASSCF(mf, 2, 2); mc.natorb = True; mc.max_cycle_macro = 200
    if mo_prev is None:
        mc.kernel(mcscf.sort_mo(mc, mf.mo_coeff, [7, 10]))        # 3sigma_g, 3sigma_u (1-based)
    else:
        mc.kernel(mcscf.project_init_guess(mc, mo_prev, prev_mol=prev_mol))
    mo_prev, prev_mol = mc.mo_coeff, mol
    (d1a, d1b), _ = mc.fcisolver.make_rdm12s(mc.ci, mc.ncas, mc.nelecas)
    print('R=%.2f CASSCF(2,2) E=%.6f conv=%s nat occ %s' % (R, mc.e_tot, mc.converged, np.round(np.diag(d1a) * 2, 4)))
    st = full_space_setup(mol, mc)
    ncore, ncas, nmo = st['ncore'], st['ncas'], st['nmo']
    C = list(range(2 * ncore)); A = list(range(2 * ncore, 2 * (ncore + ncas))); V = list(range(2 * (ncore + ncas), 2 * nmo))
    E0 = st['E0']
    rec = dict(R=R, E_CAS=mc.e_tot, E_FCI=fci.FCI(mf).kernel()[0])
    pt = mrpt.NEVPT(mc); pt.verbose = 4; buf = io.StringIO(); pt.stdout = buf; rec['E_NEVPT2'] = pt.kernel()
    for line in buf.getvalue().splitlines():
        m = re.match(r"\s*(S\w+)\s+\(([^)]*)\)('?)\s*,\s*E = ([-\d.]+)", line)
        if m: rec['nev_' + m.group(1) + m.group(2) + m.group(3)] = float(m.group(4))
    classes = {
        'cvcv': [[('c', a), ('c', b), ('d', j), ('d', i)] for i, j in itertools.combinations(C, 2) for a, b in itertools.combinations(V, 2)],
        'cvca': [[('c', a), ('c', y), ('d', j), ('d', i)] for i, j in itertools.combinations(C, 2) for a in V for y in A],
        'cvav': [[('c', a), ('c', b), ('d', y), ('d', i)] for i in C for y in A for a, b in itertools.combinations(V, 2)],
    }
    Vs = []
    for name, ops in classes.items():
        S, Hm, v, Vm = build_class(st, ops)
        rec['E2_' + name], rec['ECI_' + name], rec['rank_' + name] = class_energies(S, Hm, v, E0)
        Vs.append(Vm)
    Vall = np.vstack(Vs); Sall = Vall @ Vall.T
    HV = np.array([st['Hop'](x.reshape(st['ci'].shape)).ravel() for x in Vall])
    Hall = Vall @ HV.T; vall = Vall @ st['Hop'](st['ci']).ravel()
    rec['E2_all3'], rec['ECI_all3'], rec['rank_all3'] = class_energies(Sall, Hall, vall, E0)
    # quasi-boson extended equations restricted to the same classes
    h, v, w = so_integrals(mol, mc.mo_coeff)
    g, G, C2, A2, V2 = cas_vacuum(mc)
    X = ERPACCD(h, w, g, G, C2, A2, V2, verbose=0, max_active=1)
    for nm, (ring, ladder) in [('pt2', (0, 0)), ('ring', (1, 0)), ('ladder', (0, 1)), ('full', (1, 1))]:
        t, E, ok, hist = X.solve(ring=ring, ladder=ladder, tol=1e-9, maxiter=300)
        rec['E_' + nm] = E; rec['conv_' + nm] = bool(ok)
    # class-resolved quasi-boson second order: energy contributions per class
    t, E, ok, _ = X.solve(ring=0, ladder=0)
    isA = X.isA; nact4 = isA[:, None, None, None].astype(int) + isA[None, :, None, None] + isA[None, None, :, None] + isA[None, None, None, :]
    isV = X.isV; nvir4 = isV[:, None, None, None].astype(int) + isV[None, :, None, None]
    e4 = 0.25 * X.B4 * t
    rec['E_pt2_cvcv'] = float(e4[(nact4 == 0)].sum()); rec['E_pt2_cvca'] = float(e4[(nact4 == 1) & (nvir4 == 1)].sum()); rec['E_pt2_cvav'] = float(e4[(nact4 == 1) & (nvir4 == 2)].sum())
    out.append(rec)
    print("R=%.2f FCI-CAS=%.5f NEVPT2=%.5f [Sijrs0 %.5f Sijr+1 %.5f Srsi-1 %.5f]" % (R, rec['E_FCI'] - mc.e_tot, rec['E_NEVPT2'], rec['nev_Sijrs0'], rec['nev_Sijr+1'], rec['nev_Srsi-1']))
    print("   exact-metric: E2 cvcv %.5f cvca %.5f cvav %.5f | all3 E2 %.5f CI %.5f" % (rec['E2_cvcv'], rec['E2_cvca'], rec['E2_cvav'], rec['E2_all3'], rec['ECI_all3']))
    print("   quasi-boson : pt2 %.5f [cvcv %.5f cvca %.5f cvav %.5f]  ring %.5f(%s) ladder %.5f(%s) full %.5f(%s)"
          % (rec['E_pt2'], rec['E_pt2_cvcv'], rec['E_pt2_cvca'], rec['E_pt2_cvav'], rec['E_ring'], rec['conv_ring'], rec['E_ladder'], rec['conv_ladder'], rec['E_full'], rec['conv_full']), flush=True)
json.dump(out, open('n2_cas22.json', 'w'), indent=1)
