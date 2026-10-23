import json, itertools, numpy as np, sys
from pyscf import gto, scf, mcscf, fci
from class_exact import *
Rs = [0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.6, 1.8, 2.0, 2.4, 3.0]
out = []; mo_prev = None
for R in Rs:
    mol = gto.M(atom='N 0 0 0; N 0 0 %f' % R, basis='sto-3g', unit='angstrom', verbose=0)
    mf = scf.RHF(mol).run()
    mc = mcscf.CASSCF(mf, 6, 6); mc.natorb = True; mc.max_cycle_macro = 200
    if mo_prev is not None:
        mc.kernel(mcscf.project_init_guess(mc, mo_prev, prev_mol=prev_mol))
    else:
        mc.kernel()
    mo_prev, prev_mol = mc.mo_coeff, mol
    st = full_space_setup(mol, mc)
    ncore, ncas, nmo = st['ncore'], st['ncas'], st['nmo']
    C = list(range(2 * ncore)); A = list(range(2 * ncore, 2 * (ncore + ncas)))
    E0 = st['E0']
    rec = dict(R=R, E_CAS=mc.e_tot, E_FCI=fci.FCI(mf).kernel()[0])
    ops = [[('c', x), ('c', y), ('d', j), ('d', i)] for i, j in itertools.combinations(C, 2) for x, y in itertools.combinations(A, 2)]
    S, Hm, v, Vm = build_class(st, ops)
    rec['E2_caca'], rec['ECI_caca'], rec['rank_caca'] = class_energies(S, Hm, v, E0)
    ops2 = [[('c', z), ('d', w), ('c', x), ('d', i)] for i in C for x in A for z in A for w in A if z != w and z != x and w != x]
    S2, H2, v2, V2 = build_class(st, ops2)
    rec['E2_caaa'], rec['ECI_caaa'], rec['rank_caaa'] = class_energies(S2, H2, v2, E0, tol=1e-7)
    Vall = np.vstack([Vm, V2]); Sall = Vall @ Vall.T
    HV = np.array([st['Hop'](x.reshape(st['ci'].shape)).ravel() for x in Vall])
    Hall = Vall @ HV.T; vall = Vall @ st['Hop'](st['ci']).ravel()
    rec['E2_both'], rec['ECI_both'], rec['rank_both'] = class_energies(Sall, Hall, vall, E0, tol=1e-7)
    # quasi-boson vs exact norms: ratio statistics for (ca,ca)
    (d1a, d1b), _ = mc.fcisolver.make_rdm12s(mc.ci, ncas, mc.nelecas)
    n_so = np.zeros(2 * ncas); n_so[0::2] = np.diag(d1a); n_so[1::2] = np.diag(d1b)
    rec['nat_occ'] = [float(x) for x in np.diag(d1a)]
    ratios = []
    k = 0
    for i, j in itertools.combinations(C, 2):
        for x, y in itertools.combinations(A, 2):
            v0, ne = apply_string([('c', x), ('c', y), ('d', j), ('d', i)], st['ci'], nmo, st['nelec'])
            if v0 is None or ne != st['nelec'] or np.linalg.norm(v0) < 1e-10: continue
            xa, ya = x - 2 * ncore, y - 2 * ncore
            ratios.append(float(np.dot(v0.ravel(), v0.ravel()) / ((1 - n_so[xa]) * (1 - n_so[ya]))))
    rec['norm_ratio_max'] = max(ratios); rec['norm_ratio_min'] = min(ratios)
    out.append(rec)
    print("R=%.2f FCI-CAS=%.5f  (ca,ca): E2=%.5f CI=%.5f  (ca,aa): E2=%.5f CI=%.5f  both: E2=%.5f CI=%.5f  norm ratio [%.2f, %.1f]"
          % (R, rec['E_FCI'] - mc.e_tot, rec['E2_caca'], rec['ECI_caca'], rec['E2_caaa'], rec['ECI_caaa'], rec['E2_both'], rec['ECI_both'],
             rec['norm_ratio_min'], rec['norm_ratio_max']), flush=True)
json.dump(out, open('n2_class_exact.json', 'w'), indent=1)
