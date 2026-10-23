"""Effect of the active-space size on the external channel equations (N2)."""
import sys, io, re, time, json, numpy as np
from pyscf import gto, scf, mcscf, mrpt, fci
from erpa_ccd import *
basis = sys.argv[1]; Rs = [float(x) for x in sys.argv[2].split(',')]; spaces = [tuple(int(y) for y in s.split('/')) for s in sys.argv[3].split(',')]
out = []
for R in Rs:
    mol = gto.M(atom='N 0 0 0; N 0 0 %f' % R, basis=basis, unit='angstrom', verbose=0)
    mf = scf.RHF(mol).run()
    efci = fci.FCI(mf).kernel()[0] if basis == 'sto-3g' else None
    for (nel, ncas) in spaces:
        t0 = time.time()
        mc = mcscf.CASSCF(mf, ncas, nel); mc.natorb = True; mc.max_cycle_macro = 200; mc.run()
        pt = mrpt.NEVPT(mc); pt.verbose = 4; buf = io.StringIO(); pt.stdout = buf; e_pt = pt.kernel()
        nev = {}
        for line in buf.getvalue().splitlines():
            m = re.match(r"\s*(S\w+)\s+\(([^)]*)\)('?)\s*,\s*E = ([-\d.]+)", line)
            if m: nev[m.group(1) + m.group(2) + m.group(3)] = float(m.group(4))
        h, v, w = so_integrals(mol, mc.mo_coeff); g, G, C, A, V = cas_vacuum(mc)
        rec = dict(R=R, nel=nel, ncas=ncas, E_HF=mf.e_tot, E_CAS=mc.e_tot, E_FCI=efci, E_NEVPT2=e_pt, nev=nev,
                   ncore=len(C) // 2, nvirt=len(V) // 2, nat_occ=[float(x) for x in np.diag(g)[A][0::2]])
        X = ERPACCD(h, w, g, G, C, A, V, verbose=0, max_active=1)
        rec['n_amp1'] = int(X.mask4.sum())
        for name, (ring, ladder) in [('pt2', (0, 0)), ('ring', (1, 0)), ('ladder', (0, 1)), ('full', (1, 1))]:
            t, E, ok, hist = X.solve(ring=ring, ladder=ladder, tol=1e-8, maxiter=300)
            rec['E_' + name] = E; rec['conv_' + name] = bool(ok)
        t4a, t4full, om = X.erpa_ring_eigen(); rec['omega_min_ext'] = float(om[0]) if len(om) else None
        # quasi-boson second order with the two-active-index classes included
        X4 = ERPACCD(h, w, g, G, C, A, V, verbose=0, max_active=4)
        rec['n_amp4'] = int(X4.mask4.sum())
        t, E, ok, hist = X4.solve(ring=0, ladder=0, tol=1e-8, maxiter=300); rec['E_pt2_all'] = E; rec['conv_pt2_all'] = bool(ok)
        # norm ratios: (av,av): Gamma_xy,xy / (n_x n_y);  (ca,ca): (1-n_x-n_y+Gamma_xy,xy)/((1-n_x)(1-n_y))
        nA = np.diag(g)[A]; Gd = np.einsum('xyxy->xy', G[np.ix_(A, A, A, A)])
        iu = np.triu_indices(len(A), 1)
        r_av = (Gd / np.outer(nA, nA))[iu]; r_ca = ((1 - nA[:, None] - nA[None, :] + Gd) / np.outer(1 - nA, 1 - nA))[iu]
        ok_av = np.isfinite(r_av); ok_ca = np.isfinite(r_ca)
        rec['ratio_avav'] = [float(np.min(r_av[ok_av])), float(np.max(r_av[ok_av]))]; rec['ratio_caca'] = [float(np.min(r_ca[ok_ca])), float(np.max(r_ca[ok_ca]))]
        out.append(rec)
        one = nev.get('Sijrs0', 0) + nev.get('Sijr+1', 0) + nev.get('Srsi-1', 0)
        two = nev.get('Sij+2', 0) + nev.get('Srs-2', 0); semi = nev.get("Si+1'", 0) + nev.get("Sr-1'", 0) + nev.get("Sir0'", 0)
        print("R=%.1f CAS(%d,%d): E_CAS=%.5f %s NEVPT2=%.2f mEh [1-act %.2f | 2-act %.2f | semi %.2f]  ours(1-act): pt2 %.2f ring %.2f%s ladder %.2f full %.2f%s  pt2(all classes) %.2f%s  omega_min %.3f  norm ratio av-av [%.2f,%.1f] ca-ca [%.2f,%.1f]  nocc %s  (%.0fs)"
              % (R, nel, ncas, mc.e_tot, ("FCI-CAS=%.2f" % (1e3 * (efci - mc.e_tot))) if efci else "", 1e3 * e_pt, 1e3 * one, 1e3 * two, 1e3 * semi,
                 1e3 * rec['E_pt2'], 1e3 * rec['E_ring'], '' if rec['conv_ring'] else '*', 1e3 * rec['E_ladder'], 1e3 * rec['E_full'], '' if rec['conv_full'] else '*',
                 1e3 * rec['E_pt2_all'], '' if rec['conv_pt2_all'] else '*', rec['omega_min_ext'], rec['ratio_avav'][0], rec['ratio_avav'][1], rec['ratio_caca'][0], rec['ratio_caca'][1],
                 np.round(rec['nat_occ'], 3), time.time() - t0), flush=True)
json.dump(out, open('active_space_scan_%s.json' % basis.replace('-', ''), 'w'), indent=1)
