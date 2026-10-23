"""N2/6-31G dissociation on a plain RHF determinant: second order (=MP2), ring+crossed ring,
ladder and the combined non-mosaic equation, as total energies."""
import json, numpy as np, time, sys
from pyscf import gto, scf, mp
from erpa_ccd import *
Rs = [float(x) for x in sys.argv[1].split(',')]; tag = sys.argv[2]
out = []; dm0 = None
for R in Rs:
    t0 = time.time()
    mol = gto.M(atom='N 0 0 0; N 0 0 %f' % R, basis='6-31g', unit='angstrom', verbose=0)
    mf = scf.RHF(mol); mf.max_cycle = 200
    mf.kernel(dm0) if dm0 is not None else mf.kernel()
    dm0 = mf.make_rdm1()
    rec = dict(R=R, E_HF=mf.e_tot, hf_conv=bool(mf.converged), E_MP2=mp.MP2(mf).run(verbose=0).e_corr, gap=float(mf.mo_energy[mol.nelectron//2] - mf.mo_energy[mol.nelectron//2 - 1]))
    h, v, w = so_integrals(mol, mf.mo_coeff)
    g, G, C, A, V = hf_vacuum(mol.nelectron, h.shape[0])
    X = ERPACCD(h, w, g, G, C, A, V, verbose=0)
    for name, (ring, ladder) in [('pt2', (0, 0)), ('ring', (1, 0)), ('ladder', (0, 1)), ('full', (1, 1))]:
        t, E, ok, hist = X.solve(ring=ring, ladder=ladder, tol=1e-8, maxiter=300)
        rec['E_' + name] = E; rec['conv_' + name] = bool(ok)
    out.append(rec)
    print("R=%.2f E_HF=%.6f gap=%.3f  MP2=%.5f pt2=%.5f ring=%.5f(%s) ladder=%.5f(%s) full=%.5f(%s)  %.0fs"
          % (R, mf.e_tot, rec['gap'], rec['E_MP2'], rec['E_pt2'], rec['E_ring'], rec['conv_ring'], rec['E_ladder'], rec['conv_ladder'], rec['E_full'], rec['conv_full'], time.time() - t0), flush=True)
    json.dump(out, open('n2_631g_hf_part%s.json' % tag, 'w'), indent=1)
