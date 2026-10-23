import numpy as np, sys
from pyscf import gto, scf, cc
from erpa_ccd import *
import nonmosaic_ref as nm
basis = sys.argv[1] if len(sys.argv) > 1 else 'sto-3g'
Rs = [float(x) for x in sys.argv[2].split(',')] if len(sys.argv) > 2 else [0.9, 1.1, 1.3, 1.6, 2.0, 2.4, 3.0]
dm0 = None
print("basis %s: energies in Eh; n.c. = not converged" % basis)
print("  R    E_RHF      non-mosaic CCD   full CCD        CCSD(PySCF)")
for R in Rs:
    mol = gto.M(atom='N 0 0 0; N 0 0 %f' % R, basis=basis, unit='angstrom', verbose=0)
    mf = scf.RHF(mol); mf.max_cycle = 200; mf.kernel(dm0) if dm0 is not None else mf.kernel(); dm0 = mf.make_rdm1()
    h, v, w = so_integrals(mol, mf.mo_coeff); n = h.shape[0]
    eps = np.repeat(mf.mo_energy, 2); o = list(range(mol.nelectron)); vv = list(range(mol.nelectron, n))
    t1, E1, ok1, it1 = nm.solve_ccd(w, eps, o, vv, mosaic=False)
    t2, E2, ok2, it2 = nm.solve_ccd(w, eps, o, vv, mosaic=True)
    try:
        mycc = cc.CCSD(mf); mycc.max_cycle = 200; mycc.verbose = 0; mycc.kernel(); E3 = mycc.e_corr; ok3 = mycc.converged
    except Exception as e:
        E3, ok3 = float('nan'), False
    fmt = lambda E, ok, it: ("%.5f (%d)" % (E, it)) if ok else "n.c. (%d)" % it
    print("%5.2f %11.6f   %-16s %-16s %s" % (R, mf.e_tot, fmt(E1, ok1, it1), fmt(E2, ok2, it2), ("%.5f" % E3) if ok3 else "n.c."), flush=True)
