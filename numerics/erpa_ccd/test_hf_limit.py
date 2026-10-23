import numpy as np
from pyscf import gto, scf, cc
from erpa_ccd import *
import nonmosaic_ref
mol = gto.M(atom='N 0 0 0; N 0 0 1.1', basis='sto-3g', unit='angstrom', verbose=0)
mf = scf.RHF(mol).run()
h, v, w = so_integrals(mol, mf.mo_coeff)
n = h.shape[0]; nocc = mol.nelectron
g, G, C, A, V = hf_vacuum(nocc, n)
print("E_HF from RDMs - E_HF:", energy_from_rdms(h, v, g, G, mol.energy_nuc()) - mf.e_tot)
X = ERPACCD(h, w, g, G, C, A, V, verbose=1)
eps = np.repeat(mf.mo_energy, 2)
# check channel matrices against HF formulas
o, vv = C, V
Aref = np.zeros((n,n,n,n)); Bref = np.zeros((n,n,n,n))
for a in vv:
    for i in o:
        for b in vv:
            for j in o:
                Aref[a,i,b,j] = (eps[a]-eps[i])*(a==b)*(i==j) + w[a,j,i,b]
                Bref[a,i,b,j] = w[a,b,i,j]
print("A vs HF RPA A:", abs(X.Aph - Aref).max(), "  B vs <ab||ij>:", abs(X.Bph - Bref).max(), "  N=1:", abs(X.Nph[X.E]-1).max())
# random antisymmetric amplitude in the amplitude space
rng = np.random.default_rng(0)
t = rng.normal(size=(n,n,n,n)) * X.mask4
t = t - t.transpose(1,0,2,3); t = t - t.transpose(0,1,3,2); t *= 0.05
R = X.residual(t)
Rref = nonmosaic_ref.residual(t, w, eps, o, vv)
print("combined residual vs non-mosaic CCD residual (random t):", abs(R - Rref).max(), " |Rref| scale", abs(Rref).max())
R2, pcs = X.residual(t, pieces=True)
print("  antisymmetry of residual: ", abs(R + R.transpose(1,0,2,3)).max(), abs(R + R.transpose(0,1,3,2)).max())
# solve
for name, (ring, ladder) in [('second order', (0,0)), ('ring+crossed', (1,0)), ('ladder', (0,1)), ('non-mosaic CCD', (1,1))]:
    tt, E, ok, hist = X.solve(ring=ring, ladder=ladder)
    print("%-16s E_c = %.8f  converged=%s iters=%d" % (name, E, ok, len(hist)))
mycc = cc.CCSD(mf).run(verbose=0)
print("PySCF CCSD E_c = %.8f (for orientation; includes singles and mosaic terms)" % mycc.e_corr)
from pyscf import mp
print("MP2 E_c = %.8f" % mp.MP2(mf).run(verbose=0).e_corr)
t4a, t4full, om = X.erpa_ring_eigen()
print("pure RPAx (eigen) : E = 1/4 Tr(B t) = %.8f ; plasmon 1/2 Tr(B t) = %.8f ; lowest omega %.4f" % (X.energy(t4a), 2*X.energy(t4a), om[0]))
