"""Lowest eigenvalues of the linear part (Jacobian at t=0) of the three variants, in units of the
second-order denominators, for N2/STO-3G RHF and for the CASSCF(6,6) external manifold in 6-31G."""
import numpy as np, sys
from scipy.sparse.linalg import LinearOperator, eigs
from pyscf import gto, scf, mcscf
from erpa_ccd import *
def setup_unique(X):
    n = X.n; m = X.mask4
    ar = np.arange(n)
    U = np.argwhere((m > 0) & (ar[:, None, None, None] < ar[None, :, None, None]) & (ar[None, None, :, None] < ar[None, None, None, :]))
    def expand(u):
        t = np.zeros((n,) * 4); t[U[:, 0], U[:, 1], U[:, 2], U[:, 3]] = u
        t = t - t.transpose(1, 0, 2, 3); t = t - t.transpose(0, 1, 3, 2); return t
    D = X.denominator(); d = D[U[:, 0], U[:, 1], U[:, 2], U[:, 3]]
    return U, expand, d
def linear_spectrum(X, ring, ladder, k=3):
    U, expand, d = setup_unique(X)
    f0 = X.residual(np.zeros((X.n,) * 4), ring=ring, ladder=ladder)[U[:, 0], U[:, 1], U[:, 2], U[:, 3]]
    def mv(x):
        eps = 1e-5 / max(1e-12, np.linalg.norm(x)) * np.sqrt(len(x))
        f = X.residual(expand(eps * x), ring=ring, ladder=ladder)[U[:, 0], U[:, 1], U[:, 2], U[:, 3]]
        return (f - f0) / eps / d
    J = LinearOperator((len(U), len(U)), matvec=mv)
    sr = eigs(J, k=k, which='SR', return_eigenvectors=False, tol=1e-6, maxiter=5000)
    return np.sort(sr.real)
mode = sys.argv[1] if len(sys.argv) > 1 else 'hf'
for R in [0.9, 1.1, 1.3, 1.6]:
    mol = gto.M(atom='N 0 0 0; N 0 0 %f' % R, basis='sto-3g' if mode == 'hf' else '6-31g', unit='angstrom', verbose=0)
    mf = scf.RHF(mol).run()
    if mode == 'hf':
        h, v, w = so_integrals(mol, mf.mo_coeff); g, G, C, A, V = hf_vacuum(mol.nelectron, h.shape[0])
        X = ERPACCD(h, w, g, G, C, A, V, verbose=0)
    else:
        mc = mcscf.CASSCF(mf, 6, 6); mc.natorb = True; mc.run()
        h, v, w = so_integrals(mol, mc.mo_coeff); g, G, C, A, V = cas_vacuum(mc)
        X = ERPACCD(h, w, g, G, C, A, V, verbose=0, max_active=1)
    out = []
    for name, (ring, ladder) in [('2nd', (0, 0)), ('ring+cr', (1, 0)), ('ladder', (0, 1)), ('full', (1, 1))]:
        try:
            out.append("%s: %s" % (name, np.round(linear_spectrum(X, ring, ladder), 3)))
        except Exception as e:
            out.append("%s: failed" % name)
    print("R=%.1f [%s]  lowest eigenvalues of D^-1 J(t=0):  %s" % (R, mode, ' | '.join(out)), flush=True)
