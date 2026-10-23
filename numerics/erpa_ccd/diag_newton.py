"""Does the ring+crossed-ring equation have a real solution when the fixed-point iteration fails?
Newton-Krylov on the unique-amplitude parametrization, Jacobian spectrum, iteration history.  N2/STO-3G RHF."""
import numpy as np, sys, time
from scipy.optimize import newton_krylov
from scipy.sparse.linalg import LinearOperator, eigs
from pyscf import gto, scf
from erpa_ccd import *
R = float(sys.argv[1]) if len(sys.argv) > 1 else 1.1
basis = sys.argv[2] if len(sys.argv) > 2 else 'sto-3g'
mol = gto.M(atom='N 0 0 0; N 0 0 %f' % R, basis=basis, unit='angstrom', verbose=0)
mf = scf.RHF(mol).run()
h, v, w = so_integrals(mol, mf.mo_coeff); n = h.shape[0]
g, G, C, A, V = hf_vacuum(mol.nelectron, n)
X = ERPACCD(h, w, g, G, C, A, V, verbose=0)
m = X.mask4
U = np.argwhere((m > 0) & (np.arange(n)[:, None, None, None] < np.arange(n)[None, :, None, None]) & (np.arange(n)[None, None, :, None] < np.arange(n)[None, None, None, :]))
print("R=%.2f %s: unique amplitudes %d" % (R, basis, len(U)))
def expand(u):
    t = np.zeros((n,) * 4); t[U[:, 0], U[:, 1], U[:, 2], U[:, 3]] = u
    t = t - t.transpose(1, 0, 2, 3); t = t - t.transpose(0, 1, 3, 2)
    return t
def F(u, ring, ladder):
    Rr = X.residual(expand(u), ring=ring, ladder=ladder)
    return Rr[U[:, 0], U[:, 1], U[:, 2], U[:, 3]]
D = X.denominator(); d = D[U[:, 0], U[:, 1], U[:, 2], U[:, 3]]
B4u = X.B4[U[:, 0], U[:, 1], U[:, 2], U[:, 3]]
def energy(u):
    return X.energy(expand(u))
u2 = -B4u / d                                   # second-order amplitude (= MP2 here)
print("second order E = %.6f" % energy(u2))
Minv = LinearOperator((len(U), len(U)), matvec=lambda x: x / d)
for name, (ring, ladder) in [('ring+crossed', (1, 0)), ('ladder', (0, 1)), ('full', (1, 1))]:
    # fixed-point history (first iterations)
    t, E, ok, hist = X.solve(ring=ring, ladder=ladder, tol=1e-8, maxiter=60)
    h5 = ' '.join("%.1e" % e for _, _, e in hist[:12])
    print("%-13s fixed point: conv=%s after %d, |R| history: %s ..." % (name, ok, len(hist), h5))
    # Jacobian of the preconditioned residual at the second-order amplitude
    f0 = F(u2, ring, ladder)
    def jv(x, u0=u2, f0=f0):
        eps = 1e-6 / max(1e-12, np.linalg.norm(x)) * np.sqrt(len(x))
        return (F(u0 + eps * x, ring, ladder) - f0) / eps / d
    J = LinearOperator((len(U), len(U)), matvec=jv)
    try:
        lr = eigs(J, k=3, which='LR', return_eigenvectors=False, tol=1e-6, maxiter=3000)
        sr = eigs(J, k=3, which='SR', return_eigenvectors=False, tol=1e-6, maxiter=3000)
        print("   D^-1 J at t(2): largest real parts %s ; smallest real parts %s" % (np.round(np.sort(lr.real)[::-1], 3), np.round(np.sort(sr.real), 3)))
    except Exception as e:
        print("   eigs failed:", e)
    # Newton-Krylov
    t0 = time.time()
    try:
        sol = newton_krylov(lambda u: F(u, ring, ladder), u2, method='lgmres', inner_M=Minv, f_tol=1e-9, maxiter=60, verbose=0)
        print("   Newton-Krylov: converged, E = %.6f, |R| = %.1e, max|t| = %.3f (%.0fs)" % (energy(sol), abs(F(sol, ring, ladder)).max(), abs(sol).max(), time.time() - t0))
        f0 = F(sol, ring, ladder)
        def jv2(x, u0=sol, f0=f0):
            eps = 1e-6 / max(1e-12, np.linalg.norm(x)) * np.sqrt(len(x))
            return (F(u0 + eps * x, ring, ladder) - f0) / eps / d
        J2 = LinearOperator((len(U), len(U)), matvec=jv2)
        sr = eigs(J2, k=3, which='SR', return_eigenvectors=False, tol=1e-6, maxiter=3000)
        print("   D^-1 J at solution: smallest real parts %s" % np.round(np.sort(sr.real), 3))
    except Exception as e:
        print("   Newton-Krylov failed: %s (%.0fs)" % (str(e)[:80], time.time() - t0))
