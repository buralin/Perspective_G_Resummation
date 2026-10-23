"""RPAx/TDA stability of the RHF reference and of the external ERPA manifold on the CASSCF vacuum,
N2/6-31G along the curve."""
import numpy as np, json
from pyscf import gto, scf, mcscf
from erpa_ccd import *
def ph_matrices(X):
    pairs = np.argwhere(X.E)
    A = np.array([[X.Aph[p, q, r, s] for (r, s) in pairs] for (p, q) in pairs])
    B = np.array([[X.Bph[p, q, r, s] for (r, s) in pairs] for (p, q) in pairs])
    N = np.array([X.Nph[p, q] for (p, q) in pairs])
    sq = np.sqrt(N)
    return A / sq[:, None] / sq[None, :], B / sq[:, None] / sq[None, :]
def analyze(A, B):
    eAmB = np.linalg.eigvalsh(A - B); eApB = np.linalg.eigvalsh(A + B); eA = np.linalg.eigvalsh(A)
    w2 = np.linalg.eigvals((A - B) @ (A + B))
    wmin = np.sqrt(w2.astype(complex)); i = np.argmin(wmin.real)
    return eA.min(), eAmB.min(), eApB.min(), wmin[np.argmin(np.abs(wmin))], int((w2.real < 0).sum())
print("%5s | RHF: minTDA  min(A-B)  min(A+B)  lowest omega  #omega^2<0 | CASSCF(6,6) external: minTDA  min(A-B)  min(A+B)  lowest omega" % 'R')
dm0 = None; mo_prev = None
for R in [0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.6, 1.8, 2.0, 2.4, 3.0]:
    mol = gto.M(atom='N 0 0 0; N 0 0 %f' % R, basis='6-31g', unit='angstrom', verbose=0)
    mf = scf.RHF(mol); mf.max_cycle = 200; mf.kernel(dm0) if dm0 is not None else mf.kernel(); dm0 = mf.make_rdm1()
    h, v, w = so_integrals(mol, mf.mo_coeff); g, G, C, A_, V = hf_vacuum(mol.nelectron, h.shape[0])
    X = ERPACCD(h, w, g, G, C, A_, V, verbose=0)
    hf_res = analyze(*ph_matrices(X))
    mc = mcscf.CASSCF(mf, 6, 6); mc.natorb = True; mc.max_cycle_macro = 200
    mc.kernel(mcscf.project_init_guess(mc, mo_prev, prev_mol=prev_mol)) if mo_prev is not None else mc.kernel()
    mo_prev, prev_mol = mc.mo_coeff, mol
    h, v, w = so_integrals(mol, mc.mo_coeff); g, G, C, A_, V = cas_vacuum(mc)
    Xc = ERPACCD(h, w, g, G, C, A_, V, verbose=0, max_active=1)
    cas_res = analyze(*ph_matrices(Xc))
    print("%5.2f | %8.4f %8.4f %8.4f  %s  %d | %8.4f %8.4f %8.4f  %s" % (R, hf_res[0], hf_res[1], hf_res[2], ("%.4f" % hf_res[3].real) if abs(hf_res[3].imag) < 1e-8 else ("%.4f%+.4fi" % (hf_res[3].real, hf_res[3].imag)), hf_res[4],
          cas_res[0], cas_res[1], cas_res[2], ("%.4f" % cas_res[3].real) if abs(cas_res[3].imag) < 1e-8 else ("%.4f%+.4fi" % (cas_res[3].real, cas_res[3].imag))), flush=True)
