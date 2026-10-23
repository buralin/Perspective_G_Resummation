import io, re, numpy as np
from pyscf import gto, scf, mcscf, mrpt
from erpa_ccd import *
for R in [1.1, 2.0]:
    mol = gto.M(atom='N 0 0 0; N 0 0 %f' % R, basis='6-31g', unit='angstrom', verbose=0)
    mf = scf.RHF(mol).run(); mc = mcscf.CASSCF(mf, 6, 6); mc.natorb = True; mc.run()
    pt = mrpt.NEVPT(mc); pt.verbose = 4; buf = io.StringIO(); pt.stdout = buf; pt.kernel()
    nev = {}
    for line in buf.getvalue().splitlines():
        m = re.match(r"\s*(S\w+)\s+\(([^)]*)\)('?)\s*,\s*E = ([-\d.]+)", line)
        if m: nev[m.group(1) + m.group(2) + m.group(3)] = float(m.group(4))
    h, v, w = so_integrals(mol, mc.mo_coeff); g, G, C, A, V = cas_vacuum(mc)
    X = ERPACCD(h, w, g, G, C, A, V, verbose=0, max_active=1)
    t, E, ok, _ = X.solve(ring=0, ladder=0)
    isA, isV = X.isA, X.isV
    nact4 = isA[:, None, None, None].astype(int) + isA[None, :, None, None] + isA[None, None, :, None] + isA[None, None, None, :]
    nvir4 = isV[:, None, None, None].astype(int) + isV[None, :, None, None]
    e4 = 0.25 * X.B4 * t
    print("R=%.1f  quasi-boson 2nd order per class (mEh): cvcv %.3f  cvca %.3f  cvav %.3f   | SC-NEVPT2: Sijrs0 %.3f  Sijr+1 %.3f  Srsi-1 %.3f"
          % (R, 1e3*e4[nact4 == 0].sum(), 1e3*e4[(nact4 == 1) & (nvir4 == 1)].sum(), 1e3*e4[(nact4 == 1) & (nvir4 == 2)].sum(), 1e3*nev['Sijrs0'], 1e3*nev['Sijr+1'], 1e3*nev['Srsi-1']), flush=True)
