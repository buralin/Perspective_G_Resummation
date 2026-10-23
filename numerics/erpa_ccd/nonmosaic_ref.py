"""Independent spin-orbital non-mosaic CCD residual (Pa2CCD Eq. nonmosaic_ccd) for the HF limit."""
import numpy as np
def residual(t, w, eps, o, v):
    # t[a,b,i,j] over virtual a,b and occupied i,j (arrays over the full so space, zero elsewhere)
    woovv = w; n = w.shape[0]
    R = w.copy()                                                      # <ab||ij> at [a,b,i,j]
    D = np.zeros((n,n,n,n))
    e = eps
    D = e[:, None, None, None] + e[None, :, None, None] - e[None, None, :, None] - e[None, None, None, :]
    R = R + D * t
    R = R + 0.5 * np.einsum('abcd,cdij->abij', w, t, optimize=True)
    R = R + 0.5 * np.einsum('klij,abkl->abij', w, t, optimize=True)
    X = np.einsum('kbcj,acik->abij', w, t, optimize=True)
    R = R + X - X.transpose(0, 1, 3, 2) - X.transpose(1, 0, 2, 3) + X.transpose(1, 0, 3, 2)
    R = R + 0.25 * np.einsum('klcd,cdij,abkl->abij', w, t, t, optimize=True)
    Y = 0.5 * np.einsum('klcd,acik,bdjl->abij', w, t, t, optimize=True)
    R = R + Y - Y.transpose(0, 1, 3, 2) - Y.transpose(1, 0, 2, 3) + Y.transpose(1, 0, 3, 2)
    mask = np.zeros((n,n,n,n)); mask[np.ix_(v, v, o, o)] = 1.0
    return R * mask

def mosaic_terms(t, w, o, v):
    """The two CCD quadratic terms that belong to no single channel (one-body dressings):
    -1/2 P(ab) sum <kl||cd> t_ij^ac t_kl^bd  - 1/2 P(ij) sum <kl||cd> t_ik^ab t_jl^cd."""
    n = w.shape[0]
    Xv = 0.5 * np.einsum('klcd,bdkl->cb', w, t, optimize=True)        # chi_cb
    M1 = -np.einsum('cb,acij->abij', Xv, t, optimize=True)
    Xo = 0.5 * np.einsum('klcd,cdjl->kj', w, t, optimize=True)        # chi_kj
    M2 = -np.einsum('kj,abik->abij', Xo, t, optimize=True)
    R = (M1 - M1.transpose(1, 0, 2, 3)) + (M2 - M2.transpose(0, 1, 3, 2))
    mask = np.zeros((n, n, n, n)); mask[np.ix_(v, v, o, o)] = 1.0
    return R * mask

def solve_ccd(w, eps, o, v, mosaic=True, nonmosaic=True, tol=1e-8, maxiter=300):
    """Fixed-point + DIIS solution of CCD (full or non-mosaic) in the HF limit."""
    from pyscf import lib
    n = w.shape[0]
    mask = np.zeros((n, n, n, n)); mask[np.ix_(v, v, o, o)] = 1.0
    D = eps[:, None, None, None] + eps[None, :, None, None] - eps[None, None, :, None] - eps[None, None, None, :]
    Dsafe = np.where(mask > 0, D, 1.0)
    t = -w * mask / Dsafe
    adiis = lib.diis.DIIS(); adiis.space = 8
    for it in range(maxiter):
        R = residual(t, w, eps, o, v) if nonmosaic else (w * mask + D * t)
        if mosaic:
            R = R + mosaic_terms(t, w, o, v)
        err = abs(R).max()
        E = 0.25 * np.einsum('abij,abij', w * mask, t)
        if err < tol:
            return t, E, True, it
        tnew = t - R / Dsafe * mask
        if not np.isfinite(tnew).all() or abs(tnew).max() > 1e3:
            return t, float('nan'), False, it
        try:
            t = adiis.update(tnew)
        except Exception:
            return tnew, float('nan'), False, it
    return t, E, False, maxiter
