"""
Block Davidson eigensolver for real symmetric matrices with root following.

Root following (maximum-overlap selection): at every iteration the Ritz
vectors that are refined are not the lowest ones but those with the largest
overlap with a set of reference vectors, initially the user-supplied guess
vectors, afterwards the Ritz vectors followed in the previous iteration.
This targets interior eigenvalues (e.g. a valence ionization or an
attachment peak of a Green's-function matrix) without computing all
eigenpairs below them.
"""
import numpy as np


def _orthonormal_columns(V, tol=1e-10):
    Q, R = np.linalg.qr(V)
    keep = np.abs(np.diag(R)) > tol
    return Q[:, keep]


def _select_roots(theta, X, ref, mode, target_energies):
    """Indices of the Ritz pairs to follow."""
    nsub = theta.size
    idx = []
    if mode == 'overlap':
        ov = np.abs(ref.T @ X)                       # (m, nsub)
        for j in range(ov.shape[0]):
            row = ov[j].copy()
            row[idx] = -1.0
            idx.append(int(np.argmax(row)))
    elif mode == 'energy':
        for t in target_energies:
            dist = np.abs(theta - t)
            dist[idx] = np.inf
            idx.append(int(np.argmin(dist)))
    elif mode == 'lowest':
        idx = list(range(min(ref.shape[1], nsub)))
    else:
        raise ValueError(f"unknown selection mode {mode!r}")
    return np.array(idx, dtype=int)


def davidson_root_following(matvec, diag, guess, tol=1e-6, max_iter=200,
                            max_space=None, mode='overlap', target_energies=None,
                            precond_floor=1e-3, verbose=True, unit=1.0,
                            unit_name=''):
    """Davidson iterations following the roots with maximum overlap.

    Parameters
    ----------
    matvec : callable, X (n, k) -> H X (n, k)
    diag : (n,) diagonal of H (preconditioner)
    guess : (n, m) initial vectors; one root is followed per column
    tol : convergence threshold on the residual norm
    mode : 'overlap' (root following), 'energy' (closest to target_energies)
           or 'lowest'
    precond_floor : floor for |diag - theta| in the diagonal preconditioner
    unit, unit_name : scale factor and name used only for printing

    Returns
    -------
    theta : (m,) eigenvalues
    X : (n, m) eigenvectors
    info : dict with 'converged', 'iterations', 'residual_norms', 'history'
    """
    diag = np.asarray(diag, dtype=float)
    n = diag.size
    G = np.asarray(guess, dtype=float)
    if G.ndim == 1:
        G = G[:, None]
    if G.shape[0] != n:
        G = G.T
    G = G / np.linalg.norm(G, axis=0)
    m = G.shape[1]
    if mode == 'energy' and (target_energies is None or len(target_energies) != m):
        raise ValueError("mode='energy' needs one target energy per guess vector")
    if max_space is None:
        max_space = min(n, max(12 * m, 40))

    V = _orthonormal_columns(G)
    if V.shape[1] < m:
        raise ValueError("guess vectors are linearly dependent")
    W = matvec(V)
    X_prev = None
    history = []
    converged = np.zeros(m, dtype=bool)
    theta_sel = np.zeros(m)
    Xs = V.copy()
    rnorm = np.full(m, np.inf)

    for it in range(1, max_iter + 1):
        Hs = V.T @ W
        Hs = 0.5 * (Hs + Hs.T)
        theta, S = np.linalg.eigh(Hs)
        X = V @ S
        HX = W @ S
        ref = G if X_prev is None else X_prev
        idx = _select_roots(theta, X, ref, mode, target_energies)
        theta_sel = theta[idx]
        Xs = X[:, idx]
        R = HX[:, idx] - Xs * theta_sel[None, :]
        rnorm = np.linalg.norm(R, axis=0)
        converged = rnorm < tol
        history.append((it, V.shape[1], theta_sel.copy(), rnorm.copy()))
        if verbose:
            th = ", ".join(f"{t * unit:12.6f}" for t in theta_sel)
            print(f"  Davidson it {it:3d}  subspace {V.shape[1]:4d}  "
                  f"roots{unit_name}: {th}  max|r| = {rnorm.max():.2e}")
        if converged.all():
            break

        new = []
        for j in np.where(~converged)[0]:
            den = diag - theta_sel[j]
            small = np.abs(den) < precond_floor
            den[small] = np.where(den[small] >= 0, precond_floor, -precond_floor)
            delta = -R[:, j] / den
            for _ in range(2):
                delta = delta - V @ (V.T @ delta)
                for u in new:
                    delta = delta - u * (u @ delta)
            nrm = np.linalg.norm(delta)
            if nrm > 1e-10:
                new.append(delta / nrm)
        if not new:
            if verbose:
                print("  Davidson: no new directions, stopping")
            break
        new = np.array(new).T
        if V.shape[1] + new.shape[1] > max_space:
            V = _orthonormal_columns(np.hstack([Xs, new]))
            W = matvec(V)
        else:
            V = np.hstack([V, new])
            W = np.hstack([W, matvec(new)])
        X_prev = Xs

    info = {'converged': converged, 'iterations': it,
            'residual_norms': rnorm, 'history': history}
    return theta_sel, Xs, info
