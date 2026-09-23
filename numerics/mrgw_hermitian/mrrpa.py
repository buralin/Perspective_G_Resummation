"""
Multireference (direct) RPA screening of the residual interaction and the
spectral MR-GW self-energy of Wang, Fang, and Li (arXiv:2604.16013,
Supplemental Material Secs. S2.A--S2.D).

The zeroth-order polarizability Pi_0 of the Dyall reference contains the
four excitation channels of their Fig. 1(d):

  (1) core -> virtual            |Phi_ai>        omega = eps_a - eps_i
  (2) core -> active             (hole i) x |Xi_mu^{N_a+1}>  omega = kappa_mu+ - eps_i
  (3) active -> virtual          (particle a) x |Xi_mu^{N_a-1}>  omega = eps_a - kappa_mu-
  (4) active -> active           |Xi_mu^{N_a}>   omega = E_mu - E_0

For each channel the transition density rho^mu_{pr} = <Phi_mu|p^+ r|Phi_0>
is stored as an nso x nso matrix.  The (direct) MR-RPA matrices are

  A_{mu nu} = omega_mu delta + sum rho^mu_{pr} v^R_{pr,qs} rho^nu_{sq}
  B_{mu nu} =                  sum rho^mu_{pr} v^R_{pr,qs} rho^nu_{qs}

and the screened-interaction couplings M_{pr,I} = sum v^R_{pr,qs}
rho^mu_{qs} (X+Y)_{mu I}  [their Eq. (S43)].
"""
import numpy as np


def _solve_rpa_general(A, B, tol=1e-8):
    """RPA eigenproblem via the non-Hermitian (2n x 2n) form; used when
    A - B is not positive definite."""
    import warnings
    n = A.shape[0]
    Mbig = np.block([[A, B], [-B, -A]])
    w, V = np.linalg.eig(Mbig)
    X, Y = V[:n], V[n:]
    norm = np.sum(X * X, axis=0) - np.sum(Y * Y, axis=0)
    sel = (w.real > 0) & (np.abs(w.imag) < tol * max(1.0, np.abs(w.real).max())) & (norm.real > 0)
    if np.any(np.abs(w.imag) > tol * max(1.0, np.abs(w.real).max())):
        warnings.warn("RPA: complex eigenvalues encountered (instability); "
                      "only real positive-norm solutions are kept")
    if np.sum(sel) != n:
        warnings.warn(f"RPA: {np.sum(sel)} positive-norm solutions found for n = {n}")
    w, X, Y, norm = w[sel].real, X[:, sel].real, Y[:, sel].real, norm[sel].real
    order = np.argsort(w)
    w, X, Y, norm = w[order], X[:, order], Y[:, order], norm[order]
    X = X / np.sqrt(norm)[None, :]
    Y = Y / np.sqrt(norm)[None, :]
    return w, X + Y, X - Y


def solve_rpa(A, B):
    """Solve the RPA eigenproblem for real symmetric A, B.

    Returns Omega (>0), R = X + Y and S = X - Y with R^T S = 1.  The
    symmetric (A-B)^{1/2}(A+B)(A-B)^{1/2} route is used when A - B is
    positive definite, otherwise the general non-Hermitian solver.
    """
    if A.shape[0] == 0:
        return np.zeros(0), np.zeros((0, 0)), np.zeros((0, 0))
    ApB = A + B
    AmB = A - B
    try:
        L = np.linalg.cholesky(AmB)
    except np.linalg.LinAlgError:
        return _solve_rpa_general(A, B)
    Mmat = L.T @ ApB @ L
    Mmat = 0.5 * (Mmat + Mmat.T)
    w2, Z = np.linalg.eigh(Mmat)
    if w2.size and w2.min() <= 0.0:
        raise RuntimeError(f"MR-RPA: non-positive Omega^2 = {w2.min():.3e}")
    Omega = np.sqrt(w2)
    R = (L @ Z) / np.sqrt(Omega)[None, :]
    S = (ApB @ R) / Omega[None, :]
    return Omega, R, S


class MRRPA:
    """MR-RPA screening for a DyallReference and the MR-GW pole bath."""

    def __init__(self, ref, tol_channel=1e-10, verbose=True):
        self.ref = ref
        self.tol_channel = tol_channel
        self.verbose = verbose
        self._build_channels()
        self._solve()
        if verbose:
            self.summary()

    # ------------------------------------------------------------------
    def _build_channels(self):
        ref = self.ref
        nso, eps = ref.nso, ref.eps
        act = ref.act_so
        omegas, rhos, labels = [], [], []

        def add(omega, rho, label):
            if np.linalg.norm(rho) < self.tol_channel:
                return
            omegas.append(float(omega))
            rhos.append(rho)
            labels.append(label)

        # (1) core -> virtual (spin conserving)
        for i in ref.core_so:
            for a in ref.virt_so:
                if (i % 2) != (a % 2):
                    continue
                rho = np.zeros((nso, nso))
                rho[a, i] = 1.0
                add(eps[a] - eps[i], rho, ('cv', i, a))
        # (2) core -> active: core hole i + active N_a+1 state
        for i in ref.core_so:
            for ia, p in enumerate(ref.poles):
                if p.sign < 0:
                    continue
                rho = np.zeros((nso, nso))
                rho[act, i] = p.d
                add(p.kappa - eps[i], rho, ('ca', i, ia))
        # (3) active -> virtual: virtual particle a + active N_a-1 state
        for a in ref.virt_so:
            for ia, p in enumerate(ref.poles):
                if p.sign > 0:
                    continue
                rho = np.zeros((nso, nso))
                rho[a, act] = p.d
                add(eps[a] - p.kappa, rho, ('av', a, ia))
        # (4) active -> active
        om, tdm = ref.neutral_transition_densities()
        for k in range(om.size):
            rho = np.zeros((nso, nso))
            rho[np.ix_(act, act)] = tdm[k]
            add(om[k], rho, ('aa', k + 1))

        self.omega0 = np.array(omegas)
        self.rho = np.array(rhos) if rhos else np.zeros((0, nso, nso))
        self.labels = labels
        self.K = len(omegas)

    # ------------------------------------------------------------------
    def _solve(self):
        ref = self.ref
        VR = ref.VR
        rho = self.rho
        if self.K == 0:
            self.A = self.B = np.zeros((0, 0))
            self.Omega = np.zeros(0)
            self.R = self.S = np.zeros((0, 0))
            self.M = np.zeros((ref.nso, ref.nso, 0))
            self.wc0 = np.zeros_like(VR)
            self.W0 = VR.copy()
            self.nmodes_lost = 0
            return
        nso, K = ref.nso, self.K
        VRm = VR.reshape(nso * nso, nso * nso)              # rows (p,r), cols (q,s)
        rho_f = rho.reshape(K, nso * nso)
        rhoT_f = np.ascontiguousarray(rho.transpose(0, 2, 1)).reshape(K, nso * nso)
        Vrho = (VRm @ rho_f.T).T                            # sum_qs v_{pr,qs} rho^nu_{qs}
        VrhoT = (VRm @ rhoT_f.T).T                          # sum_qs v_{pr,qs} rho^nu_{sq}
        B = rho_f @ Vrho.T
        A = np.diag(self.omega0) + rho_f @ VrhoT.T
        self.asymmetry = max(np.abs(A - A.T).max(), np.abs(B - B.T).max())
        A = 0.5 * (A + A.T)
        B = 0.5 * (B + B.T)
        self.A, self.B = A, B
        self.Omega, self.R, self.S = solve_rpa(A, B)
        self.nmodes_lost = K - self.Omega.size          # > 0 only for an unstable RPA
        self.norm_error = np.abs(self.R.T @ self.S - np.eye(self.Omega.size)).max()
        rhobar = (rho_f.T @ self.R).T                       # (nmodes, nso^2): sum_mu rho^mu R_{mu I}
        Mf = VRm @ rhobar.T                                 # (nso^2, nmodes)
        self.M = Mf.reshape(nso, nso, self.Omega.size)      # M_{pr,I}
        self.mode_norm = np.linalg.norm(Mf, axis=0)         # zero for decoupled (spin-flip) modes
        # static correlation part of the screened residual interaction
        self.wc0 = (-2.0 * (Mf / self.Omega[None, :]) @ Mf.T).reshape(nso, nso, nso, nso)
        self.W0 = VR + self.wc0                             # W_D(omega = 0)

    # ------------------------------------------------------------------
    def bath(self, tol=1e-12):
        """Pole energies E_GW and amplitudes A (nso x nB) of the retarded
        MR-GW correlation self-energy  Sigma_c(z) = A (z - E_GW)^{-1} A^T
        [Wang, Fang, Li, Eq. (S52)].  Poles with vanishing amplitude
        (relative norm below tol) are dropped."""
        ref, M, Om = self.ref, self.M, self.Omega
        nso = ref.nso
        if Om.size == 0:
            return np.zeros(0), np.zeros((nso, 0))
        keep = self.mode_norm > tol * max(self.mode_norm.max(), 1e-300)
        Mk, Omk = M[:, :, keep], Om[keep]
        act, core, virt = ref.act_so, ref.core_so, ref.virt_so
        att = np.where(ref.pole_sign > 0)[0]
        rem = np.where(ref.pole_sign < 0)[0]
        E_parts, A_parts = [], []
        if virt:
            E_parts.append((ref.eps[virt][:, None] + Omk[None, :]).ravel())
            A_parts.append(Mk[virt, :, :].transpose(1, 0, 2).reshape(nso, -1))
        if att.size:
            d_att = ref.d_act[:, att].T                                   # (n_att, nact_so)
            E_parts.append((ref.kappa[att][:, None] + Omk[None, :]).ravel())
            A_parts.append(np.einsum('ax,xpI->paI', d_att, Mk[act, :, :]).reshape(nso, -1))
        if core:
            E_parts.append((ref.eps[core][:, None] - Omk[None, :]).ravel())
            A_parts.append(Mk[:, core, :].reshape(nso, -1))
        if rem.size:
            d_rem = ref.d_act[:, rem].T
            E_parts.append((ref.kappa[rem][:, None] - Omk[None, :]).ravel())
            A_parts.append(np.einsum('ax,pxI->paI', d_rem, Mk[:, act, :]).reshape(nso, -1))
        if not E_parts:
            return np.zeros(0), np.zeros((nso, 0))
        E = np.concatenate(E_parts)
        A = np.concatenate(A_parts, axis=1)
        nrm = np.linalg.norm(A, axis=0)
        keepcol = nrm > tol * max(nrm.max(), 1e-300)
        return E[keepcol], np.ascontiguousarray(A[:, keepcol])

    def self_energy(self, z, static=True):
        """Retarded MR-GW self-energy Sigma_stat + A (z - E)^{-1} A^T."""
        E, A = self.bath()
        Sig = A @ (A.T / (z - E)[:, None])
        if static:
            Sig = Sig + self.ref.Sigma_stat
        return Sig

    # ------------------------------------------------------------------
    def summary(self):
        from collections import Counter
        cnt = Counter(l[0] for l in self.labels)
        print("MR-RPA")
        print(f"  channels: {self.K} (cv {cnt.get('cv', 0)}, ca {cnt.get('ca', 0)}, "
              f"av {cnt.get('av', 0)}, aa {cnt.get('aa', 0)})")
        if self.K:
            from .dyall import HARTREE2EV
            print(f"  A/B asymmetry before symmetrization: {self.asymmetry:.2e}; "
                  f"|R^T S - 1|max = {self.norm_error:.2e}")
            print(f"  lowest MR-RPA excitation energies (eV): "
                  + ", ".join(f"{w * HARTREE2EV:.3f}" for w in np.sort(self.Omega)[:5]))
            E, A = self.bath()
            ncoupled = int(np.sum(self.mode_norm > 1e-12 * self.mode_norm.max()))
            print(f"  MR-RPA modes coupled to the residual interaction: {ncoupled} of {self.K}; "
                  f"MR-GW pole bath after pruning: {E.size} poles")
