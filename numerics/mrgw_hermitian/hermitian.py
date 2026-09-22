"""
Common Hermitian pole-space matrix [Eq. (7) of notes/comparison.tex]

        | K_top      T_D^+ A |                 K_top = K_D + T_D^+ Sigma_stat T_D + K_B
    H = |                    |  ,
        | A^+ T_D    E_GW    |

    G(z) = (T_D  0) (z - H)^{-1} (T_D  0)^+ ,   z = omega + i eta.

The "top" space consists of the inactive orbital poles (t_P = e_P,
kappa_P = eps_P) and the active charged poles (d_alpha, kappa_alpha) of the
Dyall Green's function; the bath consists of the MR-GW self-energy poles.
K_B carries the (bare or statically screened) mixed inactive--active
couplings

    [K_B]_{P alpha} = sum_{zyw} I_{Pz,yw} C^{D,+/-}_{alpha; z, yw},
    I = vbar^R  (bare)   or   I = v^R - [W_D(0)]^x  (parent-consistent static kernel).

Setting K_B = 0 recovers the MR-GW Dyson equation exactly; setting A = 0 and
using the bare kernel recovers the block-matrix Green's function of the
preliminary section of notes/mr-hedin.tex.
"""
import numpy as np


class HermitianGF:
    """Hermitian pole-space representation of the retarded Green's function.

    Parameters
    ----------
    ref : DyallReference
    rpa : MRRPA or None (required for bath=True or mixed='screened')
    mixed : 'none' | 'bare' | 'screened'
    bath : include the MR-GW pole bath
    cross_sector : keep couplings between core poles and attachment poles and
        between virtual poles and removal poles (Objective II); False keeps
        only same-sector couplings (superoperator picture with the
        zeroth-order reference).
    include_static : include T_D^+ Sigma_1^{11} T_D
    """

    def __init__(self, ref, rpa=None, mixed='screened', bath=True,
                 cross_sector=True, include_static=True, bath_hamiltonian='diagonal',
                 label=None):
        if mixed not in ('none', 'bare', 'screened'):
            raise ValueError("mixed must be 'none', 'bare' or 'screened'")
        if bath_hamiltonian not in ('diagonal', 'top'):
            raise ValueError("bath_hamiltonian must be 'diagonal' or 'top'")
        if (bath or mixed == 'screened') and rpa is None:
            raise ValueError("an MRRPA object is required for the bath or the screened kernel")
        self.ref, self.rpa, self.mixed, self.bath = ref, rpa, mixed, bath
        self.cross_sector, self.include_static = cross_sector, include_static
        self.bath_hamiltonian = bath_hamiltonian
        self.label = label or f"mixed={mixed}, bath={bath}, bath_hamiltonian={bath_hamiltonian}"

        nso = ref.nso
        inact, act = ref.inact_so, ref.act_so
        nI, nA = len(inact), ref.npoles
        ntop = nI + nA
        self.nso, self.nI, self.nA, self.ntop = nso, nI, nA, ntop

        # physical transition matrix T_D
        T_D = np.zeros((nso, ntop))
        T_D[inact, np.arange(nI)] = 1.0
        T_D[np.ix_(act, nI + np.arange(nA))] = ref.d_act
        self.T_D = T_D
        self.K_D = np.concatenate([ref.eps[inact], ref.kappa])

        # same-sector mask (virtual <-> attachment, core <-> removal)
        is_virt = np.array([P in set(ref.virt_so) for P in inact])
        same = (ref.pole_sign[None, :] > 0) == is_virt[:, None]
        self.same_sector = same

        K = np.diag(self.K_D)
        Kstat = T_D.T @ ref.Sigma_stat @ T_D if include_static else np.zeros((ntop, ntop))
        KB = np.zeros((nI, nA))
        if mixed != 'none':
            kern = self.mixed_kernel()
            KB = np.einsum('Pzyw,azyw->Pa', kern, ref.Cc)
        if not cross_sector:
            mask = np.ones((ntop, ntop))
            mask[:nI, nI:] = same
            mask[nI:, :nI] = same.T
            Kstat = Kstat * mask
            KB = KB * same
        K = K + Kstat
        K[:nI, nI:] += KB
        K[nI:, :nI] += KB.T
        self.K = 0.5 * (K + K.T)
        self.KB = KB
        self.Kstat = Kstat

        # sector sign of every top state: +1 attachment (virtual orbital or
        # N+1 pole), -1 removal (core orbital or N-1 pole)
        self.sigma_top = np.concatenate([np.where(is_virt, 1.0, -1.0), ref.pole_sign.astype(float)])

        # MR-GW bath
        self.Omega_B = np.zeros(0)
        self.Gt = None
        if bath and bath_hamiltonian == 'diagonal':
            # bath states (n, I) with zeroth-order energies kappa_n +- Omega_I
            self.E_B, self.Aamp = rpa.bath()
            self.Atil = T_D.T @ self.Aamp
        elif bath and bath_hamiltonian == 'top':
            # bath states (n, I) for all top states n and all coupled modes I;
            # the top-space Hamiltonian K acts inside the one-boson sector:
            #   B_I = K + sigma Omega_I,  coupling g_{m,(nI)} = t_m^T M_I t_n
            keep = rpa.mode_norm > 1e-12 * max(rpa.mode_norm.max(), 1e-300)
            self.Omega_B = rpa.Omega[keep]
            Mk = rpa.M[:, :, keep]                              # (nso, nso, KI)
            self.Gt = np.einsum('pm,prI,rn->Imn', T_D, Mk, T_D)  # (KI, ntop, ntop)
            KI = self.Omega_B.size
            self.Atil = np.ascontiguousarray(self.Gt.transpose(1, 0, 2).reshape(ntop, KI * ntop))
            self.E_B = (self.sigma_top[None, :] * self.Omega_B[:, None]).ravel() \
                + np.tile(np.diag(self.K), KI)
            self.Aamp = None
        else:
            self.E_B = np.zeros(0)
            self.Aamp = np.zeros((nso, 0))
            self.Atil = np.zeros((ntop, 0))
        self.nB = self.E_B.size
        self.n = ntop + self.nB
        self.T = np.hstack([T_D, np.zeros((nso, self.nB))])
        self._eig = None

    # ------------------------------------------------------------------
    def _bath_apply(self, Xb):
        """Action of the bath block on bath vectors Xb (nB, k)."""
        if self.bath_hamiltonian == 'diagonal' or self.nB == 0:
            return self.E_B[:, None] * Xb
        KI, ntop = self.Omega_B.size, self.ntop
        X3 = Xb.reshape(KI, ntop, -1)
        Y3 = np.einsum('mn,Ink->Imk', self.K, X3) \
            + (self.sigma_top[None, :, None] * self.Omega_B[:, None, None]) * X3
        return Y3.reshape(self.nB, -1)

    def _bath_block(self):
        """Dense bath block (only for small problems)."""
        if self.bath_hamiltonian == 'diagonal' or self.nB == 0:
            return np.diag(self.E_B)
        KI, ntop = self.Omega_B.size, self.ntop
        B = np.zeros((self.nB, self.nB))
        for I in range(KI):
            sl = slice(I * ntop, (I + 1) * ntop)
            B[sl, sl] = self.K + np.diag(self.sigma_top * self.Omega_B[I])
        return B

    # ------------------------------------------------------------------
    def mixed_kernel(self):
        """I_{Pz,yw} for P inactive, z,y,w active (local active indices)."""
        ref = self.ref
        ix = np.ix_(ref.inact_so, ref.act_so, ref.act_so, ref.act_so)
        if self.mixed == 'bare':
            return ref.VbarR[ix]
        VR = ref.VR[ix]
        W0 = self.rpa.W0[ix]
        # [I]_{Pz,yw} = v^R_{Pz,yw} - [W_D(0)]_{Pw,yz}
        return VR - W0.transpose(0, 3, 2, 1)

    # ------------------------------------------------------------------
    # matrix-free interface
    def matvec(self, X):
        X = np.asarray(X)
        one_d = X.ndim == 1
        X2 = X.reshape(self.n, -1)
        Xt, Xb = X2[:self.ntop], X2[self.ntop:]
        Yt = self.K @ Xt + self.Atil @ Xb
        Yb = self.Atil.T @ Xt + self._bath_apply(Xb)
        Y = np.vstack([Yt, Yb])
        return Y.ravel() if one_d else Y

    def diagonal(self):
        return np.concatenate([np.diag(self.K), self.E_B])

    def dense(self):
        H = np.zeros((self.n, self.n))
        H[:self.ntop, :self.ntop] = self.K
        H[:self.ntop, self.ntop:] = self.Atil
        H[self.ntop:, :self.ntop] = self.Atil.T
        H[self.ntop:, self.ntop:] = self._bath_block()
        return H

    # ------------------------------------------------------------------
    # dense reference solution
    def eig(self):
        if self._eig is None:
            E, U = np.linalg.eigh(self.dense())
            self._eig = (E, U)
        return self._eig

    def poles(self):
        """Eigenvalues E_lambda, residue vectors Z_lambda = T U_lambda and
        spectral weights |Z_lambda|^2."""
        E, U = self.eig()
        Z = self.T @ U
        return E, Z, np.sum(Z ** 2, axis=0)

    def dense_roots(self, guesses, degeneracy_tol=1e-8):
        """Dense reference for root following: for every guess vector the
        eigenvalue whose degenerate eigenspace (eigenvalues within
        degeneracy_tol) carries the largest total squared overlap with the
        guess.  Summing over degenerate clusters makes the choice independent
        of the arbitrary rotation within degenerate (e.g. spin) pairs."""
        E, U = self.eig()
        G = np.asarray(guesses, dtype=float)
        if G.ndim == 1:
            G = G[:, None]
        ov2 = (G.T @ U) ** 2                               # (m, n)
        # cluster degenerate eigenvalues
        clusters = []
        start = 0
        for k in range(1, E.size + 1):
            if k == E.size or E[k] - E[k - 1] > degeneracy_tol:
                clusters.append((start, k))
                start = k
        out = np.empty(G.shape[1])
        for j in range(G.shape[1]):
            best = max(clusters, key=lambda c: ov2[j, c[0]:c[1]].sum())
            out[j] = E[best[0]:best[1]].mean()
        return out

    def greens_function(self, z):
        """G(z) = T_D [z - K - Atil (z - E_B)^{-1} Atil^T]^{-1} T_D^T
        (Schur complement of the bath)."""
        Kz = self.K.astype(complex)
        if self.nB and self.bath_hamiltonian == 'diagonal':
            Kz = Kz + self.Atil @ ((1.0 / (z - self.E_B))[:, None] * self.Atil.T)
        elif self.nB:
            # sum_I g_I [z - K - sigma Omega_I]^{-1} g_I^T
            I_top = np.eye(self.ntop)
            for I in range(self.Omega_B.size):
                B_I = self.K + np.diag(self.sigma_top * self.Omega_B[I])
                gI = self.Gt[I]
                Kz = Kz + gI @ np.linalg.solve(z * I_top - B_I, gI.T)
        Gtop = np.linalg.solve(z * np.eye(self.ntop) - Kz, self.T_D.T)
        return self.T_D @ Gtop

    def dressed_bath_self_energy(self, z):
        """For bath_hamiltonian='top' and no cross-sector couplings the bath
        Schur complement equals the GW self-energy built with the poles
        (lambda_k, Z_k) of the top-space Hamiltonian itself,
        sum_{kI} (M_I Z_k)(M_I Z_k)^T / (z - lambda_k - sigma_k Omega_I);
        this returns that expression in the orbital basis (used as a check)."""
        lam, U = np.linalg.eigh(self.K)
        Z = self.T_D @ U                                   # (nso, ntop)
        sig = np.sign(np.einsum('nk,n,nk->k', U, self.sigma_top, U))
        keep = self.rpa.mode_norm > 1e-12 * max(self.rpa.mode_norm.max(), 1e-300)
        M = self.rpa.M[:, :, keep]
        Om = self.rpa.Omega[keep]
        a = np.einsum('prI,rk->pkI', M, Z)                 # (nso, ntop, KI)
        den = z - lam[:, None] - sig[:, None] * Om[None, :]  # (ntop, KI)
        return np.einsum('pkI,qkI,kI->pq', a, a, 1.0 / den)

    def spectral_function(self, omegas, eta, trace=True):
        """A(omega) = -1/pi Im G(omega + i eta); returns the trace or the
        full matrix for every frequency."""
        E, Z, _ = self.poles()
        omegas = np.asarray(omegas, dtype=float)
        lor = (eta / np.pi) / ((omegas[:, None] - E[None, :]) ** 2 + eta ** 2)  # (nw, nlam)
        if trace:
            return lor @ np.sum(Z ** 2, axis=0)
        return np.einsum('wl,pl,ql->wpq', lor, Z, Z)

    def spectral_function_resolvent(self, omegas, eta, trace=True):
        """A(omega) = -1/pi Im G(omega + i eta) from the resolvent (Schur
        complement of the bath); no diagonalization, suited to large baths."""
        out = []
        for w in np.asarray(omegas, dtype=float):
            G = self.greens_function(w + 1j * eta)
            out.append(-np.trace(G).imag / np.pi if trace else -G.imag / np.pi)
        return np.array(out)

    def moments(self):
        """Zeroth and first spectral moments M0 = T T^T, M1 = T H T^T."""
        M0 = self.T_D @ self.T_D.T
        M1 = self.T_D @ self.K @ self.T_D.T
        return M0, M1

    # ------------------------------------------------------------------
    # guess vectors for root following
    def unit_vector(self, idx):
        v = np.zeros(self.n)
        v[idx] = 1.0
        return v

    def index_inactive(self, P):
        return self.ref.inact_so.index(P)

    def index_active_pole(self, alpha):
        return self.nI + alpha

    def orbital_guess(self, p, sector=None):
        """Top-space vector representing orbital p: T^+ e_p, optionally
        restricted to the attachment ('attach') or removal ('remove') poles."""
        v = self.T[p, :].copy()
        if sector is not None and p in self.ref.act_so:
            sign = +1 if sector == 'attach' else -1
            keep = np.zeros(self.n, dtype=bool)
            keep[self.nI:self.ntop] = self.ref.pole_sign == sign
            v = v * keep
        nrm = np.linalg.norm(v)
        if nrm < 1e-12:
            raise ValueError("empty guess vector")
        return v / nrm

    def sector_of_root(self, vec):
        """Crude sector assignment of an eigenvector by its top-space weight."""
        wI = vec[:self.nI]
        wA = vec[self.nI:self.ntop]
        is_virt = np.array([P in set(self.ref.virt_so) for P in self.ref.inact_so])
        w_att = np.sum(wI[is_virt] ** 2) + np.sum(wA[self.ref.pole_sign > 0] ** 2)
        w_rem = np.sum(wI[~is_virt] ** 2) + np.sum(wA[self.ref.pole_sign < 0] ** 2)
        return 'attach' if w_att >= w_rem else 'remove'


# ----------------------------------------------------------------------------
def hall_insertion_gf(ref, rpa, z, mixed='bare'):
    """Illustrative Hall-form insertion of the (retarded) first-order mixed
    blocks into the generalized Dyson equation,

        M(z) = (1 - Sigma21)^{-1} G_D (1 - Sigma12)^{-1},
        G(z) = [M(z)^{-1} - Sigma11(z)]^{-1},   Sigma11 = Sigma_stat + MR-GW bath,

    which is the analogue of MR-GF1/"MR-GW + Sigma_1^{12/21}" of Wang, Fang,
    and Li (Appendix C, Fig. 6).  Because Sigma21 G_D^I Sigma12 has double
    poles, this Green's function is not PSD in general.
    """
    tmp = HermitianGF(ref, rpa, mixed=mixed, bath=False)
    B = tmp.KB                                   # (nI, nA)
    inact, act = ref.inact_so, ref.act_so
    nso = ref.nso
    G_D = ref.dyall_greens_function(z)
    Sig12 = np.zeros((nso, nso), dtype=complex)
    Sig12[np.ix_(inact, act)] = B @ (ref.d_act.T / (z - ref.kappa)[:, None])
    Sig21 = Sig12.T
    E_B, Aamp = rpa.bath()
    Sig11 = ref.Sigma_stat + Aamp @ (Aamp.T / (z - E_B)[:, None])
    I = np.eye(nso)
    M = np.linalg.solve(I - Sig21, G_D) @ np.linalg.inv(I - Sig12)
    return np.linalg.inv(np.linalg.inv(M) - Sig11)
