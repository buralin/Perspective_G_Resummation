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
                 cross_sector=True, include_static=True, label=None):
        if mixed not in ('none', 'bare', 'screened'):
            raise ValueError("mixed must be 'none', 'bare' or 'screened'")
        if (bath or mixed == 'screened') and rpa is None:
            raise ValueError("an MRRPA object is required for the bath or the screened kernel")
        self.ref, self.rpa, self.mixed, self.bath = ref, rpa, mixed, bath
        self.cross_sector, self.include_static = cross_sector, include_static
        self.label = label or f"mixed={mixed}, bath={bath}"

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

        # MR-GW bath
        if bath:
            self.E_B, self.Aamp = rpa.bath()
            self.Atil = T_D.T @ self.Aamp
        else:
            self.E_B = np.zeros(0)
            self.Aamp = np.zeros((nso, 0))
            self.Atil = np.zeros((ntop, 0))
        self.nB = self.E_B.size
        self.n = ntop + self.nB
        self.T = np.hstack([T_D, np.zeros((nso, self.nB))])
        self._eig = None

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
        Yb = self.Atil.T @ Xt + self.E_B[:, None] * Xb
        Y = np.vstack([Yt, Yb])
        return Y.ravel() if one_d else Y

    def diagonal(self):
        return np.concatenate([np.diag(self.K), self.E_B])

    def dense(self):
        H = np.zeros((self.n, self.n))
        H[:self.ntop, :self.ntop] = self.K
        H[:self.ntop, self.ntop:] = self.Atil
        H[self.ntop:, :self.ntop] = self.Atil.T
        H[self.ntop:, self.ntop:] = np.diag(self.E_B)
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

    def greens_function(self, z):
        """G(z) = T_D [z - K - Atil (z - E_B)^{-1} Atil^T]^{-1} T_D^T
        (Schur complement of the bath)."""
        Kz = self.K.astype(complex)
        if self.nB:
            Kz = Kz + self.Atil @ ((1.0 / (z - self.E_B))[:, None] * self.Atil.T)
        Gtop = np.linalg.solve(z * np.eye(self.ntop) - Kz, self.T_D.T)
        return self.T_D @ Gtop

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
