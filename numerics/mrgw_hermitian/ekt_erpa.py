"""
Extended Koopmans' theorem (EKT) charged states and extended RPA (ERPA)
neutral excitations of the active space, as a replacement for the exact
N_a +/- 1 and N_a eigenstates of the Dyall active Hamiltonian.

EKT (Day, Smith, Garrod 1974; Morrell, Parr, Levy 1975).  The N_a-1 and
N_a+1 states are expanded in the primary manifolds

    |alpha^-> = sum_q c_q a_q   |Xi_0>,      |alpha^+> = sum_q c_q a_q^+ |Xi_0>,

with q running over the active spin orbitals, and the coefficients follow
from the generalized eigenvalue problems

    A^- c = omega  gamma       c,   A^-_pq = <Xi_0| a_p^+ (H_act - E_0) a_q   |Xi_0>,
    A^+ c = omega (1 - gamma^T) c,  A^+_pq = <Xi_0| a_p   (H_act - E_0) a_q^+ |Xi_0>.

For the exact CAS ground state these matrices are functions of the 1- and
2-RDMs only (A^-_pq = <a_p^+ [H, a_q]>); here they are evaluated directly
with CI vectors, which gives the same numbers.  The EKT poles exhaust the
zeroth and first spectral moments of the active Green's function exactly:
sum_alpha d_alpha d_alpha^+ = 1 and sum_alpha kappa_alpha d_alpha d_alpha^+
equals the exact first moment.

ERPA (Chatterjee and Pernal, J. Chem. Phys. 137, 204109 (2012)).  Neutral
excitations from the equation of motion in the manifold of single
(de)excitation operators E_pq = a_p^+ a_q acting on the correlated reference,

    <0|[E_qp, [H, O_nu^+]]|0> = omega_nu <0|[E_qp, O_nu^+]|0>,

solved in the natural-spin-orbital basis, where the metric is diagonal
(n_q - n_p) and the problem takes the RPA form.  With the Dyall Hamiltonian
the inactive channels of the reference response (core -> active,
active -> virtual) reduce exactly to the EKT problems shifted by the
inactive orbital energies, so only the active-active channel is treated with
the ERPA equations.  The transition densities <nu|x^+ y|0> = (S c^nu) enter
the MR-RPA screening of the residual interaction like the exact ones.  The
ERPA response satisfies the zeroth and energy-weighted sum rules of the
exact active-space response within the single-excitation manifold.

Extended charged manifold (option charged_manifold='extended').  The
primary operators are supplemented by the 2h1p and 2p1h operators of the
active space,

    |alpha^-> = sum_z c_z a_z |Xi_0> + sum_{y, w<z} c_{ywz} a_y^+ a_w a_z |Xi_0>,
    |alpha^+> = sum_z c_z a_z^+|Xi_0> + sum_{y, w<z} c_{ywz} a_y a_w^+ a_z^+ |Xi_0>,

and the generalized eigenproblem is solved after canonical orthogonalization
of the (strongly linearly dependent) Gram matrix.  Because
(H - E_0) a_z |Xi_0> = [H, a_z] |Xi_0> lies in this span, the extended
manifold conserves the spectral moments M_0 ... M_3 of the active Green's
function exactly and reproduces the exact poles whenever it spans the
N_a -/+ 1 sector; the number of poles is bounded by the manifold rank
(at most ~ n_act^3) instead of the sector dimension.

The resulting object is a drop-in replacement for DyallReference in MRRPA
and HermitianGF (same attributes: poles, kappa, pole_sign, d_act, Cc, ...).
"""
import numpy as np

from .dyall import (DyallReference, ActivePole, apply_cre, apply_des,
                    apply_hamiltonian, HARTREE2EV)
from .mrrpa import solve_rpa


def _canonical_orthogonalization(S, tol):
    s, U = np.linalg.eigh(S)
    keep = s > tol
    return U[:, keep] / np.sqrt(s[keep])[None, :], int(np.sum(~keep))


class EKTERPAReference:
    """Dyall reference with EKT charged poles and ERPA neutral excitations.

    Parameters
    ----------
    ref : DyallReference
    charged_manifold : 'primary' (EKT: {a_x}, {a_x^+}) or 'extended'
        (additionally the 2h1p operators a_y^+ a_w a_z and the 2p1h operators
        a_y a_w^+ a_z^+ of the active space)
    occ_tol : directions of the EKT metrics (gamma, 1 - gamma) and pairs of
        the ERPA manifold with |n_q - n_p| below this value are discarded.
    lin_tol : relative threshold on the Gram-matrix eigenvalues of the
        extended charged manifold (canonical orthogonalization).
    """

    def __init__(self, ref, charged_manifold='primary', occ_tol=1e-8, lin_tol=1e-10,
                 verbose=True):
        if charged_manifold not in ('primary', 'extended'):
            raise ValueError("charged_manifold must be 'primary' or 'extended'")
        self._ref = ref
        self.charged_manifold = charged_manifold
        self.occ_tol = occ_tol
        self.lin_tol = lin_tol
        self.verbose = verbose
        self._build_ekt()
        self.C3 = DyallReference._three_operator_amplitudes(self)
        self.Cc = 0.5 * self.C3 - np.einsum('yw,za->azyw', ref.gamma_act, self.d_act)
        self._build_erpa()
        if verbose:
            self.summary()

    def __getattr__(self, name):
        if name == '_ref':
            raise AttributeError(name)
        return getattr(self._ref, name)

    # ------------------------------------------------------------------
    def _build_ekt(self):
        ref = self._ref
        ncas, nact, E0 = ref.ncas, ref.nact_so, ref.E0
        h, eri = ref.h_eff, ref.eri_cas
        rem_secs, att_secs = set(ref.remove_sectors), set(ref.attach_sectors)
        manifold = {}                      # (sign, sector) -> list of CI vectors

        def add(sign, sec, v):
            if v is None:
                return
            if (sign < 0 and sec not in rem_secs) or (sign > 0 and sec not in att_secs):
                return
            if np.linalg.norm(v) < 1e-12:
                return
            manifold.setdefault((sign, sec), []).append(v)

        # primary manifold: a_x |0>, a_x^+ |0>
        for x in range(nact):
            v, s = ref._des_xi0[x]
            add(-1, s, v)
            w, s = ref._cre_xi0[x]
            add(+1, s, w)
        # extended manifold: a_y^+ a_w a_z |0>  and  a_y a_w^+ a_z^+ |0>  (w > z)
        if self.charged_manifold == 'extended':
            for z in range(nact):
                v1, n1 = ref._des_xi0[z]
                w1, m1 = ref._cre_xi0[z]
                for w in range(z + 1, nact):
                    v2 = w2 = None
                    if v1 is not None:
                        v2, n2 = apply_des(v1, ncas, n1, w)
                    if w1 is not None:
                        w2, m2 = apply_cre(w1, ncas, m1, w)
                    for y in range(nact):
                        if v2 is not None:
                            v3, n3 = apply_cre(v2, ncas, n2, y)
                            add(-1, n3, v3)
                        if w2 is not None:
                            w3, m3 = apply_des(w2, ncas, m2, y)
                            add(+1, m3, w3)

        poles = []
        self.ekt_dropped = 0
        self.ekt_problems = {}
        self.charged_rank = {}
        for (sign, sec) in sorted(manifold, key=lambda k: (-k[0], k[1])):
            V = manifold[(sign, sec)]
            m = len(V)
            Vm = np.array([v.ravel() for v in V])
            HVm = np.array([(apply_hamiltonian(h, eri, ncas, sec, v) - E0 * v).ravel() for v in V])
            A = Vm @ HVm.T
            S = Vm @ Vm.T
            A = 0.5 * (A + A.T)
            S = 0.5 * (S + S.T)
            if self.charged_manifold == 'primary':
                tol = self.occ_tol
            else:
                tol = self.lin_tol * max(np.linalg.eigvalsh(S).max(), 1e-300)
            Xo, ndrop = _canonical_orthogonalization(S, tol)
            self.ekt_dropped += ndrop
            e, Ct = np.linalg.eigh(Xo.T @ A @ Xo)
            C = Xo @ Ct                                        # C^T S C = 1
            self.ekt_problems[(sign, sec)] = (A, S, e, C)
            self.charged_rank[(sign, sec)] = (e.size, m, ref.sectors[sec].nstates)
            ops = ref._des_xi0 if sign < 0 else ref._cre_xi0
            shape = V[0].shape
            for k in range(e.size):
                vec = (C[:, k] @ Vm).reshape(shape)
                d = np.zeros(nact)
                for x in range(nact):
                    ox, sx = ops[x]
                    if ox is None or sx != sec:
                        continue
                    # <alpha|a_x|Xi0> (removal) or <Xi0|a_x|alpha> (attachment)
                    d[x] = np.vdot(vec, ox) if sign < 0 else np.vdot(ox, vec)
                poles.append(ActivePole(sign, sec, k, sign * e[k], vec, d))
        self.poles = poles
        self.npoles = len(poles)
        self.kappa = np.array([p.kappa for p in poles])
        self.pole_sign = np.array([p.sign for p in poles])
        self.d_act = np.array([p.d for p in poles]).T

    # ------------------------------------------------------------------
    def _build_erpa(self):
        ref = self._ref
        ncas, nact, nel, E0 = ref.ncas, ref.nact_so, ref.nelecas, ref.E0
        h, eri = ref.h_eff, ref.eri_cas
        gamma = ref.gamma_act
        # natural spin orbitals, per spin block
        U = np.zeros((nact, nact))
        n = np.zeros(nact)
        for spin in (0, 1):
            xs = [x for x in range(nact) if x % 2 == spin]
            ns, Us = np.linalg.eigh(gamma[np.ix_(xs, xs)])
            n[xs] = ns
            U[np.ix_(xs, xs)] = Us
        self.no_occ, self.no_coeff = n, U
        # u_xy = x^+ y |0> (same spin) in the original basis
        u = {}
        for y in range(nact):
            v1, n1 = ref._des_xi0[y]
            if v1 is None:
                continue
            for x in range(nact):
                if x % 2 != y % 2:
                    continue
                v2, n2 = apply_cre(v1, ncas, n1, x)
                if v2 is None or n2 != nel:
                    continue
                u[(x, y)] = v2
        # manifold of natural-orbital pairs (p, q), same spin, n_p != n_q
        pairs = [(p, q) for p in range(nact) for q in range(nact)
                 if p != q and p % 2 == q % 2 and abs(n[q] - n[p]) > self.occ_tol]
        idx = {pq: i for i, pq in enumerate(pairs)}
        ut, Hut = {}, {}
        for (p, q) in pairs:
            xs = [x for x in range(nact) if x % 2 == p % 2]
            vec = None
            for x in xs:
                for y in xs:
                    if (x, y) not in u:
                        continue
                    term = U[x, p] * U[y, q] * u[(x, y)]
                    vec = term if vec is None else vec + term
            ut[(p, q)] = vec
            Hut[(p, q)] = apply_hamiltonian(h, eri, ncas, nel, vec) - E0 * vec
        npair = len(pairs)
        M = np.zeros((npair, npair))
        S = np.zeros((npair, npair))
        for i, (p, q) in enumerate(pairs):
            for j, (r, s) in enumerate(pairs):
                M[i, j] = np.vdot(ut[(p, q)], Hut[(r, s)]) + np.vdot(ut[(s, r)], Hut[(q, p)])
                S[i, j] = np.vdot(ut[(p, q)], ut[(r, s)]) - np.vdot(ut[(s, r)], ut[(q, p)])
        M = 0.5 * (M + M.T)
        exc = [i for i, (p, q) in enumerate(pairs) if n[q] > n[p]]
        dex = [idx[(pairs[i][1], pairs[i][0])] for i in exc]
        N = np.array([n[pairs[i][1]] - n[pairs[i][0]] for i in exc])
        self.erpa_metric_error = max(
            np.abs(S - np.diag(np.diag(S))).max(),
            np.abs(np.diag(S)[exc] - N).max())
        A = M[np.ix_(exc, exc)]
        B = M[np.ix_(exc, dex)]
        self.erpa_structure_error = max(np.abs(M[np.ix_(dex, dex)] - A).max(),
                                        np.abs(M[np.ix_(dex, exc)] - B).max())
        sq = np.sqrt(N)
        At = A / sq[:, None] / sq[None, :]
        Bt = B / sq[:, None] / sq[None, :]
        Omega, R, Sm = solve_rpa(At, Bt)
        Xt, Yt = 0.5 * (R + Sm), 0.5 * (R - Sm)
        X, Y = Xt / sq[:, None], Yt / sq[:, None]
        nmode = Omega.size
        rho_no = np.zeros((nmode, nact, nact))
        for k, i in enumerate(exc):
            p, q = pairs[i]
            rho_no[:, p, q] = N[k] * X[k, :]
            rho_no[:, q, p] = -N[k] * Y[k, :]
        self.erpa_omega = Omega
        self.erpa_rho = np.einsum('xp,yq,npq->nxy', U, U, rho_no)
        self.erpa_pairs, self.erpa_X, self.erpa_Y, self.erpa_N = pairs, X, Y, N

    # ------------------------------------------------------------------
    # interface used by MRRPA / HermitianGF
    def neutral_transition_densities(self):
        """ERPA excitation energies and rho[nu, x, y] = <nu|x^+ y|0>."""
        return self.erpa_omega, self.erpa_rho

    def exact_neutral_transition_densities(self):
        return self._ref.neutral_transition_densities()

    def pole_representation(self):
        nI = len(self.inact_so)
        T_D = np.zeros((self.nso, nI + self.npoles))
        T_D[self.inact_so, np.arange(nI)] = 1.0
        T_D[np.ix_(self.act_so, nI + np.arange(self.npoles))] = self.d_act
        K_D = np.concatenate([self.eps[self.inact_so], self.kappa])
        return T_D, K_D

    def dyall_greens_function(self, z):
        T_D, K_D = self.pole_representation()
        return T_D @ ((1.0 / (z - K_D))[:, None] * T_D.T)

    def pole_symmetry(self, ia):
        p = self.poles[ia]
        return self.state_symmetry(p.vec, p.sector)

    # ------------------------------------------------------------------
    def active_moments(self, order=3):
        """Spectral moments M_k = sum_alpha kappa_alpha^k d_alpha d_alpha^T,
        k = 0..order, of the active Green's function from the EKT poles and
        from the exact poles: list of (M_k, M_k_exact)."""
        ref = self._ref
        out = []
        for k in range(order + 1):
            Mk = self.d_act @ (self.kappa[:, None] ** k * self.d_act.T)
            Mkx = ref.d_act @ (ref.kappa[:, None] ** k * ref.d_act.T)
            out.append((Mk, Mkx))
        return out

    def charged_pole_deviation(self):
        """max |kappa(EKT) - kappa(exact)| over sorted poles when the numbers
        of poles coincide (complete manifold), else None."""
        ref = self._ref
        if self.npoles != ref.npoles:
            return None
        return float(np.abs(np.sort(self.kappa) - np.sort(ref.kappa)).max())

    def response_sum_rules(self):
        """Zeroth and energy-weighted sum rules of the active-space response,
        sum_nu [rho_pq rho_rs -/+ rho_qp rho_sr] (weighted by omega_nu for the
        energy-weighted one), from the ERPA and from the exact states.
        Returns the maximal deviations over same-spin pairs."""
        nact = self.nact_so
        same = np.array([[p % 2 == q % 2 and p != q for q in range(nact)] for p in range(nact)])

        def sums(om, rho):
            S0 = np.einsum('npq,nrs->pqrs', rho, rho) - np.einsum('nqp,nsr->pqrs', rho, rho)
            S1 = (np.einsum('n,npq,nrs->pqrs', om, rho, rho)
                  + np.einsum('n,nqp,nsr->pqrs', om, rho, rho))
            mask = same[:, :, None, None] & same[None, None, :, :]
            return S0 * mask, S1 * mask

        S0e, S1e = sums(*self.exact_neutral_transition_densities())
        S0r, S1r = sums(self.erpa_omega, self.erpa_rho)
        return np.abs(S0e - S0r).max(), np.abs(S1e - S1r).max(), np.abs(S1e).max()

    def summary(self):
        ref = self._ref
        label = ('primary {a_x}, {a_x^+}' if self.charged_manifold == 'primary'
                 else 'extended: 1h + 2h1p, 1p + 2p1h')
        print(f"EKT/ERPA reference (charged manifold: {label})")
        print(f"  EKT charged poles: {self.npoles} "
              f"({np.sum(self.pole_sign > 0)} attachment, {np.sum(self.pole_sign < 0)} removal); "
              f"exact: {ref.npoles}; null/linearly dependent manifold directions removed: "
              f"{self.ekt_dropped}")
        for (sign, sec), (rank, m, dim) in self.charged_rank.items():
            print(f"    sector {sec} ({'attachment' if sign > 0 else 'removal'}): "
                  f"rank {rank} from {m} manifold vectors, sector dimension {dim}")
        dev = self.charged_pole_deviation()
        if dev is not None:
            print(f"  manifold complete: max |kappa - kappa_exact| = {dev:.1e} Ha")
        att = np.sort(self.kappa[self.pole_sign > 0])
        rem = np.sort(-self.kappa[self.pole_sign < 0])
        print(f"  lowest EKT attachment energies (eV): "
              + ", ".join(f"{e * HARTREE2EV:.3f}" for e in att[:4]))
        print(f"  lowest EKT ionization energies (eV): "
              + ", ".join(f"{e * HARTREE2EV:.3f}" for e in rem[:4]))
        om_exact, _ = ref.neutral_transition_densities()
        print(f"  ERPA active excitations: {self.erpa_omega.size} (exact CAS states: {om_exact.size}); "
              f"metric/structure errors {self.erpa_metric_error:.1e}/{self.erpa_structure_error:.1e}")
        print(f"  lowest ERPA active excitation energies (eV): "
              + ", ".join(f"{w * HARTREE2EV:.3f}" for w in np.sort(self.erpa_omega)[:4]))
        print(f"  lowest exact active excitation energies (eV): "
              + ", ".join(f"{w * HARTREE2EV:.3f}" for w in np.sort(om_exact)[:4]))
