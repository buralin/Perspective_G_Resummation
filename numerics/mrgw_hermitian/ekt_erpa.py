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
                    apply_hamiltonian, HARTREE2EV, spin_orbital_one_body,
                    spin_orbital_two_body)
from .mrrpa import solve_rpa
from . import wick
from . import rdm as rdmmod


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
    response : 'erpa' (default) or 'exact'; the neutral active excitations
        that enter the active-active channel of the MR-RPA.  'exact' keeps
        the CAS eigenstates and isolates the effect of the EKT charged
        manifold.
    realization : 'ci' (default) evaluates all matrix elements by applying
        operators and H_act to CI vectors; 'rdm' evaluates them as
        contractions of the active integrals with the spin-orbital reduced
        density matrices D_1..D_4 of |Xi_0> (Wick engine of wick.py; RDMs of
        rdm.py).  Both give identical numbers for exact RDMs.
    cumulant_drop : tuple of cumulant orders set to zero in the RDM
        realization, () (exact), (4,) (D_4 without the four-body cumulant)
        or (3, 4) (D_3 and D_4 without the three- and four-body cumulants).
        Only D_3 and D_4 are affected, i.e. only the extended manifold.
    """

    def __init__(self, ref, charged_manifold='primary', occ_tol=1e-8, lin_tol=1e-10,
                 response='erpa', realization='ci', cumulant_drop=(), verbose=True):
        if charged_manifold not in ('primary', 'extended'):
            raise ValueError("charged_manifold must be 'primary' or 'extended'")
        if response not in ('erpa', 'exact'):
            raise ValueError("response must be 'erpa' or 'exact'")
        if realization not in ('ci', 'rdm'):
            raise ValueError("realization must be 'ci' or 'rdm'")
        cumulant_drop = tuple(sorted(int(k) for k in cumulant_drop))
        if cumulant_drop and realization != 'rdm':
            raise ValueError("cumulant truncation requires realization='rdm'")
        if any(k not in (3, 4) for k in cumulant_drop) or cumulant_drop == (3,):
            raise ValueError("cumulant_drop must be (), (4,) or (3, 4)")
        self._ref = ref
        self.charged_manifold = charged_manifold
        self.response = response
        self.realization = realization
        self.cumulant_drop = cumulant_drop
        self.occ_tol = occ_tol
        self.lin_tol = lin_tol
        self.verbose = verbose
        if realization == 'rdm':
            order = 4 if charged_manifold == 'extended' else 2
            self.rdms_exact = ref.rdms(order)
            self.rdms = (rdmmod.reconstruct(self.rdms_exact, drop=cumulant_drop)
                         if cumulant_drop else self.rdms_exact)
            self._tensors = {'h': spin_orbital_one_body(ref.h_eff),
                             'V': spin_orbital_two_body(ref.eri_cas)}
            self._build_ekt_rdm()
            self.C3 = self._three_operator_amplitudes_rdm()
        else:
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
    # RDM realization (Wick engine + reduced density matrices)
    # ------------------------------------------------------------------
    @staticmethod
    def _operator_types(sign):
        """Ket operators of the charged manifolds with their free labels:
        removal 1h: a_y, 2h1p: a_y^+ a_w a_z; attachment 1p: a_y^+,
        2p1h: a_y a_w^+ a_z^+.  The bra of each type is the Hermitian
        conjugate with the labels (y, w, z) -> (a, b, c)."""
        c, d = wick.cre, wick.des
        if sign < 0:
            return {'1h': ([d('y')], ('y',)), '2h1p': ([c('y'), d('w'), d('z')], ('y', 'w', 'z'))}
        return {'1p': ([c('y')], ('y',)), '2p1h': ([d('y'), c('w'), c('z')], ('y', 'w', 'z'))}

    @staticmethod
    def _bra(ops, labels):
        sub = dict(zip(labels, ('a', 'b', 'c')))
        ket = wick.string([(sub[l], dg) for l, dg in ops])
        return wick.dagger(ket), tuple(sub[l] for l in labels)

    def _eval(self, expr, free):
        return wick.evaluate(expr, free, self._ref.nact_so, self._tensors, self.rdms)

    def _manifold_entries(self, sign):
        """Ordered list of (type, index tuple) of the manifold operators,
        in the same order as the CI realization."""
        n = self._ref.nact_so
        entries = [('1h' if sign < 0 else '1p', (x,)) for x in range(n)]
        if self.charged_manifold == 'extended':
            t = '2h1p' if sign < 0 else '2p1h'
            entries += [(t, (y, w, z)) for z in range(n) for w in range(z + 1, n) for y in range(n)]
        return entries

    def _entry_sector(self, sign, typ, idx):
        na, nb = self._ref.nelecas
        # creators / annihilators of the operator, per spin orbital index
        if typ in ('1h', '1p'):
            crs, ans = ((), idx) if sign < 0 else (idx, ())
        elif typ == '2h1p':
            crs, ans = (idx[0],), (idx[1], idx[2])
        else:
            crs, ans = (idx[1], idx[2]), (idx[0],)
        da = sum(1 for x in crs if x % 2 == 0) - sum(1 for x in ans if x % 2 == 0)
        db = sum(1 for x in crs if x % 2 == 1) - sum(1 for x in ans if x % 2 == 1)
        return (na + da, nb + db)

    def _build_ekt_rdm(self):
        ref = self._ref
        n = ref.nact_so
        rem_secs, att_secs = set(ref.remove_sectors), set(ref.attach_sectors)
        H = wick.hamiltonian()
        poles = []
        self.ekt_dropped = 0
        self.ekt_problems = {}
        self.charged_rank = {}
        self.metric_negative = {}
        self.rdm_orders_used = set()
        self._pole_ops = []                       # (type, idx, coefficient) of the dominant component
        for sign in (+1, -1):
            types = self._operator_types(sign)
            if self.charged_manifold == 'primary':
                types = {k: v for k, v in types.items() if k in ('1h', '1p')}
            # block arrays S[tb,tk], A[tb,tk] over free labels (bra..., ket...)
            Sarr, Aarr, dS, C3arr = {}, {}, {}, {}
            for tk, (kops, klab) in types.items():
                ket = wick.string(kops)
                HX = wick.commutator(H, ket)
                for tb, (bops, blab) in types.items():
                    bra, bl = self._bra(bops, blab)
                    eS = wick.normal_order(wick.mul(bra, ket))
                    eA = wick.normal_order(wick.mul(bra, HX))
                    self.rdm_orders_used |= wick.rdm_orders(eA)
                    Sarr[(tb, tk)] = self._eval(eS, bl + klab)
                    Aarr[(tb, tk)] = self._eval(eA, bl + klab)
                # residues and three-operator amplitudes
                bra, bl = self._bra(kops, klab)
                three = wick.string([wick.cre('u'), wick.des('v'), wick.des('t')])
                if sign < 0:
                    dS[tk] = self._eval(wick.normal_order(wick.mul(bra, wick.string([wick.des('x')]))), bl + ('x',))
                    C3arr[tk] = self._eval(wick.normal_order(wick.mul(bra, three)), bl + ('t', 'u', 'v'))
                else:
                    dS[tk] = self._eval(wick.normal_order(wick.mul(wick.string([wick.des('x')]), ket)), ('x',) + klab)
                    C3arr[tk] = self._eval(wick.normal_order(wick.mul(three, ket)), ('t', 'u', 'v') + klab)
            # manifold entries grouped by sector, vectors of vanishing norm removed
            groups = {}
            for typ, idx in self._manifold_entries(sign):
                sec = self._entry_sector(sign, typ, idx)
                if (sign < 0 and sec not in rem_secs) or (sign > 0 and sec not in att_secs):
                    continue
                norm2 = Sarr[(typ, typ)][idx + idx]
                if norm2 < 1e-24:
                    continue
                groups.setdefault(sec, []).append((typ, idx))
            for sec in sorted(groups):
                ent = groups[sec]
                m = len(ent)
                A = np.zeros((m, m))
                S = np.zeros((m, m))
                bytype = {}
                for j, (typ, idx) in enumerate(ent):
                    bytype.setdefault(typ, []).append(j)
                for tb, rows in bytype.items():
                    I = np.array([ent[j][1] for j in rows])
                    for tk, cols in bytype.items():
                        J = np.array([ent[j][1] for j in cols])
                        sel = (tuple(I[:, None, k] for k in range(I.shape[1]))
                               + tuple(J[None, :, k] for k in range(J.shape[1])))
                        S[np.ix_(rows, cols)] = Sarr[(tb, tk)][sel]
                        A[np.ix_(rows, cols)] = Aarr[(tb, tk)][sel]
                A = 0.5 * (A + A.T)
                S = 0.5 * (S + S.T)
                sev = np.linalg.eigvalsh(S)
                self.metric_negative[(sign, sec)] = int(np.sum(sev < -1e-10))
                tol = self.occ_tol if self.charged_manifold == 'primary' else self.lin_tol * max(sev.max(), 1e-300)
                Xo, ndrop = _canonical_orthogonalization(S, tol)
                self.ekt_dropped += ndrop
                e, Ct = np.linalg.eigh(Xo.T @ A @ Xo)
                C = Xo @ Ct
                self.ekt_problems[(sign, sec)] = (A, S, e, C)
                self.charged_rank[(sign, sec)] = (e.size, m, ref.sectors[sec].nstates)
                # residues d_x = sum_j c_j <X_j^+ a_x> (removal) / <a_x X_j> (attachment)
                Dman = np.zeros((m, n))
                C3man = np.zeros((m, n, n, n))
                for tb, rows in bytype.items():
                    I = np.array([ent[j][1] for j in rows])
                    if sign < 0:
                        sel = tuple(I[:, None, k] for k in range(I.shape[1])) + (np.arange(n)[None, :],)
                        Dman[rows] = dS[tb][sel]
                        sel3 = (tuple(I[:, None, None, None, k] for k in range(I.shape[1]))
                                + (np.arange(n)[None, :, None, None], np.arange(n)[None, None, :, None],
                                   np.arange(n)[None, None, None, :]))
                        C3man[rows] = C3arr[tb][sel3]
                    else:
                        sel = (np.arange(n)[None, :],) + tuple(I[:, None, k] for k in range(I.shape[1]))
                        Dman[rows] = dS[tb][sel]
                        sel3 = ((np.arange(n)[None, :, None, None], np.arange(n)[None, None, :, None],
                                 np.arange(n)[None, None, None, :])
                                + tuple(I[:, None, None, None, k] for k in range(I.shape[1])))
                        C3man[rows] = C3arr[tb][sel3]
                d_all = C.T @ Dman                                   # (rank, n)
                C3_all = np.einsum('jk,jtuv->ktuv', C, C3man)        # [k, t=z, u=y, v=w]
                for k in range(e.size):
                    poles.append(ActivePole(sign, sec, k, sign * e[k], None, d_all[k]))
                    self._pole_ops.append((ent, C[:, k]))
                    poles[-1].vec = None
                    self._c3_rdm = getattr(self, '_c3_rdm', [])
                    self._c3_rdm.append(C3_all[k])
        self.poles = poles
        self.npoles = len(poles)
        self.kappa = np.array([p.kappa for p in poles])
        self.pole_sign = np.array([p.sign for p in poles])
        self.d_act = np.array([p.d for p in poles]).T

    def _three_operator_amplitudes_rdm(self):
        """C3[alpha, z, y, w] = <Xi0|y^+ w z|alpha^+> or <alpha^-|y^+ w z|Xi0>
        from the RDMs (assembled in _build_ekt_rdm)."""
        return np.array(self._c3_rdm)

    def pole_symmetry(self, ia):
        p = self.poles[ia]
        if p.vec is not None:
            return self.state_symmetry(p.vec, p.sector)
        ref = self._ref
        if ref.orbsym is None:
            return None
        ent, c = self._pole_ops[ia]
        typ, idx = ent[int(np.argmax(np.abs(c)))]
        sym = ref.state_symmetry(ref.xi0, ref.nelecas)
        for x in idx:
            sym ^= int(ref.orbsym[ref.ncore + x // 2])
        return sym

    def _erpa_matrices_rdm(self, pairs, U):
        """Double-commutator matrix M_(pq),(rs) = <[E_qp,[H,E_rs]]> and metric
        S_(pq),(rs) = <[E_qp,E_rs]> in the natural-orbital basis from the
        1- and 2-RDM."""
        n = self._ref.nact_so
        h = U.T @ self._tensors['h'] @ U
        V = np.einsum('pa,rb,qc,sd,prqs->abcd', U, U, U, U, self._tensors['V'], optimize=True)
        D1 = U.T @ self.rdms[1] @ U
        D2 = np.einsum('pa,qb,rc,sd,pqrs->abcd', U, U, U, U, self.rdms[2], optimize=True)
        Eqp = wick.string([wick.cre('q'), wick.des('p')])
        Ers = wick.string([wick.cre('r'), wick.des('s')])
        eM = wick.normal_order(wick.commutator(Eqp, wick.commutator(wick.hamiltonian(), Ers)))
        eS = wick.normal_order(wick.commutator(Eqp, Ers))
        assert max(wick.rdm_orders(eM)) <= 2
        Marr = wick.evaluate(eM, ['p', 'q', 'r', 's'], n, {'h': h, 'V': V}, {1: D1, 2: D2})
        Sarr = wick.evaluate(eS, ['p', 'q', 'r', 's'], n, {'h': h, 'V': V}, {1: D1, 2: D2})
        P = np.array(pairs)
        sel = (P[:, None, 0], P[:, None, 1], P[None, :, 0], P[None, :, 1])
        return Marr[sel], Sarr[sel]

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
        # u_xy = x^+ y |0> (same spin) in the original basis (CI realization)
        u = {}
        for y in range(nact if self.realization == 'ci' else 0):
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
        for (p, q) in (pairs if self.realization == 'ci' else []):
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
        if self.realization == 'rdm':
            M, S = self._erpa_matrices_rdm(pairs, U)
        else:
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
        """ERPA excitation energies and rho[nu, x, y] = <nu|x^+ y|0>
        (the exact CAS values for response='exact')."""
        if self.response == 'exact':
            return self._ref.neutral_transition_densities()
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
        real = ('CI vectors' if self.realization == 'ci' else
                'RDMs' + (f", cumulants {self.cumulant_drop} set to zero" if self.cumulant_drop else ' (exact)'))
        print(f"EKT/ERPA reference (charged manifold: {label}; active response: "
              f"{'ERPA' if self.response == 'erpa' else 'exact CAS states'}; realization: {real})")
        if self.realization == 'rdm':
            print(f"  RDM orders used in the EKT matrices: {sorted(self.rdm_orders_used)}; "
                  f"negative metric eigenvalues per sector: {self.metric_negative}")
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
