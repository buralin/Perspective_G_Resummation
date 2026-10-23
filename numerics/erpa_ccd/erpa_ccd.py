"""Extended (ERPA-based) ring / crossed-ring / ladder amplitude equations on a
CAS-type vacuum, following the common-amplitude construction of Pa2CCD.tex.

Conventions (see notes/erpa_channels.tex):
  spin orbital p = 2P + sigma;  w[p,q,r,s] = <pq||rs>;  gamma[p,q] = <p+ q>;
  Gamma[p,q,r,s] = <p+ q+ s r>;  f = generalized Fock; Lbar = symmetric Lagrangian.
  Fermionic doubles tensor t4[p,r,q,s] (p,r particle-like, q,s hole-like),
  antisymmetric in p<->r and q<->s.
"""
import numpy as np
from pyscf import gto, scf, mcscf, ao2mo, lib

# ----------------------------------------------------------------------------
# integrals and vacua
# ----------------------------------------------------------------------------
def so_integrals(mol, mo):
    nmo = mo.shape[1]
    h = mo.T @ scf.hf.get_hcore(mol) @ mo
    eri = ao2mo.restore(1, ao2mo.full(mol, mo), nmo)          # (PQ|RS) chemist
    n = 2 * nmo
    hs = np.zeros((n, n)); hs[0::2, 0::2] = h; hs[1::2, 1::2] = h
    v = np.zeros((n, n, n, n))                                 # v[p,r,q,s] = (pr|qs) spin-orbital
    for s1 in (0, 1):
        for s2 in (0, 1):
            v[s1::2, s1::2, s2::2, s2::2] = eri
    w = np.einsum('prqs->pqrs', v) - np.einsum('psqr->pqrs', v)  # <pq||rs>
    return hs, v, w

def wedge2(g):
    return np.einsum('pr,qs->pqrs', g, g) - np.einsum('ps,qr->pqrs', g, g)

def hf_vacuum(nocc_so, n):
    g = np.zeros((n, n)); g[np.arange(nocc_so), np.arange(nocc_so)] = 1.0
    G = wedge2(g)
    C = list(range(nocc_so)); A = []; V = list(range(nocc_so, n))
    return g, G, C, A, V

def cas_vacuum(mc):
    """Full-space spin-orbital gamma, Gamma for a CASSCF/CASCI vacuum (natural
    active orbitals assumed: mc.natorb=True)."""
    ncore, ncas = mc.ncore, mc.ncas
    nmo = mc.mo_coeff.shape[1]
    (d1a, d1b), (d2aa, d2ab, d2bb) = mc.fcisolver.make_rdm12s(mc.ci, ncas, mc.nelecas)
    nact = 2 * ncas
    g_act = np.zeros((nact, nact)); g_act[0::2, 0::2] = d1a; g_act[1::2, 1::2] = d1b
    G_act = np.zeros((nact,) * 4)
    a, b = slice(0, None, 2), slice(1, None, 2)
    # PySCF: dm2s[p,q,r,s] = <p+ r+ s q>;  Gamma[p,q,r,s] = <p+ q+ s r> = dm2s[p,r,q,s]
    G_act[a, a, a, a] = d2aa.transpose(0, 2, 1, 3)
    G_act[b, b, b, b] = d2bb.transpose(0, 2, 1, 3)
    G_act[a, b, a, b] = d2ab.transpose(0, 2, 1, 3)              # Gamma[pa,qb,ra,sb] = d2ab[P,R,Q,S]
    G_act[b, a, b, a] = d2ab.transpose(2, 0, 3, 1)              # Gamma[pb,qa,rb,sa] = d2ab[Q,S,P,R]
    G_act[a, b, b, a] = -d2ab.transpose(0, 2, 3, 1)             # Gamma[pa,qb,rb,sa] = -d2ab[P,S,Q,R]
    G_act[b, a, a, b] = -d2ab.transpose(2, 0, 1, 3)             # Gamma[pb,qa,ra,sb] = -d2ab[Q,R,P,S]
    nel = sum(mc.nelecas)
    assert abs(G_act + G_act.transpose(1, 0, 2, 3)).max() < 1e-10
    assert abs(G_act - G_act.transpose(2, 3, 0, 1)).max() < 1e-10
    assert abs(np.einsum('pqrq->pr', G_act) - (nel - 1) * g_act).max() < 1e-10
    n = 2 * nmo
    C = list(range(2 * ncore)); A = list(range(2 * ncore, 2 * (ncore + ncas))); V = list(range(2 * (ncore + ncas), n))
    g = np.zeros((n, n)); g[C, C] = 1.0; g[np.ix_(A, A)] = g_act
    lam = np.zeros((n,) * 4); lam[np.ix_(A, A, A, A)] = G_act - wedge2(g_act)
    G = wedge2(g) + lam
    return g, G, C, A, V

def energy_from_rdms(h, v, g, G, enuc):
    return enuc + np.einsum('pq,pq', h, g) + 0.5 * np.einsum('prqs,pqrs', v, G)

# ----------------------------------------------------------------------------
# ERPA arrays (verified formulas of notes/erpa_channels.tex)
# ----------------------------------------------------------------------------
def erpa_arrays(h, w, g, G):
    n = h.shape[0]; I = np.eye(n)
    f = h + np.einsum('ptru,tu->pr', w, g, optimize=True)
    L = np.einsum('pt,tq->pq', h, g) + 0.5 * np.einsum('putv,qutv->pq', w, G, optimize=True)
    Lb = 0.5 * (L + L.T)
    A0ph = (np.einsum('qs,pr->pqrs', g, f) + np.einsum('pr,qs->pqrs', g, f)
            - np.einsum('pr,qs->pqrs', I, Lb) - np.einsum('qs,pr->pqrs', I, Lb))
    Mph = (np.einsum('pr,qs->pqrs', h, g) + np.einsum('qs,pr->pqrs', h, g)
           - np.einsum('pr,qs->pqrs', I, Lb) - np.einsum('qs,pr->pqrs', I, Lb)
           + np.einsum('ptur,qtus->pqrs', w, G, optimize=True) + np.einsum('qtus,ptur->pqrs', w, G, optimize=True)
           - 0.5 * np.einsum('pstu,qrtu->pqrs', w, G, optimize=True) - 0.5 * np.einsum('qrtu,pstu->pqrs', w, G, optimize=True))
    Sph = np.einsum('pr,qs->pqrs', I, g) - np.einsum('qs,pr->pqrs', I, g)
    def PP(X):
        return X - X.transpose(1, 0, 2, 3) - X.transpose(0, 1, 3, 2) + X.transpose(1, 0, 3, 2)
    seed0 = np.einsum('qs,pr->pqrs', I, f - Lb) - np.einsum('qs,pr->pqrs', g, f)
    seed = seed0 - np.einsum('pr,qs->pqrs', h, g) + np.einsum('qs,pr->pqrs', g, f) + np.einsum('ptru,tsqu->pqrs', w, G, optimize=True)
    # (seed = d_qs(f-Lb)_pr - h_pr g_qs + sum w Gamma, cf. Eq. pp-M)
    dress = (np.einsum('pt,tqrs->pqrs', g, w) + np.einsum('qt,ptrs->pqrs', g, w)
             + np.einsum('rt,pqts->pqrs', g, w) + np.einsum('st,pqrt->pqrs', g, w))
    Mpp = PP(seed) + w - dress
    A0pp = PP(seed0)
    Spp = (np.einsum('pr,qs->pqrs', I, I) - np.einsum('ps,qr->pqrs', I, I) - np.einsum('pr,qs->pqrs', I, g)
           + np.einsum('ps,qr->pqrs', I, g) + np.einsum('qr,ps->pqrs', I, g) - np.einsum('qs,pr->pqrs', I, g))
    return dict(f=f, L=L, Lb=Lb, Mph=Mph, Sph=Sph, A0ph=A0ph, Mpp=Mpp, Spp=Spp, A0pp=A0pp)

# ----------------------------------------------------------------------------
# amplitude space and channel matrices
# ----------------------------------------------------------------------------
class ERPACCD:
    def __init__(self, h, w, g, G, C, A, V, verbose=1, max_active=4):
        self.n = n = h.shape[0]
        # rotate the active spin orbitals to natural spin orbitals (per spin) if gamma is not diagonal there
        A = list(A)
        gA = g[np.ix_(A, A)]
        if len(A) and abs(gA - np.diag(np.diag(gA))).max() > 1e-10:
            U = np.eye(n)
            for spin in (0, 1):
                xs = [x for x in A if x % 2 == spin]
                nocc_, Us = np.linalg.eigh(g[np.ix_(xs, xs)])
                U[np.ix_(xs, xs)] = Us
            h = U.T @ h @ U; g = U.T @ g @ U
            w = np.einsum('pa,qb,rc,sd,pqrs->abcd', U, U, U, U, w, optimize=True)
            G = np.einsum('pa,qb,rc,sd,pqrs->abcd', U, U, U, U, G, optimize=True)
            if verbose:
                print("ERPACCD: rotated active orbitals to natural spin orbitals")
        self.h, self.w, self.g, self.G = h, w, g, G
        self.C, self.A, self.V = list(C), list(A), list(V)
        isC = np.zeros(n, bool); isC[self.C] = True
        isA = np.zeros(n, bool); isA[self.A] = True
        isV = np.zeros(n, bool); isV[self.V] = True
        self.isC, self.isA, self.isV = isC, isA, isV
        arr = erpa_arrays(h, w, g, G)
        self.__dict__.update(arr)
        # ph pairs (p particle-like, q hole-like) containing an inactive index
        E = (np.outer(isV, isC) | np.outer(isA, isC) | np.outer(isV, isA))
        self.E = E
        EE = np.einsum('pq,rs->pqrs', E, E)
        self.Aph = self.Mph * EE
        self.A0 = self.A0ph * EE
        self.Kph = self.Aph - self.A0
        self.Bph = -self.Mph.transpose(0, 1, 3, 2) * EE          # B_(pq),(rs) = -M_(pq),(sr)
        self.Nph = np.einsum('pqpq->pq', self.Sph) * E
        offd = (self.Sph * EE - np.einsum('pq,pr,qs->pqrs', self.Nph, np.eye(n), np.eye(n)))
        assert abs(offd).max() < 1e-10, "ph metric not diagonal in this basis"
        assert (self.Nph[E] > 1e-10).all(), "non-positive ph norm"
        # fermionic amplitude mask
        self.mask4 = np.einsum('pq,rs,ps,rq->prqs', E, E, E, E).astype(float)
        nact4 = (isA[:, None, None, None].astype(int) + isA[None, :, None, None] + isA[None, None, :, None] + isA[None, None, None, :])
        self.mask4 = self.mask4 * (nact4 <= max_active)
        self.max_active = max_active
        # pp pairs: particle pairs with >=1 virtual, hole pairs with >=1 core
        PA = isV | isA; HA = isC | isA
        Pm = np.outer(PA, PA) & (np.outer(isV, PA) | np.outer(PA, isV)); np.fill_diagonal(Pm, False)
        Hm = np.outer(HA, HA) & (np.outer(isC, HA) | np.outer(HA, isC)); np.fill_diagonal(Hm, False)
        self.Pm, self.Hm = Pm, Hm
        PPm = np.einsum('pr,cd->prcd', Pm, Pm); HHm = np.einsum('qs,kl->qskl', Hm, Hm); PHm = np.einsum('pr,qs->prqs', Pm, Hm)
        self.App = self.Mpp * PPm; self.A0pp_ = self.A0pp * PPm; self.KA = self.App - self.A0pp_
        self.Cpp = self.Mpp * HHm; self.C0pp = self.A0pp * HHm; self.KC = self.Cpp - self.C0pp
        self.Bpp = -self.Mpp * PHm                                  # B^pp_(pr),(qs) = -M_(pr),(qs)
        self.NP = np.einsum('prpr->pr', self.Spp) * Pm
        self.NH = -np.einsum('qsqs->qs', self.Spp) * Hm
        assert (self.NP[Pm] > 1e-10).all() and (self.NH[Hm] > 1e-10).all()
        sp = self.Spp * PPm; sp2 = self.Spp * HHm
        dP = np.einsum('pr,pc,rd->prcd', self.NP, np.eye(n), np.eye(n)) - np.einsum('pr,pd,rc->prcd', self.NP, np.eye(n), np.eye(n))
        dH = -np.einsum('qs,qk,sl->qskl', self.NH, np.eye(n), np.eye(n)) + np.einsum('qs,ql,sk->qskl', self.NH, np.eye(n), np.eye(n))
        assert abs(sp - dP).max() < 1e-10 and abs(sp2 - dH).max() < 1e-10, "pp metric not diagonal"
        # driver in fermionic arrangement and energy weights
        self.B4 = self.Bph.transpose(0, 2, 1, 3) * self.mask4      # B4[p,r,q,s] = B_(pq),(rs)
        # fermionic antisymmetry of the driver (should hold on the mask)
        self.driver_asym = abs(self.B4 + self.B4.transpose(1, 0, 2, 3)).max() + abs(self.B4 + self.B4.transpose(0, 1, 3, 2)).max()
        self.verbose = verbose
        if verbose:
            print("ERPACCD: n_so=%d  core=%d active=%d virtual=%d  ph pairs=%d  amplitudes=%d  P pairs=%d H pairs=%d"
                  % (n, len(self.C), len(self.A), len(self.V), int(E.sum()), int(self.mask4.sum()), int(Pm.sum()), int(Hm.sum())))
            print("   driver antisymmetry error %.1e" % self.driver_asym)

    # -- channel residual pieces in fermionic arrangement [p,r,q,s]
    def eh_pieces(self, t4, crossed=False):
        A, A0, B, N = self.Aph, self.A0, self.Bph, self.Nph
        if not crossed:
            T = t4.transpose(0, 2, 1, 3)                     # T[p,q,r,s] = t4[p,r,q,s]
            sgn = 1.0
        else:
            T = t4.transpose(0, 3, 1, 2)                     # T'[p,s,r,q] = t4[p,r,q,s]
            sgn = -1.0
        AtN = np.einsum('pquv,uvrs,rs->pqrs', A, T, N, optimize=True)
        NtA = np.einsum('pq,pquv,uvrs->pqrs', N, T, A, optimize=True)
        A0tN = np.einsum('pquv,uvrs,rs->pqrs', A0, T, N, optimize=True)
        NtA0 = np.einsum('pq,pquv,uvrs->pqrs', N, T, A0, optimize=True)
        tBt = np.einsum('pq,pquv,uvwx,wxrs,rs->pqrs', N, T, B, T, N, optimize=True)
        Cm = sgn * B + A0tN + NtA0
        Rm = sgn * B + AtN + NtA + sgn * tBt
        if not crossed:
            back = lambda X: X.transpose(0, 2, 1, 3)
        else:
            back = lambda X: X.transpose(0, 2, 3, 1)
        return back(Cm), back(Rm)

    def pp_scatter(self, t4):
        T = t4.transpose(2, 3, 0, 1)                         # T[q,s,p,r] = t4[p,r,q,s]
        CtN = 0.5 * np.einsum('qskl,klpr,pr->qspr', self.KC, T, self.NP, optimize=True)
        NtA = 0.5 * np.einsum('qs,qscd,cdpr->qspr', self.NH, T, self.KA, optimize=True)
        tBt = 0.25 * np.einsum('qs,qscd,cdkl,klpr,pr->qspr', self.NH, T, self.Bpp, T, self.NP, optimize=True)
        return (CtN + NtA + tBt).transpose(2, 3, 0, 1)

    def pp_common(self, t4):
        T = t4.transpose(2, 3, 0, 1)
        BT = (-self.Mpp * np.einsum('pr,qs->prqs', self.Pm, self.Hm)).transpose(2, 3, 0, 1)
        CtN = 0.5 * np.einsum('qskl,klpr,pr->qspr', self.C0pp, T, self.NP, optimize=True)
        NtA = 0.5 * np.einsum('qs,qscd,cdpr->qspr', self.NH, T, self.A0pp_, optimize=True)
        return (BT + CtN + NtA).transpose(2, 3, 0, 1)

    def residual(self, t4, ring=1.0, ladder=1.0, pieces=False):
        Ceh, Reh = self.eh_pieces(t4, crossed=False)
        Cte, Rte = self.eh_pieces(t4, crossed=True)
        common = 0.5 * (Ceh + Cte)
        ringR = (Reh - Ceh) + (Rte - Cte)
        ladR = self.pp_scatter(t4)
        R = (common + ring * ringR + ladder * ladR) * self.mask4
        if pieces:
            return R, dict(common=common * self.mask4, ring=ringR * self.mask4, ladder=ladR * self.mask4)
        return R

    def denominator(self):
        n = self.n
        a0 = np.einsum('pqpq->pq', self.A0)                 # diagonal of A0 over pairs
        N = self.Nph
        D = 0.5 * (np.einsum('pq,rs->prqs', a0, N) + np.einsum('pq,rs->prqs', N, a0)
                   + np.einsum('ps,rq->prqs', a0, N) + np.einsum('ps,rq->prqs', N, a0))
        return D

    def energy(self, t4):
        return 0.25 * np.einsum('prqs,prqs', self.B4, t4)

    def solve(self, ring=1.0, ladder=1.0, tol=1e-9, maxiter=200, diis_space=8, t0=None, damping=0.0):
        D = self.denominator()
        Dsafe = np.where(self.mask4 > 0, D, 1.0)
        t = -self.B4 / Dsafe * self.mask4 if t0 is None else t0.copy()
        adiis = lib.diis.DIIS(); adiis.space = diis_space
        hist = []
        for it in range(maxiter):
            R = self.residual(t, ring, ladder)
            err = abs(R).max()
            E = self.energy(t)
            hist.append((it, E, err))
            if self.verbose > 1:
                print("  iter %3d  E_c = %.10f  |R| = %.2e" % (it, E, err))
            if err < tol:
                return t, E, True, hist
            tnew = t - R / Dsafe * self.mask4
            if damping:
                tnew = (1 - damping) * tnew + damping * t
            if not np.isfinite(tnew).all() or abs(tnew).max() > 1e3:
                if self.verbose:
                    print("  solve: amplitudes diverged at iteration %d" % it)
                return t, float('nan'), False, hist
            try:
                t = adiis.update(tnew)
            except Exception as exc:              # singular DIIS matrix after divergence
                if self.verbose:
                    print("  solve: DIIS failed (%s) at iteration %d" % (exc, it))
                return tnew, float('nan'), False, hist
        return t, self.energy(t), False, hist

    # -- second order (common term only, linear)
    def solve_second_order(self, tol=1e-10):
        return self.solve(ring=0.0, ladder=0.0, tol=tol)

    # -- pure ERPA ring: eigen-solution of the ph problem in the inactive-containing manifold
    def erpa_ring_eigen(self):
        n = self.n
        pairs = np.argwhere(self.E)
        idx = {(p, q): k for k, (p, q) in enumerate(pairs)}
        A = np.array([[self.Aph[p, q, r, s] for (r, s) in pairs] for (p, q) in pairs])
        B = np.array([[self.Bph[p, q, r, s] for (r, s) in pairs] for (p, q) in pairs])
        N = np.array([self.Nph[p, q] for (p, q) in pairs])
        sq = np.sqrt(N)
        At = A / sq[:, None] / sq[None, :]; Bt = B / sq[:, None] / sq[None, :]
        H = np.block([[At, Bt], [-Bt, -At]])
        wv, vv = np.linalg.eig(H)
        m = len(pairs)
        # positive-norm, positive-frequency solutions
        norm = (abs(vv[:m]) ** 2).sum(0) - (abs(vv[m:]) ** 2).sum(0)
        sel = (wv.real > 0) & (norm > 0)
        X = vv[:m, sel]; Y = vv[m:, sel]
        if X.shape[1] != m:
            print("  warning: ERPA ring selected %d of %d solutions; max imag %.1e" % (X.shape[1], m, abs(wv.imag).max()))
        Ts = (Y @ np.linalg.pinv(X)).real
        t = Ts / sq[:, None] / sq[None, :]
        t4 = np.zeros((n,) * 4)
        for k, (p, q) in enumerate(pairs):
            for l, (r, s) in enumerate(pairs):
                t4[p, r, q, s] = t[k, l]
        # fermionic antisymmetrization and restriction to the amplitude space
        t4a = 0.5 * (t4 - t4.transpose(0, 1, 3, 2)) * self.mask4
        omega = np.sort(wv.real[sel])
        return t4a, t4, omega
