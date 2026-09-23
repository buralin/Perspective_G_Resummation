"""
Spin-orbital reduced density matrices (RDMs) of the CAS wave function, their
cumulant decomposition and cumulant-truncated reconstructions.

Conventions
-----------
* D_k[p1, ..., pk, q1, ..., qk] = <Xi_0| p1^+ ... pk^+ qk ... q1 |Xi_0>
  (unnormalized; D_1 = gamma, D_2[p,q,r,s] = <p^+ q^+ s r>), antisymmetric
  in the first k and in the last k indices, over the active spin orbitals
  x = 2*x_spatial + spin.
* Wedge (shuffle) product of two (a,a)- and (b,b)-index tensors:

      (A ^ B)[P, Q] = sum_{S,T} sgn(S) sgn(T) A[P_S, Q_T] B[P_S', Q_T'],

  S, T running over the a-element subsets of the a+b upper/lower positions
  (order preserved, S', T' the complements).  Each distinct term appears once,
  so a product of m identical factors overcounts by m!.
* Cumulant expansion (Kutzelnigg--Mukherjee / Mazziotti):

      D_2 = 1/2! g^g + L_2
      D_3 = 1/3! g^g^g + g^L_2 + L_3
      D_4 = 1/4! g^g^g^g + 1/2! (g^g)^L_2 + 1/2! L_2^L_2 + g^L_3 + L_4

  with g = D_1.  Setting L_4 = 0 (and L_3 = 0) gives the cumulant-truncated
  reconstructions of D_4 (and D_3).
"""
import itertools
import numpy as np

from .dyall import apply_des
from pyscf.fci import cistring


# ----------------------------------------------------------------------------
# RDMs from a CI vector
# ----------------------------------------------------------------------------
def spin_orbital_rdms(vec, ncas, nelec, order=2):
    """D_1, ..., D_order of the (normalized) active-space CI vector vec in the
    (na, nb) sector, as a dict {k: D_k}."""
    nso = 2 * ncas
    out = {}
    # level k: vectors a_{q_k} ... a_{q_1} |0> for sorted tuples q_1 < ... < q_k
    level = {(): (vec, tuple(nelec))}
    for k in range(1, order + 1):
        new = {}
        for tup, (v, sec) in level.items():
            start = tup[-1] + 1 if tup else 0
            for q in range(start, nso):
                w, s2 = apply_des(v, ncas, sec, q)
                if w is None:
                    continue
                new[tup + (q,)] = (w, s2)
        level = new
        tuples = sorted(level)
        m = len(tuples)
        G = np.zeros((m, m))
        by_sector = {}
        for i, t in enumerate(tuples):
            by_sector.setdefault(level[t][1], []).append(i)
        for sec, idx in by_sector.items():
            V = np.array([level[tuples[i]][0].ravel() for i in idx])
            G[np.ix_(idx, idx)] = V @ V.T
        D = np.zeros((nso,) * (2 * k))
        P = np.array(tuples, dtype=int).reshape(m, k)
        for perm1 in itertools.permutations(range(k)):
            s1 = _perm_sign(perm1)
            Pp = P[:, perm1]
            for perm2 in itertools.permutations(range(k)):
                s2 = _perm_sign(perm2)
                Qp = P[:, perm2]
                idx = tuple(Pp[:, None, j] for j in range(k)) + tuple(Qp[None, :, j] for j in range(k))
                D[idx] = s1 * s2 * G
        out[k] = D
    return out


def _perm_sign(perm):
    sign = 1
    p = list(perm)
    for i in range(len(p)):
        while p[i] != i:
            j = p[i]
            p[i], p[j] = p[j], p[i]
            sign = -sign
    return sign


def determinant_vector(ncas, nelec, occ_a=None, occ_b=None):
    """CI vector of a single determinant (default: lowest orbitals occupied)."""
    na, nb = nelec
    occ_a = list(range(na)) if occ_a is None else occ_a
    occ_b = list(range(nb)) if occ_b is None else occ_b
    stra = sum(1 << x for x in occ_a)
    strb = sum(1 << x for x in occ_b)
    v = np.zeros((cistring.num_strings(ncas, na), cistring.num_strings(ncas, nb)))
    v[cistring.str2addr(ncas, na, stra), cistring.str2addr(ncas, nb, strb)] = 1.0
    return v


# ----------------------------------------------------------------------------
# wedge product and cumulants
# ----------------------------------------------------------------------------
def _subset_sign(S):
    """Sign of the permutation that moves the sorted positions S to the front."""
    return (-1) ** sum(s - i for i, s in enumerate(S))


def wedge(A, B):
    """Shuffle (Grassmann) product of an (a,a) tensor and a (b,b) tensor."""
    a, b = A.ndim // 2, B.ndim // 2
    k = a + b
    n = A.shape[0]
    up = 'abcdefgh'[:k]
    lo = 'ijklmnop'[:k]
    out = np.zeros((n,) * (2 * k))
    positions = range(k)
    for S in itertools.combinations(positions, a):
        Sc = [i for i in positions if i not in S]
        sS = _subset_sign(S)
        for T in itertools.combinations(positions, a):
            Tc = [j for j in positions if j not in T]
            sT = _subset_sign(T)
            subA = ''.join(up[i] for i in S) + ''.join(lo[j] for j in T)
            subB = ''.join(up[i] for i in Sc) + ''.join(lo[j] for j in Tc)
            out += (sS * sT) * np.einsum(f'{subA},{subB}->{up}{lo}', A, B)
    return out


def cumulants(rdms):
    """Cumulants L_2, ..., L_order from the RDMs D_1, ..., D_order."""
    g = rdms[1]
    L = {}
    if 2 in rdms:
        L[2] = rdms[2] - 0.5 * wedge(g, g)
    if 3 in rdms:
        L[3] = rdms[3] - wedge(wedge(g, g), g) / 6.0 - wedge(g, L[2])
    if 4 in rdms:
        gg = wedge(g, g)
        L[4] = (rdms[4] - wedge(wedge(gg, g), g) / 24.0 - 0.5 * wedge(gg, L[2])
                - 0.5 * wedge(L[2], L[2]) - wedge(g, L[3]))
    return L


def reconstruct(rdms, drop=()):
    """RDMs with the cumulants listed in `drop` set to zero.  drop=(4,) gives
    D_4 without L_4 (L_3 kept exact); drop=(3, 4) sets L_3 = L_4 = 0 in D_3
    and D_4.  D_1 and D_2 are never changed."""
    g = rdms[1]
    L = cumulants({k: rdms[k] for k in rdms if k <= 3})
    out = dict(rdms)
    L3 = np.zeros_like(rdms[3]) if 3 in drop else L[3]
    if 3 in drop and 3 in rdms:
        out[3] = wedge(wedge(g, g), g) / 6.0 + wedge(g, L[2])
    if 4 in rdms and (4 in drop or 3 in drop):
        gg = wedge(g, g)
        out[4] = (wedge(wedge(gg, g), g) / 24.0 + 0.5 * wedge(gg, L[2])
                  + 0.5 * wedge(L[2], L[2]) + wedge(g, L3))
        if 4 not in drop:
            # keep the exact L_4 (only L_3 dropped)
            L4 = rdms[4] - (wedge(wedge(gg, g), g) / 24.0 + 0.5 * wedge(gg, L[2])
                            + 0.5 * wedge(L[2], L[2]) + wedge(g, L[3]))
            out[4] = out[4] + L4
    return out


def cumulant_norms(rdms):
    """Frobenius norms of D_k and L_k, k = 2.. (diagnostics)."""
    L = cumulants(rdms)
    return {k: (float(np.linalg.norm(rdms[k])), float(np.linalg.norm(L[k]))) for k in L}
