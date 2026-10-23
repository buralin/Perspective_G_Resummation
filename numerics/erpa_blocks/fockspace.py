"""Tiny dense Fock-space toolkit for verifying RDM formulas (n <= 8 spin orbitals)."""
import numpy as np
import itertools

def annihilators(n):
    dim = 2 ** n
    A = np.zeros((n, dim, dim))
    for k in range(dim):
        for p in range(n):
            if k >> p & 1:
                sign = (-1) ** bin(k & ((1 << p) - 1)).count('1')
                A[p, k ^ (1 << p), k] = sign
    return A

def number_sector(n, N):
    return [k for k in range(2 ** n) if bin(k).count('1') == N]

def random_integrals(n, rng):
    h = rng.normal(size=(n, n)); h = 0.5 * (h + h.T)
    T = rng.normal(size=(n, n, n, n))
    # V[p,r,q,s] = (pr|qs): symmetric under p<->r, q<->s, (pr)<->(qs)
    V = np.zeros_like(T)
    for perm in [(0, 1, 2, 3), (1, 0, 2, 3), (0, 1, 3, 2), (1, 0, 3, 2),
                 (2, 3, 0, 1), (3, 2, 0, 1), (2, 3, 1, 0), (3, 2, 1, 0)]:
        V += np.transpose(T, perm)
    return h, V / 8.0

def antisym(V):
    """w[p,q,r,s] = <pq||rs> = (pr|qs) - (ps|qr)."""
    return np.einsum('prqs->pqrs', V) - np.einsum('psqr->pqrs', V)

def hamiltonian(h, V, A):
    n = A.shape[0]
    Ad = np.transpose(A, (0, 2, 1))
    H = np.einsum('pq,pij,qjk->ik', h, Ad, A)
    # 1/2 sum V[p,r,q,s] p+ q+ s r
    for p in range(n):
        for q in range(n):
            cc = Ad[p] @ Ad[q]
            for r in range(n):
                for s in range(n):
                    H += 0.5 * V[p, r, q, s] * (cc @ A[s] @ A[r])
    return H

def rdms(psi, A):
    n = A.shape[0]
    Ad = np.transpose(A, (0, 2, 1))
    ap = np.array([A[p] @ psi for p in range(n)])          # a_p |psi>
    g = np.einsum('pi,qi->pq', ap, ap)                      # <a_p+ a_q> = (a_p psi).(a_q psi)
    aa = np.array([[A[s] @ (A[r] @ psi) for s in range(n)] for r in range(n)])  # a_s a_r |psi>  (index [r,s])
    # Gamma_{pq,rs} = <a_p+ a_q+ a_s a_r> = (a_q a_p psi) . (a_s a_r psi)
    G = np.einsum('pqi,rsi->pqrs', aa, aa)
    return g, G
