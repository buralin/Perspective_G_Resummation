import numpy as np, sys, time
from fockspace import *
np.set_printoptions(precision=3, linewidth=150)
n, N = 7, 3
rng = np.random.default_rng(7)
A = annihilators(n); Ad = np.transpose(A, (0, 2, 1)); dim = 2 ** n
h, V = random_integrals(n, rng)
w = antisym(V)
H = hamiltonian(h, V, A)
sec = number_sector(n, N)
psi = np.zeros(dim); psi[sec] = rng.normal(size=len(sec)); psi /= np.linalg.norm(psi)
g, G = rdms(psi, A)
print("Tr gamma =", np.trace(g), " (N =", N, ")  sym err", abs(g - g.T).max(), " Gamma trace", np.einsum('pqpq', G), "= N(N-1)")
E = np.einsum('pq,pq', h, g) + 0.5 * np.einsum('prqs,pqrs', V, G)
print("E from RDMs - <H> =", E - psi @ H @ psi)
I = np.eye(n)
# ---------- brute force ph double commutator  M[p,q,r,s] = <[E_qp,[H,E_rs]]>
Hpsi = H @ psi
Epsi = np.array([[Ad[p] @ (A[q] @ psi) for q in range(n)] for p in range(n)])          # E_pq |psi>
Cpsi = np.einsum('ij,rsj->rsi', H, Epsi) - np.array([[Ad[r] @ (A[s] @ Hpsi) for s in range(n)] for r in range(n)])  # [H,E_rs]|psi>
# <[E_qp, X]> = (E_pq psi).(X psi) - (X^+ psi).(E_qp psi);  [H,E_rs]^+ = -[H,E_sr]
Mph = np.einsum('pqi,rsi->pqrs', Epsi, Cpsi) + np.einsum('sri,qpi->pqrs', Cpsi, Epsi)
Sph = np.einsum('pqi,rsi->pqrs', Epsi, Epsi) - np.einsum('sri,qpi->pqrs', Epsi, Epsi)
# ---------- compact formulas
calF = np.einsum('pt,tq->pq', h, g) + np.einsum('ptuv,qutv->pq', V, G)            # energy-weighted density (Lagrangian)
calF2 = np.einsum('pt,tq->pq', h, g) + 0.5 * np.einsum('putv,qutv->pq', w, G)
print("Lagrangian: v-form vs w-form", abs(calF - calF2).max())
F = 0.5 * (calF + calF.T)
FD = h + np.einsum('ptru,tu->pr', w, g)                                            # generalized Fock
Mc = (np.einsum('pr,qs->pqrs', h, g) + np.einsum('sq,rp->pqrs', h, g)
      - np.einsum('pr,sq->pqrs', I, calF) - np.einsum('qs,rp->pqrs', I, calF)
      + np.einsum('ptur,tqsu->pqrs', w, G) + np.einsum('tsqu,trpu->pqrs', w, G)
      + 0.5 * np.einsum('pstu,qrut->pqrs', w, G) + 0.5 * np.einsum('tuqr,tusp->pqrs', w, G))
print("ph plain double commutator: compact 8-term form vs brute force:", abs(Mc - Mph).max())
Msym = 0.5 * (Mph + np.transpose(Mph, (2, 3, 0, 1)))
Mnice = (np.einsum('pr,qs->pqrs', h, g) + np.einsum('qs,pr->pqrs', h, g)
         - np.einsum('pr,qs->pqrs', I, F) - np.einsum('qs,pr->pqrs', I, F)
         + np.einsum('ptur,qtus->pqrs', w, G) + np.einsum('qtus,ptur->pqrs', w, G)
         - 0.5 * np.einsum('pstu,qrtu->pqrs', w, G) - 0.5 * np.einsum('qrtu,pstu->pqrs', w, G))
print("ph symmetrized: 'nice' form vs brute force:", abs(Mnice - Msym).max())
Sc = np.einsum('pr,qs->pqrs', I, g) - np.einsum('qs,pr->pqrs', I, g)
print("ph metric:", abs(Sc - Sph).max())
# <[H,E_pq]> = ?
comm1 = np.einsum('i,pqi->pq', psi, Cpsi)
print("<[H,E_pq]> - (calF_pq - calF_qp):", abs(comm1 - (calF - calF.T)).max(), "  or + :", abs(comm1 + (calF - calF.T)).max())
# antisymmetric part of M
print("M - M^T vs -d_pr(calF_sq-calF_qs) - d_qs(calF_rp - calF_pr):",
      abs((Mph - np.transpose(Mph, (2, 3, 0, 1))) - (-np.einsum('pr,sq->pqrs', I, calF - calF.T) - np.einsum('qs,rp->pqrs', I, calF - calF.T))).max())
# cumulant form
lam = G - np.einsum('pr,qs->pqrs', g, g) + np.einsum('ps,qr->pqrs', g, g)
Mcum = (np.einsum('qs,pr->pqrs', g, FD) + np.einsum('pr,qs->pqrs', g, FD)
        - np.einsum('pr,qs->pqrs', I, F) - np.einsum('qs,pr->pqrs', I, F)
        + np.einsum('ptur,qu,ts->pqrs', w, g, g) + np.einsum('qtus,pu,tr->pqrs', w, g, g)
        - np.einsum('pstu,qt,ru->pqrs', w, g, g) - np.einsum('qrtu,pt,su->pqrs', w, g, g)
        + np.einsum('ptur,qtus->pqrs', w, lam) + np.einsum('qtus,ptur->pqrs', w, lam)
        - 0.5 * np.einsum('pstu,qrtu->pqrs', w, lam) - 0.5 * np.einsum('qrtu,pstu->pqrs', w, lam))
print("ph symmetrized: cumulant form vs brute force:", abs(Mcum - Msym).max())
Fcum = 0.5 * (np.einsum('pt,tq->pq', FD, g) + np.einsum('pt,tq->pq', FD, g).T) + 0.5 * 0.5 * (np.einsum('putv,qutv->pq', w, lam) + np.einsum('putv,qutv->pq', w, lam).T)
print("Lagrangian cumulant form F = sym(FD gamma + 1/2 w lam):", abs(Fcum - F).max())
# ---------- pp:  M[p,q,r,s] = <[a_q a_p,[H,a_r+ a_s+]]>
Pdpsi = np.array([[Ad[r] @ (Ad[s] @ psi) for s in range(n)] for r in range(n)])        # a_r+ a_s+ |psi>
Papsi = np.array([[A[q] @ (A[p] @ psi) for p in range(n)] for q in range(n)])          # a_q a_p |psi>  index [q,p]
Cpppsi = np.einsum('ij,rsj->rsi', H, Pdpsi) - np.array([[Ad[r] @ (Ad[s] @ Hpsi) for s in range(n)] for r in range(n)])
# bra operator a_q a_p, its adjoint a_p+ a_q+ ; <[a_q a_p, X]> = (a_p+ a_q+ psi).(X psi) - (X^+ psi).(a_q a_p psi); [H,a_r+a_s+]^+ = -[H,a_s a_r]
Cpa = np.einsum('ij,rsj->rsi', H, np.transpose(Papsi,(1,0,2))) - np.array([[A[s] @ (A[r] @ Hpsi) for s in range(n)] for r in range(n)])  # [H, a_s a_r]|psi>, index [r,s]
Mpp = np.einsum('pqi,rsi->pqrs', Pdpsi, Cpppsi) + np.einsum('rsi,qpi->pqrs', Cpa, Papsi)
Spp = np.einsum('pqi,rsi->pqrs', Pdpsi, Pdpsi) - np.einsum('rsi,qpi->pqrs', np.transpose(Papsi,(1,0,2)), Papsi)
def PP(X):
    return X - np.transpose(X, (1, 0, 2, 3)) - np.transpose(X, (0, 1, 3, 2)) + np.transpose(X, (1, 0, 3, 2))
seed = (np.einsum('qs,pr->pqrs', I, FD) - np.einsum('qs,rp->pqrs', I, calF) - np.einsum('pr,qs->pqrs', h, g)
        + np.einsum('ptru,tsqu->pqrs', w, G))
Mppc = PP(seed) + w - (np.einsum('pt,tqrs->pqrs', g, w) + np.einsum('qt,ptrs->pqrs', g, w)
                       + np.einsum('rt,pqts->pqrs', g, w) + np.einsum('st,pqrt->pqrs', g, w))
print("pp plain double commutator: compact form vs brute force:", abs(Mppc - Mpp).max())
Sppc = (np.einsum('pr,qs->pqrs', I, I) - np.einsum('ps,qr->pqrs', I, I) - np.einsum('pr,qs->pqrs', I, g)
        + np.einsum('ps,qr->pqrs', I, g) + np.einsum('qr,ps->pqrs', I, g) - np.einsum('qs,pr->pqrs', I, g))
print("pp metric:", abs(Sppc - Spp).max())
Mppsym = 0.5 * (Mpp + np.transpose(Mpp, (2, 3, 0, 1)))
seed2 = (np.einsum('qs,pr->pqrs', I, FD - F) - np.einsum('qs,pr->pqrs', g, FD)
         + np.einsum('ptru,qt,su->pqrs', w, g, g) + np.einsum('ptru,tsqu->pqrs', w, lam))
Mppcum = PP(seed2) + w - (np.einsum('pt,tqrs->pqrs', g, w) + np.einsum('qt,ptrs->pqrs', g, w)
                          + np.einsum('rt,pqts->pqrs', g, w) + np.einsum('st,pqrt->pqrs', g, w))
print("pp symmetrized: cumulant form vs brute force:", abs(Mppcum - Mppsym).max())
seed3 = (np.einsum('qs,pr->pqrs', I, FD - F) - np.einsum('pr,qs->pqrs', h, g) + np.einsum('ptru,tsqu->pqrs', w, G))
Mppg = PP(seed3) + w - (np.einsum('pt,tqrs->pqrs', g, w) + np.einsum('qt,ptrs->pqrs', g, w)
                        + np.einsum('rt,pqts->pqrs', g, w) + np.einsum('st,pqrt->pqrs', g, w))
print("pp symmetrized: Gamma form vs brute force:", abs(Mppg - Mppsym).max())
# exact-eigenstate check: for the FCI ground state in the sector, M should be symmetric and calF symmetric
Hs = H[np.ix_(sec, sec)]
e, c = np.linalg.eigh(Hs)
psi0 = np.zeros(dim); psi0[sec] = c[:, 0]
g0, G0 = rdms(psi0, A)
calF0 = np.einsum('pt,tq->pq', h, g0) + np.einsum('ptuv,qutv->pq', V, G0)
print("eigenstate: Lagrangian asymmetry", abs(calF0 - calF0.T).max())
