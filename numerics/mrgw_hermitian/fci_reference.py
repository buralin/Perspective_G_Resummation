"""
Exact (full CI) one-particle Green's function for small systems, and two
independent checks of the Hermitian pole-space construction:

* check_same_sector_couplings: the same-sector blocks of K_top (bare kernel,
  no bath) must equal the Hamiltonian matrix elements between the charged
  Dyall states  a_a^+ |Psi_0>, |Phi_I>|Xi_mu^{N_a+1}>  (attachment) and
  a_i |Psi_0>, |Phi_I>|Xi_mu^{N_a-1}>  (removal, with a minus sign), because
  the Dyall reference makes these states orthonormal.
* check_first_order_gf: d/d lambda of the exact Green's function of
  H_D + lambda V_R at lambda = 0 (finite differences on the full-CI Green's
  function) must equal T_D R_D K^(1) R_D T_D^+, i.e. the first-order content
  of the Hermitian matrix including the cross-sector couplings.

All CI vectors are handled with the PySCF determinant machinery.
"""
import numpy as np
from pyscf.fci import cistring

from .dyall import (dense_ci_hamiltonian, apply_hamiltonian, apply_cre,
                    apply_des, Sector, HARTREE2EV)


class FCIReference:
    """Full-CI Green's function in the (canonicalized) MO basis of a
    DyallReference.  Optional (h1, eri) override the physical integrals,
    which is used for the Dyall Hamiltonian and for H_D + lambda V_R."""

    def __init__(self, ref, h1=None, eri=None, verbose=True, label='FCI'):
        self.ref = ref
        self.label = label
        norb = ref.nmo
        nelec = tuple(int(x) for x in ref.mol.nelec)
        h1 = ref.h_mo if h1 is None else h1
        eri = ref.eri_mo if eri is None else eri
        self.h1, self.eri, self.norb, self.nelec = h1, eri, norb, nelec
        na, nb = nelec
        attach, remove = [], []
        if na + 1 <= norb:
            attach.append((na + 1, nb))
        if nb + 1 <= norb:
            attach.append((na, nb + 1))
        if na >= 1:
            remove.append((na - 1, nb))
        if nb >= 1:
            remove.append((na, nb - 1))
        self.sectors = {}
        for nel in [nelec] + attach + remove:
            if nel in self.sectors:
                continue
            H = dense_ci_hamiltonian(h1, eri, norb, nel)
            E, U = np.linalg.eigh(H)
            self.sectors[nel] = Sector(nel, E, U, norb)
        neutral = self.sectors[nelec]
        self.E0 = float(neutral.E[0])
        self.psi0 = neutral.vec(0)
        nso = 2 * norb
        self.nso = nso
        cre0 = [apply_cre(self.psi0, norb, nelec, p) for p in range(nso)]
        des0 = [apply_des(self.psi0, norb, nelec, p) for p in range(nso)]
        K, T = [], []
        for sign, secs in ((+1, attach), (-1, remove)):
            ops = cre0 if sign > 0 else des0
            for nel in secs:
                sec = self.sectors[nel]
                for k in range(sec.nstates):
                    vec = sec.vec(k)
                    t = np.zeros(nso)
                    for p in range(nso):
                        w, nw = ops[p]
                        if w is None or nw != nel:
                            continue
                        t[p] = np.vdot(w, vec) if sign > 0 else np.vdot(vec, w)
                    K.append(sign * (sec.E[k] - self.E0))
                    T.append(t)
        self.K = np.array(K)
        self.T = np.array(T).T                       # (nso, npoles)
        self.weights = np.sum(self.T ** 2, axis=0)
        if verbose:
            print(f"{label}: E0 = {self.E0:.10f} Ha (electronic), "
                  f"{self.K.size} charged poles, sum rule |T T^T - 1| = "
                  f"{np.abs(self.T @ self.T.T - np.eye(nso)).max():.2e}")

    def greens_function(self, z):
        return self.T @ ((1.0 / (z - self.K))[:, None] * self.T.T)

    def spectral_function(self, omegas, eta, trace=True):
        omegas = np.asarray(omegas, dtype=float)
        lor = (eta / np.pi) / ((omegas[:, None] - self.K[None, :]) ** 2 + eta ** 2)
        if trace:
            return lor @ self.weights
        return np.einsum('wl,pl,ql->wpq', lor, self.T, self.T)

    def moments(self):
        return self.T @ self.T.T, self.T @ (self.K[:, None] * self.T.T)

    def principal_poles(self, nmax=4, wmin=1e-3):
        """Most intense removal and attachment poles (energy, weight)."""
        out = {}
        for name, sel in (('remove', self.K < 0), ('attach', self.K > 0)):
            idx = np.where(sel & (self.weights > wmin))[0]
            idx = idx[np.argsort(-self.weights[idx])][:nmax]
            out[name] = [(self.K[i], self.weights[i]) for i in idx]
        return out


# ----------------------------------------------------------------------------
def embed_active_vector(ref, vec_act, nelec_act):
    """Embed an active-space CI vector (ncas orbitals) into the full orbital
    space: core orbitals doubly occupied, virtual orbitals empty.  Returns
    (full_vec, nelec_full)."""
    ncore, ncas, nmo = ref.ncore, ref.ncas, ref.nmo
    na_act, nb_act = nelec_act
    na, nb = na_act + ncore, nb_act + ncore
    core_bits = (1 << ncore) - 1
    stra = cistring.make_strings(range(ncas), na_act)
    strb = cistring.make_strings(range(ncas), nb_act)
    addr_a = [cistring.str2addr(nmo, na, core_bits | (int(s) << ncore)) for s in stra]
    addr_b = [cistring.str2addr(nmo, nb, core_bits | (int(s) << ncore)) for s in strb]
    full = np.zeros((cistring.num_strings(nmo, na), cistring.num_strings(nmo, nb)))
    full[np.ix_(addr_a, addr_b)] = np.asarray(vec_act).reshape(len(addr_a), len(addr_b))
    return full, (na, nb)


def check_same_sector_couplings(ref, verbose=True):
    """Compare the same-sector blocks of K_top (bare, no bath) with exact
    Hamiltonian matrix elements between charged Dyall states.

    Returns a dict of maximal absolute deviations (Hartree)."""
    from .hermitian import HermitianGF
    gf = HermitianGF(ref, None, mixed='bare', bath=False)
    K, nI = gf.K, gf.nI
    nmo, h1, eri = ref.nmo, ref.h_mo, ref.eri_mo
    psi0, nel0 = embed_active_vector(ref, ref.xi0, ref.nelecas)
    E0H = float(np.vdot(psi0, apply_hamiltonian(h1, eri, nmo, nel0, psi0)))
    inact = ref.inact_so
    virt = set(ref.virt_so)
    # charged inactive states a_a^+ psi0 (virtual a) and a_i psi0 (core i)
    inact_states = []
    for P in inact:
        if P in virt:
            w, nw = apply_cre(psi0, nmo, nel0, P)
        else:
            w, nw = apply_des(psi0, nmo, nel0, P)
        inact_states.append((w, nw))
    # embedded active poles
    act_states = []
    for p in ref.poles:
        full, nel = embed_active_vector(ref, p.vec, p.sector)
        act_states.append((full, nel, apply_hamiltonian(h1, eri, nmo, nel, full)))

    dev = {}
    # active-active block: <Psi_alpha|H|Psi_beta> - E0 delta = sign * kappa delta
    err = 0.0
    for a, (fa, nela, Hfa) in enumerate(act_states):
        for b, (fb, nelb, Hfb) in enumerate(act_states):
            if nela != nelb:
                continue
            val = np.vdot(fa, Hfb) - (E0H if a == b else 0.0)
            expect = ref.pole_sign[a] * K[nI + a, nI + b]
            err = max(err, abs(val - expect))
    dev['active-active'] = err
    # inactive-inactive block
    err = 0.0
    for iP, (wP, nP) in enumerate(inact_states):
        HwP = apply_hamiltonian(h1, eri, nmo, nP, wP)
        for iQ, (wQ, nQ) in enumerate(inact_states):
            if nP != nQ:
                continue
            val = np.vdot(wQ, HwP) - (E0H if iP == iQ else 0.0)
            sgn = +1.0 if inact[iP] in virt else -1.0
            err = max(err, abs(val - sgn * K[iQ, iP]))
    dev['inactive-inactive'] = err
    # inactive-active same-sector couplings, up to one common sign per pole
    err, err_abs = 0.0, 0.0
    ncmp = 0
    for a, (fa, nela, Hfa) in enumerate(act_states):
        sgn = float(ref.pole_sign[a])
        hcol, kcol = [], []
        for iP, (wP, nP) in enumerate(inact_states):
            if nP != nela:
                continue
            hcol.append(sgn * np.vdot(wP, Hfa))
            kcol.append(K[iP, nI + a])
        if not hcol:
            continue
        hcol, kcol = np.array(hcol), np.array(kcol)
        ncmp += hcol.size
        s = np.sign(np.dot(hcol, kcol)) or 1.0
        err = max(err, np.abs(hcol - s * kcol).max())
        err_abs = max(err_abs, np.abs(np.abs(hcol) - np.abs(kcol)).max())
    dev['inactive-active (same sector, common sign per pole)'] = err
    dev['inactive-active (absolute values)'] = err_abs
    dev['n_couplings_compared'] = ncmp
    if verbose:
        print("Same-sector coupling check (bare kernel, no bath) against exact "
              "Hamiltonian matrix elements between charged Dyall states:")
        for k, v in dev.items():
            if k.startswith('n_'):
                print(f"  {k}: {v}")
            else:
                print(f"  max deviation {k}: {v:.3e} Ha")
    return dev


def check_superoperator_propagator(ref, zs=None, verbose=True):
    """Numerical check of the superoperator form of the retarded propagator,

        G_pq(z) = (a_p^+ | (z - Hsuper)^{-1} | a_q^+),
        (X|Y) = <0|[X^+, Y]_+|0>,   Hsuper Y = [H, Y],

    against the Lehmann representation.  The resolvent is evaluated
    explicitly: a_q^+ is expanded in the eigenoperators |m><n| of Hsuper over
    all pairs of exact eigenstates of adjacent particle-number sectors
    (eigenvalue E_m - E_n), and the binary product with a_p^+ is formed with
    the anticommutator.  Returns the maximal deviation over p, q and z."""
    if zs is None:
        zs = [w + 0.05j for w in (-0.8, -0.3, 0.2, 0.7)]
    fci = FCIReference(ref, verbose=False)
    norb, nelec, nso = fci.norb, fci.nelec, fci.nso
    sec0 = fci.sectors[nelec]
    psi0 = fci.psi0
    E0 = fci.E0
    err = 0.0
    for z in zs:
        G_lehmann = fci.greens_function(z)
        G_super = np.zeros((nso, nso), dtype=complex)
        for q in range(nso):
            # block of a_q^+ from the neutral sector to the attachment sector:
            # A[m, n] = <m| a_q^+ |n>, and the resolvent factor 1/(z - (E_m - E_n))
            w, secA = apply_cre(psi0, norb, nelec, q)
            if w is None:
                continue
            SA = fci.sectors[secA]
            # Y|0> = sum_m <m|a_q^+|0>/(z - (E_m - E0)) |m>   (all m of the attachment sector)
            Y0 = np.zeros(SA.nstates, dtype=complex)
            for m in range(SA.nstates):
                Y0[m] = np.vdot(SA.vec(m), w) / (z - (SA.E[m] - E0))
            # block from the removal sector to the neutral sector: <m|a_q^+|n>, n in removal sector
            for secR in fci.sectors:
                if sum(secR) != sum(nelec) - 1:
                    continue
                SR = fci.sectors[secR]
                # only the removal sector that a_q^+ maps into the neutral sector contributes
                test, sec_test = apply_cre(SR.vec(0), norb, secR, q)
                if test is None or sec_test != nelec:
                    continue
                Ablock = np.zeros((sec0.nstates, SR.nstates))
                for n in range(SR.nstates):
                    v, _ = apply_cre(SR.vec(n), norb, secR, q)
                    for m in range(sec0.nstates):
                        Ablock[m, n] = np.vdot(sec0.vec(m), v)
                for p in range(nso):
                    # <0| a_p Y |0>  with Y|0> in the attachment sector
                    wp, secp = apply_cre(psi0, norb, nelec, p)
                    term1 = 0.0
                    if wp is not None and secp == secA:
                        term1 = sum(np.vdot(wp, SA.vec(m)) * Y0[m] for m in range(SA.nstates))
                    # <0| Y a_p |0>  with a_p|0> in the removal sector secR
                    vp, secvp = apply_des(psi0, norb, nelec, p)
                    term2 = 0.0
                    if vp is not None and secvp == secR:
                        c = np.array([np.vdot(SR.vec(n), vp) for n in range(SR.nstates)])
                        Ymn = Ablock / (z - (sec0.E[:, None] - SR.E[None, :]))
                        term2 = (Ymn @ c)[0]          # component on |0> (m = 0)
                    G_super[p, q] += term1 + term2
        err = max(err, np.abs(G_super - G_lehmann).max())
        if verbose:
            print(f"  z = {z.real:+.2f}{z.imag:+.2f}i : max|G_super - G_Lehmann| = "
                  f"{np.abs(G_super - G_lehmann).max():.2e}  (max|G| = {np.abs(G_lehmann).max():.2e})")
    return err


def check_first_order_gf(ref, zs=None, lam=1e-3, verbose=True):
    """Finite-difference check of the first-order Green's function.

    G_exact(lambda) is the full-CI Green's function of H_D + lambda V_R.
    Its derivative at lambda = 0 is compared with T_D R_D K^(1) R_D T_D^+,
    where K^(1) = T_D^+ Sigma_1^{11} T_D + K_B^(1) (bare kernel, all pole
    pairs including cross-sector ones)."""
    from .hermitian import HermitianGF
    if zs is None:
        zs = [w + 0.05j for w in (-1.2, -0.6, -0.2, 0.2, 0.6, 1.2)]
    dh = ref.h_mo - ref.h1_D
    deri = ref.eri_mo - ref.eri_D
    fci0 = FCIReference(ref, ref.h1_D, ref.eri_D, verbose=False, label='FCI(H_D)')
    fcip = FCIReference(ref, ref.h1_D + lam * dh, ref.eri_D + lam * deri, verbose=False)
    fcim = FCIReference(ref, ref.h1_D - lam * dh, ref.eri_D - lam * deri, verbose=False)
    gf1 = HermitianGF(ref, None, mixed='bare', bath=False)
    T_D, K_D = gf1.T_D, gf1.K_D
    K1 = gf1.K - np.diag(K_D)
    out = []
    for z in zs:
        G_D_fci = fci0.greens_function(z)
        G_D_pole = ref.dyall_greens_function(z)
        e0 = np.abs(G_D_fci - G_D_pole).max()
        dG_fd = (fcip.greens_function(z) - fcim.greens_function(z)) / (2.0 * lam)
        R = 1.0 / (z - K_D)
        dG_h = T_D @ (R[:, None] * K1 * R[None, :]) @ T_D.T
        scale = np.abs(dG_fd).max()
        e1 = np.abs(dG_fd - dG_h).max()
        out.append((z, e0, e1, scale))
        if verbose:
            print(f"  z = {z.real:+.2f}{z.imag:+.2f}i : |G_D(FCI) - G_D(poles)| = {e0:.2e}, "
                  f"|dG/dlam (FD) - dG/dlam (Hermitian)| = {e1:.2e}  (max|dG/dlam| = {scale:.2e})")
    return out
