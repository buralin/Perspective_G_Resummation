"""
Dyall (CAS) reference in a spin-orbital basis, built on PySCF.

Conventions
-----------
* Spin orbitals are labelled p = 2*p_spatial + sigma with sigma = 0 (alpha)
  and sigma = 1 (beta).
* Two-electron integrals are stored as

      V[p, r, q, s] = v_{pr,qs} = <pq|rs> = (pr|qs),

  so that the interaction reads 1/2 sum v_{pr,qs} p^+ q^+ s r (the
  convention of Wang, Fang, and Li, arXiv:2604.16013).  Antisymmetrized
  integrals are vbar_{pr,qs} = v_{pr,qs} - v_{ps,qr}.
* The Dyall partition H = H_D + V_R follows Appendix A of Wang, Fang, and
  Li: the inactive one-body part of H_D is the diagonal of the Dyall Fock
  operator F_PQ = h_PQ + <Pk||Qk> + <Px||Qy> gamma_xy (core and virtual
  blocks are canonicalized), the active part is h_eff + V^A.
* All active-space states are obtained by dense diagonalization of the
  Dyall active Hamiltonian in the N_a, N_a+1, N_a-1 sectors with the PySCF
  FCI machinery (direct_spin1).  Operators a_p, a_p^+ are applied with
  pyscf.fci.addons.{cre,des}_{a,b}.
"""
import numpy as np
from pyscf import gto, scf, mcscf, ao2mo, symm
from pyscf.fci import direct_spin1, cistring, addons

HARTREE2EV = 27.211386245988


# ----------------------------------------------------------------------------
# molecule / CASSCF helpers
# ----------------------------------------------------------------------------
def build_h4(r=1.0, basis='sto-6g', verbose=0):
    """Linear H4 with nearest-neighbour distance r (Angstrom)."""
    atom = [('H', (0.0, 0.0, k * r)) for k in range(4)]
    return gto.M(atom=atom, basis=basis, unit='Angstrom', verbose=verbose)


def run_casscf(mol, ncas=2, nelecas=2, verbose=0):
    """RHF followed by CASSCF(nelecas, ncas)."""
    mf = scf.RHF(mol)
    mf.verbose = verbose
    mf.kernel()
    mc = mcscf.CASSCF(mf, ncas, nelecas)
    mc.verbose = verbose
    mc.kernel()
    return mf, mc


def run_casci(mol, ncas, nelecas, verbose=0):
    """RHF followed by CASCI(nelecas, ncas) in the RHF orbitals around the
    Fermi level (the ozone setting of Wang, Fang, and Li)."""
    mf = scf.RHF(mol)
    mf.verbose = verbose
    mf.kernel()
    mc = mcscf.CASCI(mf, ncas, nelecas)
    mc.verbose = verbose
    mc.kernel()
    return mf, mc


def build_o3(basis='6-31g', r=1.278, theta=116.8, symmetry=True, verbose=0):
    """Ozone at the CCCBDB experimental geometry used by Wang, Fang, and Li
    (R(O-O) = 1.278 A, theta(O-O-O) = 116.8 deg).  The molecule lies in the
    yz plane with the C2 axis along z, so that the out-of-plane pi orbitals
    are b1 and the in-plane ones b2, as in their Table S3."""
    th = np.deg2rad(theta)
    y, z = r * np.sin(th / 2.0), r * np.cos(th / 2.0)
    atom = [('O', (0.0, 0.0, 0.0)), ('O', (0.0, y, -z)), ('O', (0.0, -y, -z))]
    return gto.M(atom=atom, basis=basis, unit='Angstrom', symmetry=symmetry,
                 verbose=verbose)


# ----------------------------------------------------------------------------
# spin-orbital integrals
# ----------------------------------------------------------------------------
def spin_orbital_one_body(h_mo):
    n = h_mo.shape[0]
    h_so = np.zeros((2 * n, 2 * n))
    for s in (0, 1):
        h_so[s::2, s::2] = h_mo
    return h_so


def spin_orbital_two_body(eri_mo):
    """eri_mo[p,r,q,s] = (pr|qs) (chemist) -> V[p,r,q,s] with spin deltas."""
    n = eri_mo.shape[0]
    V = np.zeros((2 * n,) * 4)
    for s1 in (0, 1):
        for s2 in (0, 1):
            V[s1::2, s1::2, s2::2, s2::2] = eri_mo
    return V


def antisymmetrize(V):
    """vbar_{pr,qs} = v_{pr,qs} - v_{ps,qr}."""
    return V - V.transpose(0, 3, 2, 1)


def _canonicalize_block(Fb, symb):
    """Eigenvectors of a Fock block sorted by energy.  If irrep labels are
    given the block is diagonalized within each irrep, so that the labels
    survive exactly.  Returns (U, labels_sorted)."""
    n = Fb.shape[0]
    if symb is None:
        _, U = np.linalg.eigh(Fb)
        return U, None
    e = np.zeros(n)
    U = np.zeros((n, n))
    for ir in np.unique(symb):
        idx = np.where(symb == ir)[0]
        ei, Ui = np.linalg.eigh(Fb[np.ix_(idx, idx)])
        e[idx] = ei
        U[np.ix_(idx, idx)] = Ui
    order = np.argsort(e)
    return U[:, order], symb[order]


# ----------------------------------------------------------------------------
# CI helpers (PySCF determinant machinery)
# ----------------------------------------------------------------------------
def dense_ci_hamiltonian(h1, eri, norb, nelec):
    """Dense Hamiltonian matrix in the (na, nb) determinant space."""
    na, nb = nelec
    dim_a = cistring.num_strings(norb, na)
    dim_b = cistring.num_strings(norb, nb)
    dim = dim_a * dim_b
    if na + nb == 0:
        return np.zeros((1, 1))
    h2e = direct_spin1.absorb_h1e(h1, eri, norb, nelec, 0.5)
    H = np.empty((dim, dim))
    for k in range(dim):
        e = np.zeros(dim)
        e[k] = 1.0
        H[:, k] = direct_spin1.contract_2e(
            h2e, e.reshape(dim_a, dim_b), norb, nelec).ravel()
    return 0.5 * (H + H.T)


def apply_hamiltonian(h1, eri, norb, nelec, vec):
    """H |vec> for a CI vector in the (na, nb) sector."""
    if sum(nelec) == 0:
        return np.zeros_like(vec)
    h2e = direct_spin1.absorb_h1e(h1, eri, norb, nelec, 0.5)
    return direct_spin1.contract_2e(h2e, vec, norb, nelec)


def apply_cre(vec, norb, nelec, p):
    """a_p^+ |vec>, p a spin-orbital index (2*spatial + spin).

    Returns (new_vec, new_nelec) or (None, None) if the target sector is empty.
    """
    ps, s = divmod(int(p), 2)
    na, nb = nelec
    if s == 0:
        if na >= norb:
            return None, None
        return addons.cre_a(vec, norb, (na, nb), ps), (na + 1, nb)
    if nb >= norb:
        return None, None
    return addons.cre_b(vec, norb, (na, nb), ps), (na, nb + 1)


def apply_des(vec, norb, nelec, p):
    """a_p |vec>, p a spin-orbital index (2*spatial + spin)."""
    ps, s = divmod(int(p), 2)
    na, nb = nelec
    if s == 0:
        if na == 0:
            return None, None
        return addons.des_a(vec, norb, (na, nb), ps), (na - 1, nb)
    if nb == 0:
        return None, None
    return addons.des_b(vec, norb, (na, nb), ps), (na, nb - 1)


class Sector:
    """All eigenstates of a Hamiltonian in one (na, nb) determinant sector."""

    def __init__(self, nelec, energies, vectors, norb):
        self.nelec = (int(nelec[0]), int(nelec[1]))
        self.E = np.asarray(energies)
        self.V = np.asarray(vectors)
        self.dim_a = cistring.num_strings(norb, self.nelec[0])
        self.dim_b = cistring.num_strings(norb, self.nelec[1])

    @property
    def nstates(self):
        return self.E.size

    def vec(self, k):
        return self.V[:, k].reshape(self.dim_a, self.dim_b)


class ActivePole:
    """One charged pole (mu, +/-) of the active-space Dyall Green's function.

    sign  : +1 (N_a+1 state) or -1 (N_a-1 state)
    sector: (na, nb) of the active state
    kappa : signed pole energy, +(E_mu^{N+1}-E_0) or -(E_mu^{N-1}-E_0)
    vec   : active-space CI vector of the state
    d     : transition vector over local active spin orbitals,
            [d]_x = <Xi_0|x|Xi_mu^+>  (attachment) or  <Xi_mu^-|x|Xi_0> (removal)
    """
    __slots__ = ('sign', 'sector', 'state', 'kappa', 'vec', 'd')

    def __init__(self, sign, sector, state, kappa, vec, d):
        self.sign = int(sign)
        self.sector = tuple(sector)
        self.state = int(state)
        self.kappa = float(kappa)
        self.vec = vec
        self.d = d


# ----------------------------------------------------------------------------
# the Dyall reference
# ----------------------------------------------------------------------------
class DyallReference:
    """Dyall/CAS reference: orbitals, integrals, active poles, transition tensors.

    Parameters
    ----------
    mc : converged pyscf CASSCF or CASCI object (single state).
    """

    def __init__(self, mc, verbose=True):
        self.mc = mc
        self.mol = mc.mol
        self.mf = mc._scf
        self.verbose = verbose
        mol, mf = self.mol, self.mf

        ncore, ncas = int(mc.ncore), int(mc.ncas)
        nelecas = mc.nelecas
        if isinstance(nelecas, (int, np.integer)):
            nelecas = ((int(nelecas) + mol.spin) // 2, (int(nelecas) - mol.spin) // 2)
        nelecas = (int(nelecas[0]), int(nelecas[1]))
        nocc = ncore + ncas
        C = np.array(mc.mo_coeff, dtype=float, copy=True)
        nmo = C.shape[1]
        self.ncore, self.ncas, self.nocc, self.nmo = ncore, ncas, nocc, nmo
        self.nelecas = nelecas

        # ---- orbital symmetry labels (if the molecule has symmetry) ----
        orbsym = getattr(mc.mo_coeff, 'orbsym', None)
        if orbsym is None and getattr(mol, 'symmetry', False):
            try:
                orbsym = scf.hf_symm.get_orbsym(mol, C)
            except Exception:
                orbsym = None
        orbsym = None if orbsym is None else np.array(orbsym, dtype=int)
        self.groupname = getattr(mol, 'groupname', None) if orbsym is not None else None

        # ---- Dyall Fock operator and canonicalization of core / virtual ----
        casdm1 = mc.fcisolver.make_rdm1(mc.ci, ncas, nelecas)
        dm_mo = np.zeros((nmo, nmo))
        dm_mo[:ncore, :ncore] = 2.0 * np.eye(ncore)
        dm_mo[ncore:nocc, ncore:nocc] = casdm1
        dm_ao = C @ dm_mo @ C.T
        hcore_ao = mf.get_hcore()
        F_ao = hcore_ao + mf.get_veff(mol, dm_ao)
        F_mo = C.T @ F_ao @ C
        if ncore > 0:
            U, lab = _canonicalize_block(F_mo[:ncore, :ncore],
                                         None if orbsym is None else orbsym[:ncore])
            C[:, :ncore] = C[:, :ncore] @ U
            if orbsym is not None:
                orbsym[:ncore] = lab
        if nmo > nocc:
            U, lab = _canonicalize_block(F_mo[nocc:, nocc:],
                                         None if orbsym is None else orbsym[nocc:])
            C[:, nocc:] = C[:, nocc:] @ U
            if orbsym is not None:
                orbsym[nocc:] = lab
        self.orbsym = orbsym
        F_mo = C.T @ F_ao @ C
        self.mo_coeff = C
        self.F_mo = F_mo
        self.h_mo = C.T @ hcore_ao @ C
        self.eri_mo = ao2mo.restore(1, ao2mo.kernel(mol, C), nmo)

        act = slice(ncore, nocc)
        core = slice(0, ncore)
        h_eff = self.h_mo[act, act].copy()
        if ncore > 0:
            h_eff += (2.0 * np.einsum('xykk->xy', self.eri_mo[act, act, core, core])
                      - np.einsum('xkky->xy', self.eri_mo[act, core, core, act]))
        self.h_eff = h_eff
        self.eri_cas = np.ascontiguousarray(self.eri_mo[act, act, act, act])

        # spatial one- and two-body parts of H_D (used for exact checks)
        h1_D = np.zeros((nmo, nmo))
        inact_sp = list(range(ncore)) + list(range(nocc, nmo))
        h1_D[inact_sp, inact_sp] = np.diag(F_mo)[inact_sp]
        h1_D[act, act] = h_eff
        eri_D = np.zeros_like(self.eri_mo)
        eri_D[act, act, act, act] = self.eri_cas
        self.h1_D, self.eri_D = h1_D, eri_D

        # ---- spin-orbital quantities ----
        nso = 2 * nmo
        self.nso = nso
        self.h_so = spin_orbital_one_body(self.h_mo)
        self.V = spin_orbital_two_body(self.eri_mo)
        self.core_so = [2 * i + s for i in range(ncore) for s in (0, 1)]
        self.act_so = [2 * x + s for x in range(ncore, nocc) for s in (0, 1)]
        self.virt_so = [2 * a + s for a in range(nocc, nmo) for s in (0, 1)]
        self.inact_so = self.core_so + self.virt_so
        self.nact_so = 2 * ncas
        is_act = np.zeros(nso, dtype=bool)
        is_act[self.act_so] = True
        self.is_act = is_act
        mask = (is_act[:, None, None, None] & is_act[None, :, None, None]
                & is_act[None, None, :, None] & is_act[None, None, None, :])
        self.VR = self.V * (~mask)
        self._Vbar = None                                # built on demand
        self._rdms = None                                # spin-orbital RDMs, on demand
        self.VbarR = antisymmetrize(self.VR)
        self.eps = np.repeat(np.diag(F_mo), 2)          # eps[2p+s] = F_pp
        self.hD_so = spin_orbital_one_body(h1_D)
        self.u_so = self.h_so - self.hD_so

        # ---- active-space sectors ----
        na, nb = nelecas
        attach, remove = [], []
        if na + 1 <= ncas:
            attach.append((na + 1, nb))
        if nb + 1 <= ncas:
            attach.append((na, nb + 1))
        if na >= 1:
            remove.append((na - 1, nb))
        if nb >= 1:
            remove.append((na, nb - 1))
        self.attach_sectors, self.remove_sectors = attach, remove
        self.sectors = {}
        for nel in [nelecas] + attach + remove:
            if nel in self.sectors:
                continue
            H = dense_ci_hamiltonian(h_eff, self.eri_cas, ncas, nel)
            E, U = np.linalg.eigh(H)
            self.sectors[nel] = Sector(nel, E, U, ncas)
        neutral = self.sectors[nelecas]
        self.E0 = float(neutral.E[0])
        self.xi0 = neutral.vec(0)
        ci_ref = np.asarray(mc.ci)
        self.overlap_with_mc_ci = float(abs(np.vdot(self.xi0, ci_ref)))

        # ---- charged poles ----
        nact_so = self.nact_so
        self._cre_xi0 = [apply_cre(self.xi0, ncas, nelecas, x) for x in range(nact_so)]
        self._des_xi0 = [apply_des(self.xi0, ncas, nelecas, x) for x in range(nact_so)]
        poles = []
        for sign, secs in ((+1, attach), (-1, remove)):
            ops = self._cre_xi0 if sign > 0 else self._des_xi0
            for nel in secs:
                sec = self.sectors[nel]
                for k in range(sec.nstates):
                    vec = sec.vec(k)
                    d = np.zeros(nact_so)
                    for x in range(nact_so):
                        w, nw = ops[x]
                        if w is None or nw != nel:
                            continue
                        d[x] = np.vdot(w, vec) if sign > 0 else np.vdot(vec, w)
                    poles.append(ActivePole(sign, nel, k, sign * (sec.E[k] - self.E0), vec, d))
        self.poles = poles
        self.npoles = len(poles)
        self.kappa = np.array([p.kappa for p in poles])
        self.pole_sign = np.array([p.sign for p in poles])
        self.d_act = np.array([p.d for p in poles]).T      # (nact_so, npoles)

        # ---- one-body density and static self-energy ----
        gamma_act = np.zeros((nact_so, nact_so))
        for x in range(nact_so):
            wx, nx = self._des_xi0[x]
            if wx is None:
                continue
            for y in range(nact_so):
                wy, ny = self._des_xi0[y]
                if wy is None or ny != nx:
                    continue
                gamma_act[x, y] = np.vdot(wx, wy)         # <Xi0| x^+ y |Xi0>
        self.gamma_act = gamma_act
        gamma_so = np.zeros((nso, nso))
        gamma_so[self.core_so, self.core_so] = 1.0
        gamma_so[np.ix_(self.act_so, self.act_so)] = gamma_act
        self.gamma_so = gamma_so
        # Sigma_1^{11} = u + vbar^R gamma  (Wang, Fang, Li, Eq. (20))
        self.Sigma_stat = self.u_so + np.einsum('pqrs,rs->pq', self.VbarR, gamma_so)

        # ---- three-operator transition amplitudes ----
        self.C3 = self._three_operator_amplitudes()
        # connected tensors C^{D,+/-}_{alpha; z, yw} (Dyall--Hedin notes,
        # "Screened mixed residues")
        self.Cc = 0.5 * self.C3 - np.einsum('yw,za->azyw', gamma_act, self.d_act)

        if verbose:
            self.summary()

    # ------------------------------------------------------------------
    def _three_operator_amplitudes(self):
        """C3[alpha, z, y, w] = <Xi0|y^+ w z|Xi_alpha^+>  (attachment poles)
                             = <Xi_alpha^-|y^+ w z|Xi0>  (removal poles)."""
        ncas, nact_so = self.ncas, self.nact_so
        C3 = np.zeros((self.npoles, nact_so, nact_so, nact_so))
        att = [(ia, p) for ia, p in enumerate(self.poles) if p.sign > 0]
        rem = [(ia, p) for ia, p in enumerate(self.poles) if p.sign < 0]
        # attachment: <Xi0| y^+ w z |Xi_mu> = < (z^+ w^+ y) Xi0 | Xi_mu >
        for y in range(nact_so):
            v1, n1 = self._des_xi0[y]
            if v1 is None:
                continue
            for w in range(nact_so):
                v2, n2 = apply_cre(v1, ncas, n1, w)
                if v2 is None:
                    continue
                for z in range(nact_so):
                    v3, n3 = apply_cre(v2, ncas, n2, z)
                    if v3 is None:
                        continue
                    for ia, p in att:
                        if p.sector == n3:
                            C3[ia, z, y, w] = np.vdot(v3, p.vec)
        # removal: <Xi_mu| y^+ w z |Xi0>
        for z in range(nact_so):
            u1, m1 = self._des_xi0[z]
            if u1 is None:
                continue
            for w in range(nact_so):
                u2, m2 = apply_des(u1, ncas, m1, w)
                if u2 is None:
                    continue
                for y in range(nact_so):
                    u3, m3 = apply_cre(u2, ncas, m2, y)
                    if u3 is None:
                        continue
                    for ia, p in rem:
                        if p.sector == m3:
                            C3[ia, z, y, w] = np.vdot(p.vec, u3)
        return C3

    # ------------------------------------------------------------------
    def neutral_transition_densities(self):
        """Excitation energies and rho[mu, x, y] = <Xi_mu|x^+ y|Xi_0>, mu >= 1
        (local active spin-orbital indices)."""
        sec = self.sectors[self.nelecas]
        nact_so = self.nact_so
        nexc = sec.nstates - 1
        rho = np.zeros((nexc, nact_so, nact_so))
        vecs = [sec.vec(k) for k in range(1, sec.nstates)]
        for y in range(nact_so):
            v1, n1 = self._des_xi0[y]
            if v1 is None:
                continue
            for x in range(nact_so):
                v2, n2 = apply_cre(v1, self.ncas, n1, x)
                if v2 is None or n2 != self.nelecas:
                    continue
                for k, vk in enumerate(vecs):
                    rho[k, x, y] = np.vdot(vk, v2)
        return sec.E[1:] - self.E0, rho

    # ------------------------------------------------------------------
    def dyall_greens_function(self, z):
        """Retarded Dyall Green's function G_D(z) in the spin-orbital basis."""
        T_D, K_D = self.pole_representation()
        return T_D @ ((1.0 / (z - K_D))[:, None] * T_D.T)

    def pole_representation(self):
        """T_D (nso x npoles_total) and signed pole energies K_D of G_D."""
        nI = len(self.inact_so)
        T_D = np.zeros((self.nso, nI + self.npoles))
        T_D[self.inact_so, np.arange(nI)] = 1.0
        T_D[np.ix_(self.act_so, nI + np.arange(self.npoles))] = self.d_act
        K_D = np.concatenate([self.eps[self.inact_so], self.kappa])
        return T_D, K_D

    # ------------------------------------------------------------------
    def rdms(self, order=2):
        """Spin-orbital RDMs D_1..D_order of |Xi_0> over the active spin
        orbitals (see rdm.py for the conventions); cached."""
        from .rdm import spin_orbital_rdms
        if self._rdms is None or max(self._rdms) < order:
            self._rdms = spin_orbital_rdms(self.xi0, self.ncas, self.nelecas, order)
        return {k: v for k, v in self._rdms.items() if k <= order}

    # ------------------------------------------------------------------
    @property
    def Vbar(self):
        """Fully antisymmetrized integrals (all elements, including active)."""
        if self._Vbar is None:
            self._Vbar = antisymmetrize(self.V)
        return self._Vbar

    # ------------------------------------------------------------------
    # symmetry labels
    def irrep_name(self, sym):
        if sym is None or self.groupname is None:
            return '?'
        return symm.irrep_id2name(self.groupname, int(sym))

    def orbital_irrep(self, p_spatial):
        return None if self.orbsym is None else int(self.orbsym[p_spatial])

    def state_symmetry(self, vec, nelec):
        """Irrep id (PySCF convention, product = XOR) of an active-space CI
        vector, taken from its dominant determinant."""
        if self.orbsym is None:
            return None
        osym = self.orbsym[self.ncore:self.nocc]
        vec = np.asarray(vec)
        ia, ib = np.unravel_index(int(np.argmax(np.abs(vec))), vec.shape)
        sym = 0
        for nel, addr in ((nelec[0], ia), (nelec[1], ib)):
            if nel == 0:
                continue
            s = cistring.addr2str(self.ncas, nel, addr)
            for x in range(self.ncas):
                if (s >> x) & 1:
                    sym ^= int(osym[x])
        return sym

    def pole_symmetry(self, ia):
        p = self.poles[ia]
        return self.state_symmetry(p.vec, p.sector)

    # ------------------------------------------------------------------
    def so_label(self, p):
        ps, s = divmod(int(p), 2)
        kind = 'core' if ps < self.ncore else ('act' if ps < self.nocc else 'virt')
        return f"{ps}{'ab'[s]}({kind})"

    def summary(self):
        print("Dyall reference")
        print(f"  nmo = {self.nmo}, ncore = {self.ncore}, ncas = {self.ncas}, "
              f"nelecas = {self.nelecas}, nso = {self.nso}"
              + (f", point group {self.groupname}" if self.groupname else ""))
        print(f"  active sectors: " + ", ".join(
            f"{k}: {v.nstates} states" for k, v in self.sectors.items()))
        print(f"  charged active poles: {self.npoles} "
              f"({np.sum(self.pole_sign > 0)} attachment, {np.sum(self.pole_sign < 0)} removal)")
        print(f"  |<Xi0|mc.ci>| = {self.overlap_with_mc_ci:.12f}")
        print(f"  inactive orbital energies (eV): "
              + ", ".join(f"{self.eps[p] * HARTREE2EV:.3f}" for p in self.inact_so[::2]))
        att = np.sort(self.kappa[self.pole_sign > 0])
        rem = np.sort(-self.kappa[self.pole_sign < 0])
        print(f"  lowest active attachment energies (eV): "
              + ", ".join(f"{e * HARTREE2EV:.3f}" for e in att[:4]))
        print(f"  lowest active ionization energies (eV): "
              + ", ".join(f"{e * HARTREE2EV:.3f}" for e in rem[:4]))
