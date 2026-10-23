"""Brute-force (full determinant space) treatment of internally contracted doubles
classes on top of a CASSCF vacuum: exact overlaps, exact Hamiltonian blocks,
second-order (Hylleraas with full H in the doubles block) and class-CI energies."""
import numpy as np, itertools, sys
from pyscf import gto, scf, mcscf, fci, ao2mo, lib
from pyscf.fci import cistring, addons

def full_space_setup(mol, mc):
    nmo = mc.mo_coeff.shape[1]; nelec = (mol.nelectron // 2, mol.nelectron // 2)
    h1 = mc.mo_coeff.T @ scf.hf.get_hcore(mol) @ mc.mo_coeff
    eri = ao2mo.restore(1, ao2mo.full(mol, mc.mo_coeff), nmo)
    h2e = fci.direct_spin1.absorb_h1e(h1, eri, nmo, nelec, 0.5)
    # embed CAS vector into the full space
    ncore, ncas = mc.ncore, mc.ncas
    na = cistring.num_strings(nmo, nelec[0]); nb = cistring.num_strings(nmo, nelec[1])
    ci_full = np.zeros((na, nb))
    cas_na = cistring.num_strings(ncas, mc.nelecas[0]); cas_nb = cistring.num_strings(ncas, mc.nelecas[1])
    stra = cistring.gen_strings4orblist(list(range(ncas)), mc.nelecas[0])
    strb = cistring.gen_strings4orblist(list(range(ncas)), mc.nelecas[1])
    core_bits = (1 << ncore) - 1
    for ia, sa in enumerate(stra):
        fa = cistring.str2addr(nmo, nelec[0], (int(sa) << ncore) | core_bits)
        for ib, sb in enumerate(strb):
            fb = cistring.str2addr(nmo, nelec[1], (int(sb) << ncore) | core_bits)
            ci_full[fa, fb] = mc.ci[ia, ib]
    def Hop(c, ne=nelec):
        return fci.direct_spin1.contract_2e(h2e, c, nmo, ne)
    E0 = np.dot(ci_full.ravel(), Hop(ci_full).ravel())          # electronic energy of the vacuum
    return dict(nmo=nmo, nelec=nelec, h2e=h2e, ci=ci_full, Hop=Hop, E0=E0, ncore=ncore, ncas=ncas, enuc=mol.energy_nuc())

def apply_so(op, ci, nmo, nelec, p):
    """apply creation (op='c') or annihilation (op='d') of spin orbital p = 2P+sigma; returns (vec, nelec')."""
    P, s = divmod(p, 2)
    if op == 'c':
        if s == 0: return addons.cre_a(ci, nmo, nelec, P), (nelec[0] + 1, nelec[1])
        else:      return addons.cre_b(ci, nmo, nelec, P), (nelec[0], nelec[1] + 1)
    else:
        if s == 0: return addons.des_a(ci, nmo, nelec, P), (nelec[0] - 1, nelec[1])
        else:      return addons.des_b(ci, nmo, nelec, P), (nelec[0], nelec[1] - 1)

def apply_string(ops, ci, nmo, nelec):
    """ops: list of (op, p) applied right-to-left as written left-to-right in an operator product,
    i.e. apply_string([('c',x),('c',y),('d',j),('d',i)]) = a_x+ a_y+ a_j a_i |ci>."""
    v, ne = ci, nelec
    for op, p in reversed(ops):
        # electron count must stay in range
        if op == 'd' and ((p % 2 == 0 and ne[0] == 0) or (p % 2 == 1 and ne[1] == 0)): return None, None
        if op == 'c' and ((p % 2 == 0 and ne[0] == nmo) or (p % 2 == 1 and ne[1] == nmo)): return None, None
        v, ne = apply_so(op, v, nmo, ne, p)
    return v, ne

def class_energies(S, Hm, v, E0, tol=1e-8):
    """S overlap, Hm = <Phi|H|Phi'>, v = <Phi|H|Psi0>; returns E2 (full-H Hylleraas), E_CI(class), rank."""
    s, U = np.linalg.eigh(S)
    keep = s > tol * s.max()
    X = U[:, keep] / np.sqrt(s[keep])
    Ht = X.T @ (Hm - E0 * S) @ X
    vt = X.T @ v
    E2 = -vt @ np.linalg.solve(Ht, vt)
    # class CI: diagonalize [[0, vt],[vt, Ht]]
    M = np.zeros((len(vt) + 1, len(vt) + 1)); M[0, 1:] = vt; M[1:, 0] = vt; M[1:, 1:] = Ht
    e = np.linalg.eigvalsh(M)[0]
    return E2, e, int(keep.sum())

def build_class(setup, ops_list):
    """ops_list: list of operator strings; returns S, H, v for the nonvanishing states."""
    ci, nelec, nmo, Hop = setup['ci'], setup['nelec'], setup['nmo'], setup['Hop']
    Hpsi = Hop(ci)
    vecs = []
    for ops in ops_list:
        v, ne = apply_string(ops, ci, nmo, nelec)
        if v is None or ne != nelec: continue
        if np.linalg.norm(v) < 1e-10: continue
        vecs.append(v.ravel())
    if not vecs: return None
    Vm = np.array(vecs)
    S = Vm @ Vm.T
    HV = np.array([Hop(x.reshape(ci.shape)).ravel() for x in vecs])
    Hm = Vm @ HV.T
    v = Vm @ Hpsi.ravel()
    return S, Hm, v, Vm

if __name__ == '__main__':
    R = float(sys.argv[1]) if len(sys.argv) > 1 else 1.1
    mol = gto.M(atom='N 0 0 0; N 0 0 %f' % R, basis='sto-3g', unit='angstrom', verbose=0)
    mf = scf.RHF(mol).run()
    mc = mcscf.CASSCF(mf, 6, 6); mc.natorb = True; mc.run()
    st = full_space_setup(mol, mc)
    print("R=%.2f  E0 (full-space) - E_CAS = %.2e" % (R, st['E0'] + st['enuc'] - mc.e_tot))
    e_fci = fci.FCI(mf).kernel()[0]
    print("FCI - CAS = %.6f" % (e_fci - mc.e_tot))
    ncore, ncas, nmo = st['ncore'], st['ncas'], st['nmo']
    C = list(range(2 * ncore)); A = list(range(2 * ncore, 2 * (ncore + ncas))); V = list(range(2 * (ncore + ncas), 2 * nmo))
    E0 = st['E0']
    # class (ca,ca): a_x+ a_y+ a_j a_i
    ops = [[('c', x), ('c', y), ('d', j), ('d', i)] for i, j in itertools.combinations(C, 2) for x, y in itertools.combinations(A, 2)]
    S, Hm, v, Vm = build_class(st, ops)
    E2, Eci, rank = class_energies(S, Hm, v, E0)
    print("class (ca,ca)  [ij->xy]: states %d rank %d   E2(full H) = %.6f   E_CI(class) = %.6f" % (len(v), rank, E2, Eci))
    # compare exact norms with quasi-boson product norms for a few doubles
    (d1a, d1b), _ = mc.fcisolver.make_rdm12s(mc.ci, ncas, mc.nelecas)
    n_so = np.zeros(2 * ncas); n_so[0::2] = np.diag(d1a); n_so[1::2] = np.diag(d1b)
    print("  natural occupations (so):", np.round(n_so, 4))
    k = 0
    print("  double           exact norm   (1-n_x)(1-n_y)   ratio")
    for i, j in itertools.combinations(C, 2):
        for x, y in itertools.combinations(A, 2):
            v0, ne = apply_string([('c', x), ('c', y), ('d', j), ('d', i)], st['ci'], nmo, st['nelec'])
            if v0 is None or ne != st['nelec'] or np.linalg.norm(v0) < 1e-10: continue
            if (i, j) == (0, 1) or (i, j) == (0, 2):
                xa, ya = x - 2 * ncore, y - 2 * ncore
                qb = (1 - n_so[xa]) * (1 - n_so[ya]); ex = np.dot(v0.ravel(), v0.ravel())
                if k < 14:
                    print("  (%d,%d)->(%d,%d)   %10.3e   %10.3e   %8.2f" % (i, j, x, y, ex, qb, ex / qb))
                k += 1
    # semi-internal class (ca,aa): a_z+ a_w a_x+ a_i  (w != z, x != z)
    ops2 = [[('c', z), ('d', w), ('c', x), ('d', i)] for i in C for x in A for z in A for w in A if z != w and z != x and w != x]
    S2, H2, v2, _ = build_class(st, ops2)
    E2b, Ecib, rank2 = class_energies(S2, H2, v2, E0, tol=1e-7)
    print("class (ca,aa)  [i->x, w->z]: states %d rank %d   E2(full H) = %.6f   E_CI(class) = %.6f" % (len(v2), rank2, E2b, Ecib))
    # both classes together
    Vall = np.vstack([Vm, _])
    Sall = Vall @ Vall.T
    HV = np.array([st['Hop'](x.reshape(st['ci'].shape)).ravel() for x in Vall])
    Hall = Vall @ HV.T; vall = Vall @ st['Hop'](st['ci']).ravel()
    E2c, Ecic, rankc = class_energies(Sall, Hall, vall, E0, tol=1e-7)
    print("both classes together:  rank %d   E2(full H) = %.6f   E_CI = %.6f" % (rankc, E2c, Ecic))
