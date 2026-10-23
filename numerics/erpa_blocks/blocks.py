"""Resolve the (verified) ERPA double-commutator formulas into core/active/virtual
blocks for a CAS-type vacuum, print LaTeX, and verify each block numerically."""
import numpy as np, itertools, sys
from fractions import Fraction as Fr
from fockspace import *

# ----------------------------------------------------------------------------
# symbolic terms: (coef, factors) with factors = tuple of (name, (indices...))
# index = (label, type) ; type in 'CAV' ; summation labels start with '_'
# ----------------------------------------------------------------------------
SYM2 = {'g', 'd', 'FD', 'F', 'h'}          # symmetric 2-index objects
PAIR4 = {'w', 'L'}                         # antisymmetric in each pair, symmetric under pair swap

def term(coef, *factors):
    return (Fr(coef), tuple((nm, tuple(ix)) for nm, ix in factors))

def is_sum(lbl):
    return lbl.startswith('_')

def general_ph():
    p, q, r, s, t, u = 'p', 'q', 'r', 's', '_t', '_u'
    T = []
    T += [term(1, ('g', (q, s)), ('FD', (p, r))), term(1, ('g', (p, r)), ('FD', (q, s))),
          term(-1, ('d', (p, r)), ('F', (q, s))), term(-1, ('d', (q, s)), ('F', (p, r)))]
    T += [term(1, ('w', (p, t, u, r)), ('g', (q, u)), ('g', (t, s))),
          term(1, ('w', (q, t, u, s)), ('g', (p, u)), ('g', (t, r))),
          term(-1, ('w', (p, s, t, u)), ('g', (q, t)), ('g', (r, u))),
          term(-1, ('w', (q, r, t, u)), ('g', (p, t)), ('g', (s, u)))]
    T += [term(1, ('w', (p, t, u, r)), ('L', (q, t, u, s))),
          term(1, ('w', (q, t, u, s)), ('L', (p, t, u, r))),
          term(Fr(-1, 2), ('w', (p, s, t, u)), ('L', (q, r, t, u))),
          term(Fr(-1, 2), ('w', (q, r, t, u)), ('L', (p, s, t, u)))]
    return T

def general_pp():
    p, q, r, s, t, u = 'p', 'q', 'r', 's', '_t', '_u'
    seed = [term(1, ('d', (q, s)), ('FD', (p, r))), term(-1, ('d', (q, s)), ('F', (p, r))),
            term(-1, ('g', (q, s)), ('FD', (p, r))),
            term(1, ('w', (p, t, r, u)), ('g', (q, t)), ('g', (s, u))),
            term(1, ('w', (p, t, r, u)), ('L', (t, s, q, u)))]
    def relabel(T, mp):
        return [(c, tuple((nm, tuple(mp.get(i, i) for i in ix)) for nm, ix in f)) for c, f in T]
    T = []
    T += seed
    T += [(-c, f) for c, f in relabel(seed, {p: q, q: p})]
    T += [(-c, f) for c, f in relabel(seed, {r: s, s: r})]
    T += relabel(seed, {p: q, q: p, r: s, s: r})
    T += [term(1, ('w', (p, q, r, s))),
          term(-1, ('g', (p, t)), ('w', (t, q, r, s))), term(-1, ('g', (q, t)), ('w', (p, t, r, s))),
          term(-1, ('g', (r, t)), ('w', (p, q, t, s))), term(-1, ('g', (s, t)), ('w', (p, q, r, t)))]
    return T

# ---------------------------------------------------------------------------
def typed_expand(T, free_types):
    """Assign types to the free labels (dict) and enumerate types of summation labels.
    Returns list of (coef, factors) with indices as (label, type)."""
    out = []
    for c, f in T:
        sums = sorted({i for _, ix in f for i in ix if is_sum(i)})
        for types in itertools.product('CAV', repeat=len(sums)):
            tp = dict(free_types); tp.update(dict(zip(sums, types)))
            out.append((c, tuple((nm, tuple((i, tp[i]) for i in ix)) for nm, ix in f)))
    return out

def cas_rules(c, f):
    """Apply CAS-vacuum rules; returns (coef, factors) or None."""
    new = []
    for nm, ix in f:
        ty = [t for _, t in ix]
        if nm == 'g':
            if ty == ['C', 'C']:
                new.append(('d', ix))
            elif ty == ['A', 'A']:
                new.append((nm, ix))
            else:
                return None
        elif nm == 'L':
            if ty != ['A'] * 4:
                return None
            new.append((nm, ix))
        elif nm == 'd':
            if ty[0] != ty[1]:
                return None
            new.append((nm, ix))
        elif nm == 'F':
            if ty == ['C', 'C']:
                new.append(('FD', ix))
            elif ty == ['V', 'V']:
                return None
            elif set(ty) == {'C', 'V'}:
                c = c * Fr(1, 2); new.append(('FD', ix))
            else:
                new.append((nm, ix))
        else:
            new.append((nm, ix))
    # resolve deltas with summation labels
    f = list(new)
    changed = True
    while changed:
        changed = False
        for k, (nm, ix) in enumerate(f):
            if nm == 'd':
                (a, ta), (b, tb) = ix
                if a == b:
                    del f[k]; changed = True; break
                if is_sum(a) or is_sum(b):
                    old, nw = ((a, ta), (b, tb)) if is_sum(a) else ((b, tb), (a, ta))
                    del f[k]
                    f = [(n2, tuple(nw if i == old else i for i in ix2)) for n2, ix2 in f]
                    changed = True; break
    return (c, tuple(f))

def canon_factor(nm, ix):
    sign = 1
    if nm in SYM2:
        ix = tuple(sorted(ix))
    elif nm in PAIR4:
        a, b, cc, d = ix
        if a > b: a, b = b, a; sign = -sign
        if cc > d: cc, d = d, cc; sign = -sign
        if (a, b) > (cc, d): a, b, cc, d = cc, d, a, b
        ix = (a, b, cc, d)
    return sign, (nm, ix)

def canon(c, f):
    """Canonical form incl. renaming of summation labels."""
    sums = sorted({i for _, ix in f for i in ix if is_sum(i[0])})
    best = None
    for perm in itertools.permutations(range(len(sums))):
        names = ['_%d' % k for k in range(len(sums))]
        mp = {sums[j]: (names[perm[j]], sums[j][1]) for j in range(len(sums))}
        ff = [(nm, tuple(mp.get(i, i) for i in ix)) for nm, ix in f]
        sign = 1; cf = []
        for nm, ix in ff:
            s, fac = canon_factor(nm, ix); sign *= s; cf.append(fac)
        cf = tuple(sorted(cf))
        key = (cf, sign)
        if best is None or cf < best[0]:
            best = (cf, sign)
    return (c * best[1], best[0])

def resolve(T, free_types):
    acc = {}
    for c, f in typed_expand(T, free_types):
        r = cas_rules(c, f)
        if r is None:
            continue
        c2, f2 = canon(*r)
        acc[f2] = acc.get(f2, Fr(0)) + c2
    return [(c, f) for f, c in acc.items() if c != 0]

# ---------------------------------------------------------------------------
# LaTeX rendering
LET = {'C': 'ijkl', 'A': 'xyzw', 'V': 'abcd'}
SUMLET = {'C': ['k', 'l', 'm', 'n'], 'A': ['u', 'v', 'u\'', 'v\''], 'V': ['c', 'd', 'e', 'f']}

def render(terms, free_names):
    """free_names: dict label->letter. Summation labels get letters by type."""
    def order_key(t):
        c, f = t
        names = [nm for nm, _ in f]
        return (names.count('w') + names.count('L') * 2, names.count('L'), names.count('g'), -names.count('d'), -names.count('FD'), -names.count('F'), str(f))
    pieces = []
    for c, f in sorted(terms, key=order_key):
        sums = sorted({i for _, ix in f for i in ix if is_sum(i[0])})
        used_free = set(free_names.values())
        smap = {}; cnt = {'C': 0, 'A': 0, 'V': 0}
        for lbl, ty in sums:
            cands = [x for x in SUMLET[ty] if x not in used_free]
            smap[lbl] = cands[cnt[ty]]; cnt[ty] += 1
        def nm_(i):
            lbl, ty = i
            return smap[lbl] if is_sum(lbl) else free_names[lbl]
        facs = []
        for nm, ix in sorted(f, key=lambda x: {'d': 0, 'FD': 1, 'F': 2, 'h': 3, 'g': 5, 'w': 4, 'L': 6}[x[0]]):
            a = [nm_(i) for i in ix]
            if nm == 'd': facs.append(r'\delta_{%s%s}' % tuple(a))
            elif nm == 'FD': facs.append(r'f_{%s%s}' % tuple(a))
            elif nm == 'F': facs.append(r'\bar{\mathcal L}_{%s%s}' % tuple(a))
            elif nm == 'h': facs.append(r'h_{%s%s}' % tuple(a))
            elif nm == 'g': facs.append(r'\gamma_{%s%s}' % tuple(a))
            elif nm == 'w': facs.append(r'\langle %s%s||%s%s\rangle' % tuple(a))
            elif nm == 'L': facs.append(r'\lambda_{%s%s,%s%s}' % tuple(a))
        coef = ''
        if c == 1: coef = '+'
        elif c == -1: coef = '-'
        elif c.denominator == 1: coef = '%+d' % c.numerator
        else: coef = ('+' if c > 0 else '-') + r'\tfrac{%d}{%d}' % (abs(c.numerator), c.denominator)
        sm = ''
        if sums:
            sm = r'\sum_{%s}' % ''.join(smap[l] for l, _ in sums)
        pieces.append(coef + sm + ''.join(facs))
    s = ' '.join(pieces)
    if s.startswith('+'): s = s[1:]
    return s if s else '0'

# ---------------------------------------------------------------------------
# numerics for a CAS-type vacuum
def cas_setup(seed=3):
    rng = np.random.default_rng(seed)
    n = 8; C = [0, 1]; Ac = [2, 3, 4, 5]; Vv = [6, 7]; nact_el = 2
    A = annihilators(n); Ad = np.transpose(A, (0, 2, 1)); dim = 2 ** n
    h, V = random_integrals(n, rng); w = antisym(V)
    H = hamiltonian(h, V, A)
    psi = np.zeros(dim)
    for k in range(dim):
        if all(k >> p & 1 for p in C) and not any(k >> p & 1 for p in Vv) and sum(k >> p & 1 for p in Ac) == nact_el:
            psi[k] = rng.normal()
    psi /= np.linalg.norm(psi)
    g, G = rdms(psi, A)
    lam = G - np.einsum('pr,qs->pqrs', g, g) + np.einsum('ps,qr->pqrs', g, g)
    I = np.eye(n)
    calL = np.einsum('pt,tq->pq', h, g) + 0.5 * np.einsum('putv,qutv->pq', w, G)
    F = 0.5 * (calL + calL.T)
    FD = h + np.einsum('ptru,tu->pr', w, g)
    # reference (symmetrized) double commutators
    Hpsi = H @ psi
    Epsi = np.array([[Ad[p] @ (A[q] @ psi) for q in range(n)] for p in range(n)])
    Cpsi = np.einsum('ij,rsj->rsi', H, Epsi) - np.array([[Ad[r] @ (A[s] @ Hpsi) for s in range(n)] for r in range(n)])
    Mph = np.einsum('pqi,rsi->pqrs', Epsi, Cpsi) + np.einsum('sri,qpi->pqrs', Cpsi, Epsi)
    Mph = 0.5 * (Mph + np.transpose(Mph, (2, 3, 0, 1)))
    Sph = np.einsum('pqi,rsi->pqrs', Epsi, Epsi) - np.einsum('sri,qpi->pqrs', Epsi, Epsi)
    Pdpsi = np.array([[Ad[r] @ (Ad[s] @ psi) for s in range(n)] for r in range(n)])
    Papsi = np.array([[A[q] @ (A[p] @ psi) for p in range(n)] for q in range(n)])
    Cpppsi = np.einsum('ij,rsj->rsi', H, Pdpsi) - np.array([[Ad[r] @ (Ad[s] @ Hpsi) for s in range(n)] for r in range(n)])
    Cpa = np.einsum('ij,rsj->rsi', H, np.transpose(Papsi, (1, 0, 2))) - np.array([[A[s] @ (A[r] @ Hpsi) for s in range(n)] for r in range(n)])
    Mpp = np.einsum('pqi,rsi->pqrs', Pdpsi, Cpppsi) + np.einsum('rsi,qpi->pqrs', Cpa, Papsi)
    Mpp = 0.5 * (Mpp + np.transpose(Mpp, (2, 3, 0, 1)))
    Spp = np.einsum('pqi,rsi->pqrs', Pdpsi, Pdpsi) - np.einsum('rsi,qpi->pqrs', np.transpose(Papsi, (1, 0, 2)), Papsi)
    arrays = {'h': h, 'w': w, 'g': g, 'L': lam, 'd': I, 'FD': FD, 'F': F}
    ranges = {'C': C, 'A': Ac, 'V': Vv}
    return dict(n=n, h=h, w=w, g=g, G=G, lam=lam, FD=FD, F=F, calL=calL, Mph=Mph, Sph=Sph, Mpp=Mpp, Spp=Spp,
                arrays=arrays, ranges=ranges, psi=psi, H=H)

def evaluate_block(terms, free, arrays, ranges):
    """free: list of (label, type) in output order."""
    letters = 'abcdefghijklmnopqrstuvwxyz'
    shape = tuple(len(ranges[t]) for _, t in free)
    out = np.zeros(shape)
    for c, f in terms:
        lab = {}
        def L(i):
            if i not in lab: lab[i] = letters[len(lab)]
            return lab[i]
        for i in free: L(i)
        subs, ops = [], []
        for nm, ix in f:
            subs.append(''.join(L(i) for i in ix))
            arr = arrays[nm]
            ops.append(arr[np.ix_(*[ranges[t] for _, t in ix])])
        outsub = ''.join(L(i) for i in free)
        used = set(''.join(subs))
        if not all(ch in used for ch in outsub):
            # term without dependence on some free index (cannot happen here)
            raise RuntimeError
        out += float(c) * np.einsum(','.join(subs) + '->' + outsub, *ops, optimize=True)
    return out

# ---------------------------------------------------------------------------
PH_SECTORS = {'cv': ('V', 'C'), 'ca': ('A', 'C'), 'av': ('V', 'A'), 'aa': ('A', 'A')}
PP_SECTORS = {'vv': ('V', 'V'), 'av': ('V', 'A'), 'aa': ('A', 'A'), 'ca': ('C', 'A'), 'cc': ('C', 'C')}

def free_letters(types):
    """assign display letters to the four free labels p,q,r,s given their types."""
    cnt = {'C': 0, 'A': 0, 'V': 0}; names = {}
    for lbl, ty in zip('pqrs', types):
        names[lbl] = LET[ty][cnt[ty]]; cnt[ty] += 1
    return names

def run():
    D = cas_setup()
    n = D['n']; R = D['ranges']
    # ---- structural checks of the CAS vacuum
    g, lam, FD, F, calL, w, h = D['g'], D['lam'], D['FD'], D['F'], D['calL'], D['w'], D['h']
    C, Ac, Vv = R['C'], R['A'], R['V']
    print("gamma core block = 1:", abs(g[np.ix_(C, C)] - np.eye(len(C))).max(), " virtual block:", abs(g[np.ix_(Vv, Vv)]).max(),
          " mixed:", max(abs(g[np.ix_(C, Ac)]).max(), abs(g[np.ix_(Ac, Vv)]).max(), abs(g[np.ix_(C, Vv)]).max()))
    lam_act = np.zeros_like(lam); lam_act[np.ix_(Ac, Ac, Ac, Ac)] = lam[np.ix_(Ac, Ac, Ac, Ac)]
    print("cumulant outside all-active block:", abs(lam - lam_act).max())
    # Lagrangian CAS forms
    Lam = 0.5 * np.einsum('putv,qutv->pq', w, lam)          # Lambda_{pq} = 1/2 sum <pu||tv> lam_{qu,tv}
    pred = np.zeros((n, n))
    pred[:, C] = FD[:, C]
    pred[:, Ac] = np.einsum('px,xy->py', FD[:, Ac], g[np.ix_(Ac, Ac)]) + Lam[:, Ac]
    print("Lagrangian CAS form (L_pj=f_pj, L_pb=0, L_py=sum f_px g_xy + Lambda_py):", abs(pred - calL).max())
    fcore = h + np.einsum('pjqj->pq', w[:, C][:, :, :, C])
    FDpred = fcore + np.einsum('xy,pxqy->pq', g[np.ix_(Ac, Ac)], w[:, Ac][:, :, :, Ac])
    print("generalized Fock f = h + core Coulomb-exchange + active:", abs(FDpred - FD).max())
    out = {}
    # ---- ph metric blocks
    Sph = D['Sph']
    # ---- ph A and B blocks
    Tph = general_ph()
    print("\n==== ph-ERPA blocks (symmetrized M), A-type [(pq),(rs)] and B-type [-M_(pq),(sr)]")
    keys = list(PH_SECTORS)
    for i1, k1 in enumerate(keys):
        for k2 in keys[i1:]:
            P1, H1 = PH_SECTORS[k1]; P2, H2 = PH_SECTORS[k2]
            for kind in ('A', 'B'):
                if kind == 'A':
                    types = (P1, H1, P2, H2); sign = 1
                else:
                    types = (P1, H1, H2, P2); sign = -1
                ft = dict(zip('pqrs', types))
                terms = resolve(Tph, ft)
                terms = [(sign * c, f) for c, f in terms]
                free = [(l, ft[l]) for l in 'pqrs']
                val = evaluate_block(terms, free, D['arrays'], R)
                ref = sign * D['Mph'][np.ix_(R[types[0]], R[types[1]], R[types[2]], R[types[3]])]
                err = abs(val - ref).max()
                names = free_letters(types)
                tex = render(terms, names)
                lab = '%s_{(%s%s),(%s%s)}^{%s-%s}' % (kind, names['p'], names['q'], names['r' if kind == 'A' else 's'], names['s' if kind == 'A' else 'r'], k1, k2)
                print("  %s-%s %s: %d terms, max err %.1e" % (k1, k2, kind, len(terms), err))
                out[('ph', kind, k1, k2)] = (lab, tex, err, len(terms))
    # ---- pp blocks
    Tpp = general_pp()
    print("\n==== pp-ERPA blocks (symmetrized M) [(pq),(rs)]")
    keys = list(PP_SECTORS)
    for i1, k1 in enumerate(keys):
        for k2 in keys[i1:]:
            types = PP_SECTORS[k1] + PP_SECTORS[k2]
            ft = dict(zip('pqrs', types))
            terms = resolve(Tpp, ft)
            free = [(l, ft[l]) for l in 'pqrs']
            val = evaluate_block(terms, free, D['arrays'], R)
            ref = D['Mpp'][np.ix_(R[types[0]], R[types[1]], R[types[2]], R[types[3]])]
            err = abs(val - ref).max()
            names = free_letters(types)
            tex = render(terms, names)
            lab = 'M_{(%s%s),(%s%s)}^{%s-%s}' % (names['p'], names['q'], names['r'], names['s'], k1, k2)
            print("  %s-%s: %d terms, max err %.1e" % (k1, k2, len(terms), err))
            out[('pp', 'M', k1, k2)] = (lab, tex, err, len(terms))
    # metric blocks numerically (sanity): ph
    print("\n==== metric sanity")
    Spp = D['Spp']
    print(" ph S cv-cv = d d:", abs(Sph[np.ix_(Vv, C, Vv, C)] - np.einsum('ab,ij->aibj', np.eye(2), np.eye(2))).max())
    print(" ph S ca-ca = d_ij (1-g)_{yx}... check d_ij(delta_xy - g_yx):", abs(Sph[np.ix_(Ac, C, Ac, C)] - np.einsum('ij,xy->xiyj', np.eye(2), np.eye(4) - g[np.ix_(Ac, Ac)].T)).max())
    print(" ph S av-av = d_ab g_xy:", abs(Sph[np.ix_(Vv, Ac, Vv, Ac)] - np.einsum('ab,xy->axby', np.eye(2), g[np.ix_(Ac, Ac)])).max())
    print(" pp S av-av = d_ab (1-g)_xy:", abs(Spp[np.ix_(Vv, Ac, Vv, Ac)] - np.einsum('ab,xy->axby', np.eye(2), np.eye(4) - g[np.ix_(Ac, Ac)])).max())
    print(" pp S ca-ca = -d_ij g_xy:", abs(Spp[np.ix_(C, Ac, C, Ac)] + np.einsum('ij,xy->ixjy', np.eye(2), g[np.ix_(Ac, Ac)])).max())
    print(" pp S cv-cv = 0:", abs(Spp[np.ix_(Vv, C, Vv, C)]).max(), " ph S mixed-sector e.g. cv-ca:", abs(Sph[np.ix_(Vv, C, Ac, C)]).max())
    return out

if __name__ == '__main__':
    out = run()
    with open('blocks_out.tex', 'w') as fh:
        for key, (lab, tex, err, nt) in out.items():
            fh.write('%% %s  (%d terms, err %.1e)\n%s &= %s\n\n' % (key, nt, err, lab, tex))
    print("\nwrote blocks_out.tex")
