"""
Minimal symbolic normal-ordering (Wick) engine for strings of fermionic
creation/annihilation operators, used to reduce expectation values of
operator products in a correlated state to contractions of integrals with
reduced density matrices (RDMs).

Conventions
-----------
* An operator string is a tuple of (label, dagger) pairs, read left to right.
* Normal order means all creators to the left of all annihilators; it is
  reached by repeated use of  a_i a_j^+ = delta_ij - a_j^+ a_i  (no other
  contractions, i.e. the reference is the physical vacuum), so that the
  expectation value of a normal-ordered string with k creators and k
  annihilators in the state |0> is the k-particle RDM

      D_k[p1, ..., pk, q1, ..., qk] = <0| p1^+ ... pk^+ qk ... q1 |0>

  (creators in the order written, annihilators reversed), and strings with
  unequal numbers of creators and annihilators have zero expectation value.
* Labels starting with '_' are summation (dummy) indices; all other labels
  are free (external) indices.  Kronecker deltas produced by the
  anticommutation are used to eliminate dummies; deltas between two free
  labels are kept as identity matrices.
* A term is (coeff, deltas, tensors, ops):
      coeff   : float
      deltas  : tuple of (label, label) pairs
      tensors : tuple of (name, (labels...)) integral factors
      ops     : operator string
  An expression is a list of terms.

The Hamiltonian is  H = sum_pq h[p,q] p^+ q + 1/2 sum V[p,r,q,s] p^+ q^+ s r
(V[p,r,q,s] = (pr|qs) = <pq|rs>, the convention of the rest of the package).
"""
import functools
import itertools
import numpy as np

DUMMY = '_'


# ----------------------------------------------------------------------------
# construction of expressions
# ----------------------------------------------------------------------------
def string(ops, coeff=1.0, tensors=()):
    """Expression consisting of a single operator string."""
    return [(float(coeff), (), tuple(tensors), tuple(ops))]


def cre(label):
    return (label, True)


def des(label):
    return (label, False)


def dagger(expr):
    """Hermitian conjugate of an expression (real coefficients/integrals)."""
    out = []
    for c, d, t, ops in expr:
        out.append((c, d, t, tuple((l, not dg) for (l, dg) in reversed(ops))))
    return out


def scale(expr, factor):
    return [(c * factor, d, t, o) for c, d, t, o in expr]


def add(*exprs):
    out = []
    for e in exprs:
        out.extend(e)
    return out


def mul(A, B):
    out = []
    for ca, da, ta, oa in A:
        for cb, db, tb, ob in B:
            out.append((ca * cb, da + db, ta + tb, oa + ob))
    return out


def commutator(A, B):
    return add(mul(A, B), scale(mul(B, A), -1.0))


def hamiltonian(suffix=''):
    """H with dummy labels _p, _q, _r, _s (optionally suffixed)."""
    p, q, r, s = (DUMMY + x + suffix for x in 'pqrs')
    one = string([cre(p), des(q)], tensors=[('h', (p, q))])
    two = string([cre(p), cre(q), des(s), des(r)], coeff=0.5, tensors=[('V', (p, r, q, s))])
    return add(one, two)


# ----------------------------------------------------------------------------
# normal ordering
# ----------------------------------------------------------------------------
@functools.lru_cache(maxsize=None)
def _normal_order_ops(ops):
    """Normal order an operator string.  Returns a tuple of
    (sign, deltas, ops_normal)."""
    for i in range(len(ops) - 1):
        (la, da), (lb, db) = ops[i], ops[i + 1]
        if (not da) and db:
            rest = ops[:i] + ops[i + 2:]
            swapped = ops[:i] + (ops[i + 1], ops[i]) + ops[i + 2:]
            out = []
            for s, d, o in _normal_order_ops(rest):
                out.append((s, ((la, lb),) + d, o))
            for s, d, o in _normal_order_ops(swapped):
                out.append((-s, d, o))
            return tuple(out)
    return ((1, (), ops),)


def _is_dummy(label):
    return label.startswith(DUMMY)


def _substitute(term, old, new):
    c, deltas, tensors, ops = term
    deltas = tuple((new if a == old else a, new if b == old else b) for a, b in deltas)
    tensors = tuple((name, tuple(new if l == old else l for l in labels)) for name, labels in tensors)
    ops = tuple((new if l == old else l, dg) for l, dg in ops)
    return (c, deltas, tensors, ops)


def _resolve_deltas(term):
    """Eliminate dummies through deltas; keep free-free deltas."""
    c, deltas, tensors, ops = term
    while True:
        keep, done = [], True
        for k, (a, b) in enumerate(deltas):
            if a == b:
                continue
            if _is_dummy(a) or _is_dummy(b):
                old, new = (a, b) if _is_dummy(a) else (b, a)
                rest = deltas[:k] + deltas[k + 1:]
                term = _substitute((c, rest, tensors, ops), old, new)
                c, deltas, tensors, ops = term
                done = False
                break
            keep.append((a, b) if a <= b else (b, a))
        if done:
            return (c, tuple(sorted(keep)), tensors, ops)


def _sort_sign(labels):
    """Sort labels; return (sign of the permutation, sorted tuple) or
    (0, None) if a label is repeated (a^+ a^+ = a a = 0)."""
    labels = list(labels)
    if len(set(labels)) != len(labels):
        return 0, None
    sign = 1
    for i in range(len(labels)):
        for j in range(len(labels) - 1 - i):
            if labels[j] > labels[j + 1]:
                labels[j], labels[j + 1] = labels[j + 1], labels[j]
                sign = -sign
    return sign, tuple(labels)


def _canonical(term):
    c, deltas, tensors, ops = term
    crs = [l for l, dg in ops if dg]
    ans = [l for l, dg in ops if not dg]
    s1, crs = _sort_sign(crs)
    s2, ans = _sort_sign(ans)
    if s1 == 0 or s2 == 0:
        return None
    ops = tuple((l, True) for l in crs) + tuple((l, False) for l in ans)
    return (c * s1 * s2, deltas, tuple(sorted(tensors)), ops)


def normal_order(expr, tol=1e-14):
    """Normal order an expression: expand, eliminate dummy deltas, bring the
    creators/annihilators to canonical (sorted) order and combine equal
    terms."""
    acc = {}
    for c, deltas, tensors, ops in expr:
        for sign, d2, ops_n in _normal_order_ops(tuple(ops)):
            term = _resolve_deltas((c * sign, deltas + d2, tensors, ops_n))
            term = _canonical(term)
            if term is None:
                continue
            cc, dd, tt, oo = term
            key = (dd, tt, oo)
            acc[key] = acc.get(key, 0.0) + cc
    return [(c, d, t, o) for (d, t, o), c in acc.items() if abs(c) > tol]


# ----------------------------------------------------------------------------
# numerical evaluation
# ----------------------------------------------------------------------------
def evaluate(expr, free, n, tensors, rdm, optimize='greedy'):
    """Expectation value <0| expr |0> as an array over the free labels.

    tensors : dict name -> array (e.g. {'h': h, 'V': V})
    rdm     : dict k -> D_k (k = 1, 2, ...); D_0 is 1
    """
    free = list(free)
    out = np.zeros((n,) * len(free))
    letters = {}

    def letter(label):
        if label not in letters:
            letters[label] = chr(ord('a') + len(letters)) if len(letters) < 26 else chr(ord('A') + len(letters) - 26)
        return letters[label]

    for c, deltas, tens, ops in expr:
        letters.clear()
        for l in free:
            letter(l)
        subs, operands = [], []
        for name, labels in tens:
            subs.append(''.join(letter(l) for l in labels))
            operands.append(tensors[name])
        for a, b in deltas:
            subs.append(letter(a) + letter(b))
            operands.append(np.eye(n))
        crs = [l for l, dg in ops if dg]
        ans = [l for l, dg in ops if not dg]
        if len(crs) != len(ans):
            continue
        k = len(crs)
        if k > 0:
            subs.append(''.join(letter(l) for l in crs) + ''.join(letter(l) for l in reversed(ans)))
            operands.append(rdm[k])
        used = set(''.join(subs))
        outsub = ''.join(letter(l) for l in free)
        missing = [ch for ch in outsub if ch not in used]
        if missing:
            raise ValueError(f"free label without operand in term {(c, deltas, tens, ops)}")
        if not operands:
            out += c
            continue
        val = np.einsum(','.join(subs) + '->' + outsub, *operands, optimize=optimize)
        out += c * val
    return out


def rdm_orders(expr):
    """Set of RDM orders k that the expression needs."""
    ks = set()
    for c, d, t, ops in expr:
        ncr = sum(1 for l, dg in ops if dg)
        nan = len(ops) - ncr
        if ncr == nan:
            ks.add(ncr)
    return ks


# ----------------------------------------------------------------------------
# LaTeX output
# ----------------------------------------------------------------------------
def to_latex(expr, free_as=None):
    """Render an expression (after normal_order) as a LaTeX sum of terms:
    coefficient, integrals, deltas and RDM elements
    D_k[creators; annihilators-reversed]."""
    def lab(l):
        if l.startswith(DUMMY):
            return l[1:] + "'"                 # summation indices are primed
        return free_as.get(l, l) if free_as else l

    pieces = []
    for c, deltas, tens, ops in expr:
        crs = [lab(l) for l, dg in ops if dg]
        ans = [lab(l) for l, dg in ops if not dg]
        if len(crs) != len(ans):
            continue
        parts = []
        for a, b in deltas:
            parts.append(f"\\delta_{{{lab(a)}{lab(b)}}}")
        for name, labels in tens:
            if name == 'h':
                parts.append(f"h_{{{lab(labels[0])}{lab(labels[1])}}}")
            else:
                parts.append(f"v_{{{lab(labels[0])}{lab(labels[1])},{lab(labels[2])}{lab(labels[3])}}}")
        k = len(crs)
        if k:
            parts.append(f"D^{{({k})}}_{{{''.join(crs)},{''.join(reversed(ans))}}}")
        coef = f"{c:+g}"
        if abs(abs(c) - 1.0) < 1e-12:
            coef = '+' if c > 0 else '-'
        elif abs(abs(c) - 0.5) < 1e-12:
            coef = ('+' if c > 0 else '-') + '\\tfrac12'
        pieces.append(coef + ''.join(parts))
    return ' '.join(pieces)
