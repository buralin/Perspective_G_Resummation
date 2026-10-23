import blocks as B
import numpy as np

def render_lines(terms, names, per_line=3):
    s = B.render(terms, names)
    # split at top-level +/- signs (terms start with +,- followed by \sum or \delta etc.)
    import re
    parts = re.split(r'(?<=[^{^_])(?=[+-](?:\\sum|\\delta|f_|\\bar|\\gamma|\\langle|\\lambda|\\tfrac|\d))', ' ' + s)
    parts = [p.strip() for p in parts if p.strip()]
    lines = []
    for k in range(0, len(parts), per_line):
        lines.append(' '.join(parts[k:k + per_line]))
    return (' \\nonumber\\\\\n  &\\quad ').join(lines)

D = B.cas_setup()
R = D['ranges']
Tph = B.general_ph(); Tpp = B.general_pp()
keys = list(B.PH_SECTORS)
out = []
# identity check M_(qp),(sr) = M_(pq),(rs)
M = D['Mph']
print("identity M_(qp),(sr) = M_(pq),(rs):", abs(M - np.transpose(M, (1, 0, 3, 2))).max())
Mpp = D['Mpp']
print("pp antisymmetry p<->q:", abs(Mpp + np.transpose(Mpp, (1, 0, 2, 3))).max(), " r<->s:", abs(Mpp + np.transpose(Mpp, (0, 1, 3, 2))).max())
out.append("%% ---------------- ph blocks ----------------")
for i1, k1 in enumerate(keys):
    for k2 in keys[i1:]:
        P1, H1 = B.PH_SECTORS[k1]; P2, H2 = B.PH_SECTORS[k2]
        kinds = ('A',) if k2 == 'aa' else ('A', 'B')
        for kind in kinds:
            types = (P1, H1, P2, H2) if kind == 'A' else (P1, H1, H2, P2)
            sign = 1 if kind == 'A' else -1
            ft = dict(zip('pqrs', types))
            terms = [(sign * c, f) for c, f in B.resolve(Tph, ft)]
            names = B.free_letters(types)
            if k2 == 'aa':
                lab = r'M_{(%s%s),(%s%s)}' % (names['p'], names['q'], names['r'], names['s'])
            else:
                lab = r'%s_{(%s%s),(%s%s)}' % (kind, names['p'], names['q'], names['r' if kind == 'A' else 's'], names['s' if kind == 'A' else 'r'])
            out.append(r'%% %s-%s %s' % (k1, k2, kind))
            out.append(r'  %s &= %s' % (lab, render_lines(terms, names)))
            out.append(r'  \label{eq:ph-%s-%s-%s}\\' % (kind, k1, k2))
out.append("%% ---------------- pp blocks ----------------")
keys = list(B.PP_SECTORS)
for i1, k1 in enumerate(keys):
    for k2 in keys[i1:]:
        types = B.PP_SECTORS[k1] + B.PP_SECTORS[k2]
        ft = dict(zip('pqrs', types))
        terms = B.resolve(Tpp, ft)
        names = B.free_letters(types)
        lab = r'M^{\pp}_{(%s%s),(%s%s)}' % (names['p'], names['q'], names['r'], names['s'])
        out.append(r'%% %s-%s' % (k1, k2))
        out.append(r'  %s &= %s' % (lab, render_lines(terms, names)))
        out.append(r'  \label{eq:pp-%s-%s}\\' % (k1, k2))
open('blocks_formatted.tex', 'w').write('\n'.join(out) + '\n')
print(open('blocks_formatted.tex').read())
