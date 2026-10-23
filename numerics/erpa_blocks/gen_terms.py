import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'mrgw_hermitian'))
import wick
H = wick.hamiltonian()
c, d = wick.cre, wick.des
# ph: <[E_qp,[H,E_rs]]>
Eqp = wick.string([c('q'), d('p')]); Ers = wick.string([c('r'), d('s')])
eM = wick.normal_order(wick.commutator(Eqp, wick.commutator(H, Ers)))
print("ph double commutator: %d terms, orders %s" % (len(eM), sorted(wick.rdm_orders(eM))))
print(wick.to_latex(eM)); print()
# pp: <[a_q a_p,[H,a_r^+ a_s^+]]>
Bra = wick.string([d('q'), d('p')]); Ket = wick.string([c('r'), c('s')])
eMpp = wick.normal_order(wick.commutator(Bra, wick.commutator(H, Ket)))
print("pp double commutator: %d terms, orders %s" % (len(eMpp), sorted(wick.rdm_orders(eMpp))))
print(wick.to_latex(eMpp)); print()
eSpp = wick.normal_order(wick.commutator(Bra, Ket))
print("pp metric: %d terms" % len(eSpp)); print(wick.to_latex(eSpp)); print()
# raw term dump for pp
for t in eMpp: print(t)
