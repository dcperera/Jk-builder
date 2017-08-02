import JK_builder_C
import numpy as np
import psi4

class JK_builder:
    def __init__(self):
        pass

    def by_conventional(self, I, D):
        return JK_builder_C.form_JK_conventional(I, D)
    
    def by_df(self, Ig, D, C):
        return JK_builder_C.form_JK_df(Ig, D, C)

    def calc_Ig(self, wfn, mol, basname):
        orb = wfn.basisset()
        mints = psi4.core.MintsHelper(orb)
        aux = psi4.core.BasisSet.build(mol, fitrole="JKFIT", other=basname)
        zero_bas = psi4.core.BasisSet.zero_ao_basis_set()
        Qls_tilde = mints.ao_eri(zero_bas, aux, orb, orb)
        Qls_tilde = np.squeeze(Qls_tilde)
        metric = mints.ao_eri(zero_bas, aux, zero_bas, aux)
        metric.power(-0.5, 1.e-14)
        metric = np.squeeze(metric)
        Pls = metric @ Qls_tilde
        return Pls
