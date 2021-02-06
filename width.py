"""
* Calculate the width mean and std for every z-bin.
* Plot the redshift distributions.

Results
-------
2mpz  : w = 1.0002454506913296 +/- 0.0014776915542366337
wisc1 : w = 1.0005280835364896 +/- 0.00046956888262295786
wisc2 : w = 1.0005254756078834 +/- 0.0004886242861704727
wisc3 : w = 0.9925133132417371 +/- 0.003513286079390885
wisc4 : w = 1.0004355370346598 +/- 0.0004570770425962469
wisc5 : w = 0.9859408806256956 +/- 0.0018051163523158348
"""
import numpy as np
from funcs import Likelihood

# names of z-bins
zbins = ["2mpz"] + ["wisc%d" % b for b in range(1, 6)]

for zbin in zbins:
    # load N(z)
    f = np.load("out/DIR_%s.npz" % zbin)
    z_mid, Nz = f["z_arr"], f["nz_arr"]

    # load JKs
    diff_sq = 0
    Njk = 0
    while True:
        try:
            f = np.load("out/DIR_%s_jk%s.npz" % (zbin, Njk))
            diff_sq += (f["nz_arr"]-Nz)**2
            Njk += 1
        except FileNotFoundError:
            break

    # run likelihood
    dNz = np.sqrt(Njk/(Njk-1) * diff_sq)
    l = Likelihood(z_mid, Nz, dNz)
    w, dw = l.prob()
    print(f"{zbin} : w = {w} +/- {dw}")
