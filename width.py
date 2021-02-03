"""
Calculate the width mean and std for every z-bin.

Results
-------
2mpz : w = 1.0005244841876295 +/- 0.00046476413050352693
wisc1 : w = 0.9988705810345839 +/- 0.010429321829146921
wisc2 : w = 0.9997152056910912 +/- 0.011159028865620063
wisc3 : w = 0.9994581222032399 +/- 0.01142192739230215
wisc4 : w = 0.9989281281612168 +/- 0.011157022627974507
wisc5 : w = 0.9985619687842962 +/- 0.010799995134135933
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
