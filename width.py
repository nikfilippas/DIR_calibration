"""
* Calculate the width mean and std for every z-bin.
* Plot the redshift distributions.

Results
-------
2mpz  : w = 1.0012677579134883 +/- 0.00095770892183479800
wisc1 : w = 1.0004753794174424 +/- 0.00035192981400517946
wisc2 : w = 1.0003905203637462 +/- 0.00029281910671999460
wisc3 : w = 1.0015899531635275 +/- 0.00145711261063026880
wisc4 : w = 1.000642112260229 +/- 0.000489068417322734400
wisc5 : w = 1.0001195698677399 +/- 0.00106238073747336920
"""
import numpy as np
from funcs import Likelihood

# names of z-bins
zbins = ["2mpz"] + ["wisc%d" % b for b in range(1, 6)]

for zbin in zbins:
    # if zbin != "wisc5": continue
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
