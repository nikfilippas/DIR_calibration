"""
* Calculate the width mean and std for every z-bin.
* Plot the redshift distributions.

Results
-------
2mpz :  w = 1.0006932628036858 +/- 0.0005790971268628165
wisc1 : w = 1.0005493937785184 +/- 0.0004833475255687241
wisc2 : w = 1.0007753932979970 +/- 0.0006381720859299973
wisc3 : w = 1.0007639012371237 +/- 0.0007071641836272273
wisc4 : w = 1.0002472701429062 +/- 0.0002300252207527303
wisc5 : w = 1.0007892595210480 +/- 0.0006970804970449077
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
