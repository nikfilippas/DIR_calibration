"""
Create and save the redshift distributions.
"""
import numpy as np
from funcs import Likelihood

# global
z = np.arange(0.0005, 1.0000, 0.001)
zbins = ["2mpz"] + ["wisc%d" % b for b in range(1, 6)]

for zbin in zbins:
    # load N(z)
    f = np.load("out/DIR_%s.npz" % zbin)
    z_mid, Nz = f["z_arr"], f["nz_arr"]

    l = Likelihood(z_mid, Nz)
    X = np.column_stack((z, l.Nzi(z)))
    np.savetxt("data/dndz/"+zbin+"_DIR.txt", X)
