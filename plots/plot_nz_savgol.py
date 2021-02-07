"""
Plot the nominal and smoothed redshift distributions.
"""
# move to parent dir
import os, sys
THIS_PATH = os.path.dirname(os.path.realpath(__file__))
NEW_PATH = "/".join(THIS_PATH.split("/")[:-1])
os.chdir(NEW_PATH)
sys.path.append(NEW_PATH)
#########################
import numpy as np
import matplotlib.pyplot as plt
from funcs import Likelihood

# names of z-bins
zbins = ["2mpz"] + ["wisc%d" % b for b in range(1, 6)]

fig, ax = plt.subplots(2, 3, figsize=(12, 8))
[a.set_ylabel("N(z)", fontsize=16) for a in ax[:, 0]]
[a.set_xlabel("z", fontsize=16) for a in ax[1]]
ax = ax.flatten()
fig.tight_layout()

for a, zbin in zip(ax, zbins):
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
            a.plot(f["z_arr"], f["nz_arr"], "grey", alpha=0.2)
            Njk += 1
        except FileNotFoundError:
            break

    # run likelihood
    dNz = np.sqrt(Njk/(Njk-1) * diff_sq)
    l = Likelihood(z_mid, Nz, dNz)
    a.errorbar(l.z, l.Nz, l.dNz, fmt="k.", label=zbin)
    a.plot(l.z, l.Nz_smooth, "r-", lw=3)
    a.set_xlim(l.z[0], l.z[-1])
    a.legend(loc="upper right")

fig.savefig("img/nz_savgol.pdf", bbox_inches="tight")
