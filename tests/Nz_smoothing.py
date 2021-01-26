"""
Test of the Savgol filter applied to the finely binned N(z) DIR data.
"""
import os
os.chdir("..")
import numpy as np
import matplotlib.pyplot as plt
from funcs import Likelihood

# load DIR
f = np.load("out/DIR.npz")
z_mid, Nz = f["z_arr"], f["nz_arr"]

# load JKs
Nz_jk = []
diff_sq = 0
jk_id = 0
while True:
    try:
        f = np.load("out/DIR_jk%s.npz" % jk_id)
        nz, _ = f["nz_arr"], f["z_arr"]
        diff_sq += (nz-Nz)**2
        jk_id += 1
    except FileNotFoundError:
        break

dNz = np.sqrt(jk_id/(jk_id-1) * diff_sq)
l = Likelihood(z_mid, Nz, dNz)

fig, ax = plt.subplots()
ax.set_xlabel("z", fontsize=16)
ax.set_ylabel("N(z)", fontsize=16)
ax.grid(ls=":")
ax.errorbar(l.z, l.Nz, fmt="k.")
ax.plot(l.z, l.Nz_smooth, "r-", lw=3)
fig.savefig("tests/savgol_filter.pdf", bbox_inches="tight")
