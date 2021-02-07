"""
* Plot the smoothed redshift distributions in a single plot.
* Compare with redshift distributions in 1909.09102.
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
from matplotlib import cm
from funcs import Likelihood

# global
z_arr = np.linspace(0, 1, 1000)
normed = False  # normed or absolute N(z)'s
# open every cat and store number of galaxies
if not normed:
    print("Finding catalogue sizes...")
    from DIR import xref
    fname_mask = "data/maps/mask_v3.fits"
    sizes = []
    # 2MPZ
    fname_data = "data/cats/2MPZ_FULL_wspec_coma_complete.fits"
    q = xref(fname_data)
    q.remove_galplane(fname_mask, "L", "B")
    q.cutoff("ZPHOTO", [-1.00, 0.10])
    sizes.append(len(q.cat_fid))
    # WIxSC
    fname_data = "data/cats/wiseScosPhotoz160708.csv"
    for i in range(1, 6):
        fname = fname_data.split(".")[0] + "_bin%d.csv" % i
        q = xref(fname)
        q.remove_galplane(fname_mask, "l", "b")
        sizes.append(len(q.cat_fid))


## N(z)'s in 1909.09102 ##
zm, Nzm = [], []
f = np.loadtxt("data/dndz/2MPZ.txt").T
zm.append(f[0])
Nzm.append(f[1])
for b in range(1, 6):
    f = np.loadtxt("data/dndz/WISC_bin%s.txt" % b).T
    zm.append(f[0])
    Nzm.append(f[1])
Nzm = [N*s for N, s in zip(Nzm, sizes)] if not normed else Nzm

## DIR ##
# names of z-bins
zbins = ["2mpz"] + ["wisc%d" % b for b in range(1, 6)]

# line colours
cols = ["r"]
cm_wisc = cm.get_cmap("Blues")
for i in range(1, 6):
    cols.append(cm_wisc(0.2 + 0.8*i/5))

fig, ax = plt.subplots(figsize=(12, 6))
ax.set_ylabel("N(z)", fontsize=16)
ax.set_xlabel("z", fontsize=16)
ax.set_xlim(0.0, 0.5)
fig.tight_layout()

# N(z) in 1909.09102
for col, zz, NN in zip(cols, zm, Nzm):
    ax.plot(zz, NN, c=col, lw=2, ls="--")

for i, (col, zbin) in enumerate(zip(cols, zbins)):
    print(zbin)
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
    # error bars
    dNz = np.sqrt(Njk/(Njk-1) * diff_sq)
    # normalisation
    Nz = Nz*sizes[i] if not normed else Nz
    dNz = dNz*sizes[i] if not normed else dNz
    # run likelihood
    l = Likelihood(z_mid, Nz, dNz)
    ax.plot(z_arr, l.Nzi(z_arr), c=col, lw=2, label=zbin)

ax.set_ylim(0,)
ax.legend(loc="upper right", fontsize=14)
fig.savefig("img/nz_all_abs.pdf", bbox_inches="tight")

