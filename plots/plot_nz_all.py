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
import healpy as hp
hp.disable_warnings()
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

# global
z_arr = np.linspace(0, 1, 1000)
normed = False  # normed or absolute N(z)'s
# open every cat and store number of galaxies
if not normed:
    print("Finding catalogue sizes...")
    # from DIR import xref
    # fname_mask = "data/maps/mask_v3.fits"
    # sizes = []
    # # 2MPZ
    # fname_data = "data/cats/2MPZ_FULL_wspec_coma_complete.fits"
    # q = xref(fname_data)
    # q.remove_galplane(fname_mask, "L", "B")
    # q.cutoff("ZPHOTO", [-1.00, 0.10])
    # sizes.append(len(q.cat_fid))
    # # WIxSC
    # fname_data = "data/cats/wiseScosPhotoz160708.csv"
    # for i in range(1, 6):
    #     fname = fname_data.split(".")[0] + "_bin%d.csv" % i
    #     q = xref(fname)
    #     q.remove_galplane(fname_mask, "l", "b")
    #     sizes.append(len(q.cat_fid))
    sizes = [476422, 3458260, 3851322, 4000017, 3412366, 1296810]
    fsky = np.mean(hp.read_map("data/maps/mask_v3.fits"))
    area = 4*np.pi*fsky*(180/np.pi)**2
    


## N(z)'s in 1909.09102 ##
zm, Nzm = [], []
f = np.loadtxt("data/dndz/2MPZ.txt").T
zm.append(f[0])
Nzm.append(f[1])
for b in range(1, 6):
    f = np.loadtxt("data/dndz/WISC_bin%s.txt" % b).T
    zm.append(f[0])
    Nzm.append(f[1])
Nzm = [N*s/area for N, s in zip(Nzm, sizes)] if not normed else Nzm

## DIR ##
# names of z-bins
zbins = ["2mpz"] + ["wisc%d" % b for b in range(1, 6)]
zd, Nzd = [], []
for zbin in zbins:
    f = np.loadtxt("data/dndz/%s_DIR.txt" % zbin).T
    zd.append(f[0])
    Nzd.append(f[1])
Nzd = [N*s/area for N, s in zip(Nzd, sizes)] if not normed else Nzd


## Plot ##
# line colours
cols = ["r"]
cm_wisc = cm.get_cmap("Blues")
for i in range(1, 6):
    cols.append(cm_wisc(0.2 + 0.8*i/5))

fig, ax = plt.subplots(figsize=(12, 6))
if normed:
    ax.set_ylabel("N(z)", fontsize=16)
else:
    ax.set_ylabel(r"$dN/dz\,d\Omega\,\,[10^2\,{\rm deg}^{-2}]$",fontsize=14)
ax.set_xlabel("z", fontsize=16)
ax.set_xlim(0.0, 0.5)
fig.tight_layout()

# N(z) in 1909.09102
for col, zz, NN in zip(cols, zm, Nzm):
    ax.plot(zz, NN/100, c=col, lw=2, ls="--")

# DIR-calibrated N(z)'s
for zbin, col, zz, NN in zip(zbins, cols, zd, Nzd):
    ax.plot(zz, NN/100, c=col, lw=2, label=zbin)

ax.set_ylim(0,)
ax.legend(loc="upper right", fontsize=14)
fname_out = "img/nz_all.pdf"
if normed: fname_out = "_norm.".join(fname_out.split("."))
fig.savefig(fname_out, bbox_inches="tight")
