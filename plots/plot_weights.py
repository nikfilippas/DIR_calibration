"""
Plot raw N(z)'s for the entire photo-z sample and for
the training sample only. Weight the training sample
to retrieve original N(z) of entire sample.
"""
import os
#os.chdir("..")
import numpy as np
import matplotlib.pyplot as plt
from DIR import xref

# global
#hp.disable_warnings()
fname_mask = "data/maps/mask_v3.fits"
step = 0.001  # bin width
bins = np.arange(0, 1+step, step=step)

fig, ax = plt.subplots(2, 3, figsize=(17, 9))
[a.set_ylabel("flux", fontsize=16) for a in ax[:, 0]]
[a.set_xlabel("z", fontsize=16) for a in ax[1]]
ax = ax.flatten()
fig.tight_layout()


## 2MPZ ##
fname_data = "data/cats/2MPZ_FULL_wspec_coma_complete.fits"
q = xref(fname_data)
q.remove_galplane(fname_mask, "L", "B")
q.cutoff("ZPHOTO", [-1.00, 0.10])
cat = q.cat_fid
q.cutoff("ZSPEC", -999)
xcat = q.cat_fid

weights = np.load("out/weights_2mpz.npz")["weights"]

ax[0].hist(cat.ZPHOTO, bins=bins, density=True,
           histtype="step", lw=2, label="photo-z")
ax[0].hist(xcat.ZPHOTO, bins=bins, density=True,
           histtype="step", lw=2, label="photo-z of training set")
ax[0].hist(xcat.ZPHOTO, bins=bins, weights=weights, density=True,
           histtype="step", lw=2, label="photo-z training set + weights")
ax[0].set_xlim(cat.ZPHOTO.min(), cat.ZPHOTO.max())


## WIxSC ##
for i in range(1, 6):
    fname_data = "data/cats/wiseScosPhotoz160708_bin%s.csv" % i
    q = xref(fname_data)
    q.remove_galplane(fname_mask, "l", "b")
    cat = q.cat_fid
    q.cutoff("Zspec", -999)
    xcat = q.cat_fid

    weights = np.load("out/weights_wisc%s.npz" % i)["weights"]

    ax[i].hist(cat.zPhoto_Corr, bins=bins, density=True,
               histtype="step", lw=2, label="photo-z")
    ax[i].hist(xcat.zPhoto_Corr, bins=bins, density=True,
               histtype="step", lw=2, label="photo-z of training set")
    ax[i].hist(xcat.zPhoto_Corr, bins=bins, weights=weights, density=True,
               histtype="step", lw=2, label="photo-z training set + weights")
    ax[i].set_xlim(cat.zPhoto_Corr.min(), cat.zPhoto_Corr.max())

[a.legend(loc="upper left") for a in ax[:1]]
fig.savefig("img/weights.pdf", bbox_inches="tight")
