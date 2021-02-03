"""
Compute and save the N(z) of the 2MPZ and WIxSC bins.
(Optional): Run jackknives to estimate the N(z) error.
"""
import numpy as np
from DIR import DIR_cross_match, nz_from_weights

# global
step = 0.001  # bin width
bins = np.arange(0, 1+step, step=step)
Njk = 100


## 2MPZ ##
# load
print("2MPZ")
fname_data = "data/2MPZ_FULL_wspec_coma_complete.fits"
fname_mask = "data/mask_v3.fits"
q = DIR_cross_match(fname_data)
q.remove_galplane(fname_mask, "L", "B")
q.cutoff("ZPHOTO", [-1.00, 0.10])
cat = q.cat_fid
q.cutoff("ZSPEC", -999)
xcat = q.cat_fid

# compute N(z)
weights = np.load("out/weights_2mpz.npz")["weights"]
nz_from_weights(xcat, weights, bins=bins, z_col="ZSPEC",
                save="out/DIR_2mpz")

# run JKs
use = len(xcat)//Njk * Njk  # effective length of xcat for the JKs
idx = np.arange(len(xcat[:use]))
np.random.shuffle(idx)  # shuffle indices

for jk in range(Njk):
    pre_jk = "out/DIR_2mpz_jk%s" % jk
    indices = np.delete(idx, idx[jk::Njk])
    nz_from_weights(xcat[:use],
                    weights[:use],
                    bins=bins,
                    z_col="ZSPEC",
                    indices=indices,
                    save=pre_jk)


## WIxSC ##
fname_data = "data/wiseScosPhotoz160708.csv"
fname_mask = "data/mask_v3.fits"

for b in range(1, 6):
    print("WIxSC bin %d" % b)
    # load
    fname = fname_data.split(".")[0] + "_bin%d.csv" % b
    q = DIR_cross_match(fname)
    q.remove_galplane(fname_mask, "l", "b")
    cat = q.cat_fid
    q.cutoff("Zspec", -999)
    xcat = q.cat_fid

    # compute N(z)
    weights = np.load("out/weights_wisc%d.npz" % b)["weights"]
    nz_from_weights(xcat, weights, bins=bins, z_col="Zspec",
                    save="out/DIR_wisc%i" % b)

    # run JKs
    use = len(xcat)//Njk * Njk  # effective length of xcat for the JKs
    idx = np.arange(len(xcat[:use]))
    np.random.shuffle(idx)  # shuffle indices

    for jk in range(Njk):
        pre_jk = "out/DIR_wisc%d_jk%d" % (b, jk)
        indices = np.delete(idx, idx[jk::Njk])
        nz_from_weights(xcat[:use],
                        weights[:use],
                        bins=bins,
                        z_col="Zspec",
                        indices=indices,
                        save=pre_jk)
