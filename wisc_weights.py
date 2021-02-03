import numpy as np
from DIR import DIR_cross_match, DIR_weights, nz_from_weights

# global
fname_data = "data/wiseScosPhotoz160708.csv"
fname_mask = "data/mask_v3.fits"
colors = ["W1c", "W2c", "Bcc", "Rcc"]

for i in range(1, 6):
    # sample selection
    fname = fname_data.split(".")[0] + "_bin%d.csv" % i
    q = DIR_cross_match(fname)
    q.remove_galplane(fname_mask, "l", "b")
    cat = q.cat_fid
    q.cutoff("Zspec", -999)
    xcat = q.cat_fid

    # spectroscopic sample weights
    weights, _ = DIR_weights(xcat, cat, colors, save="out/weights_wisc%d" % i)

