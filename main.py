import numpy as np
from DIR import DIR_cross_match, DIR_weights, nz_from_weights
from funcs import Likelihood

# sample selection
fname_data = "data/2MPZ_FULL_wspec_coma_complete.fits"
fname_mask = "data/mask_v3.fits"
q = DIR_cross_match(fname_data)  # size: 928352
q.remove_galplane(fname_mask, "L", "B")  # size: 716078
q.cutoff("ZPHOTO", [-1.00, 0.10])  # size: 476422
cat = q.cat_fid  # photo-z sample
q.cutoff("ZSPEC", -999)  # size: 216751
xcat = q.cat_fid  # spec-z sample

# spectroscopic sample weights
colors = ["JCORR", "HCORR", "KCORR", "W1MCORR",
          "W2MCORR", "BCALCORR", "RCALCORR", "ICALCORR"]
weights, _ = DIR_weights(xcat, cat, colors, save="out/weights")  # expensive
weights = np.load("out/weights.npz")["weights"]  # load weights

# entire sample
prefix = "out/DIR"
step = 0.001  # bin width
bins = np.arange(0, 1+step, step=step)
Nz, z_mid = nz_from_weights(xcat, weights, bins=bins,
                            save=prefix, full_output=True)
# load
f = np.load("out/DIR.npz")
z_mid, Nz = f["z_arr"], f["nz_arr"]

## jackknives ##
Njk = 100
print(f"Due to Njk={Njk}, {len(xcat)%Njk} entries are discarded.")
use = len(xcat)//Njk * Njk  # effective length of xcat for the JKs
idx = np.arange(len(xcat[:use]))
np.random.shuffle(idx)  # shuffle indices

diff_sq = 0  # (Nz - <Nz>)^2
for i in range(Njk):
    pre_jk = prefix + "_jk%s" % i
    indices = np.delete(idx, idx[i::Njk])
    res, _ = nz_from_weights(xcat[:use], weights[:use],
                              indices=indices, bins=bins,
                              save=pre_jk, full_output=True)
    diff_sq += (res-Nz)**2


# jackknives load
diff_sq = 0
jk_id = 0
while True:
    try:
        f = np.load("out/DIR_jk%s.npz" % jk_id)
        diff_sq += (f["nz_arr"]-Nz)**2
        jk_id += 1
    except FileNotFoundError:
        break

dNz = np.sqrt(Njk/(Njk-1) * diff_sq)

l = Likelihood(z_mid, Nz, dNz)
w, dw = l.prob()
print(f"w = {w} +- {dw}")

"""
2MPZ : w = 1.0005476670854323 +- 0.00048031555773951184
"""
