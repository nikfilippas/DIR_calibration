from tqdm import tqdm
import numpy as np
from scipy.optimize import curve_fit
from DIR import DIR_cross_match, DIR_weights, nz_from_weights
from funcs import width_func, nearest_divisor, Likelihood

# sample selection
fname_data = "data/2MPZ_FULL_wspec_coma_complete.fits"
fname_mask = "data/mask_v3.fits"
q = DIR_cross_match(fname_data)  # size: 928352
q.remove_galplane(fname_mask, "SUPRA", "SUPDEC")  # size: 716055
q.cutoff("ZPHOTO", [0.05, 0.10])  # size: 360164
q.cutoff("ZSPEC", -999)  # size: 141552
xcat = q.cat_fid

# spectroscopic sample weights
# colors = ["JCORR", "HCORR", "KCORR", "W1MCORR",
#           "W2MCORR", "BCALCORR", "RCALCORR", "ICALCORR"]
# weights, _ = DIR_weights(xcat, cat, colors, save="out/weights")  # expensive
weights = np.load("out/weights.npz")["weights"]  # load weights

# entire sample
prefix = "out/DIR"
# step = 0.001  # bin width
# bins = np.arange(0, 1+step, step=step)
# Nz, z_mid = nz_from_weights(xcat, weights, bins=bins,
#                             save=prefix, full_output=True)
# load
f = np.load("out/DIR.npz")
z_mid, Nz = f["z_arr"], f["nz_arr"]

## jackknives ##

# Given the cross-mathced catalogue, we almost cetrainly cannot split
# it into ``jk["num"]`` equally-sized bins, so we would have to assign
# weights to each jackknife. We overcome this by finding the divisor
# of the size of the catalogue, which is closest to ``jk["num"]``.

N_jk = 100
N_jk = nearest_divisor(N_jk, len(xcat))  # effective number of JKs
print("Jackknife size:\t%d" % (len(xcat)/N_jk))
print("# of jackknives:\t×%d" % N_jk)
print("Catalogue size:\t=%d" % len(xcat))
# idx = np.arange(len(xcat))
# np.random.shuffle(idx)  # shuffle indices

# Nz_jk = []
# diff_sq = 0  # (Nz - <Nz>)^2
# for i in range(N_jk):
#     pre_jk = prefix + "_jk%s" % i
#     indices = np.delete(idx, idx[i::N_jk])
#     res, _ = nz_from_weights(xcat, weights,
#                               indices=indices, bins=bins,
#                               save=pre_jk, full_output=True)
#     Nz_jk.append(res)
#     diff_sq += (res-Nz)**2

# jackknives load
Nz_jk = [np.load(prefix+"_jk%s.npz"%i)["nz_arr"] for i in range(N_jk)]
diff_sq = np.sum([(nz-Nz)**2 for nz in Nz_jk], axis=0)
dNz = np.sqrt(N_jk/(N_jk-1) * diff_sq)

l = Likelihood(z_mid, Nz, dNz)
w, dw = l.prob()
print(f"w = {w} +- {dw}")
