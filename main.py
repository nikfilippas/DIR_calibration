from tqdm import tqdm
import numpy as np
from scipy.optimize import curve_fit
from DIR import DIR_cross_match, nz_from_weights
from funcs import width_func, nearest_divisor

# sample selection
fname_data = "2MPZ_FULL_wspec_coma_complete.fits"
fname_mask = "mask_v3.fits"
q = DIR_cross_match(fname_data)  # size: 928352
q.remove_galplane(fname_mask, "SUPRA", "SUPDEC")  # size: 716055
q.cutoff("ZPHOTO", [0.05, 0.10])  # size: 360164
q.cutoff("ZSPEC", -999)  # size: 141552
xcat = q.cat_fid

# spectroscopic sample weights
colors = ["JCORR", "HCORR", "KCORR", "W1MCORR",
          "W2MCORR", "BCALCORR", "RCALCORR", "ICALCORR"]
# from DIR import DIR_weights
# weights, _ = DIR_weights(xcat, cat, colors, save="out/weights")  # expensive
weights = np.load("out/weights.npz")["weights"]  # load weights

# entire sample
step = 0.005  # bin width
bins = np.arange(0, 1+step, step=step)
prefix = "out/DIR"
Nz, z_mid = nz_from_weights(xcat, weights, bins=bins,
                            save=prefix, full_output=True)

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
idx = np.arange(len(xcat))
np.random.shuffle(idx)  # shuffle indices

Nz_jk = []
for i in range(N_jk):
    pre_jk = prefix + "_jk%s" % i
    indices = np.delete(idx, idx[i::N_jk])
    res, _ = nz_from_weights(xcat, weights,
                             indices=indices, bins=bins,
                             save=pre_jk, full_output=True)
    Nz_jk.append(res)

# jackknives load
# Nz_jk = [np.load(prefix+"_jk%s.npz"%i)["nz_arr"] for i in range(N_jk)]

# width
w = []
z_avg = np.average(z_mid, weights=Nz)
fitfunc = lambda Nz, width: width_func(z_mid, Nz, width, z_avg)
for N in tqdm(Nz_jk):
    popt, pcov = curve_fit(fitfunc, N, Nz, p0=[1.], bounds=(0.8, 1.2))
    w.append(popt[0])

dw = np.sqrt(N_jk/(N_jk-1) * np.sum((w - np.mean(w))**2))  # 0.05332289703791278
print("\n", dw)
