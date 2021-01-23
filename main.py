from tqdm import tqdm
import healpy as hp
from healpy.rotator import Rotator
import numpy as np
from sympy import divisors
from astropy.io import fits
from scipy.optimize import curve_fit
from DIR import DIR_weights, nz_from_weights, width_func

## Preamble :: Sample Selection
# global parameters
fname_data = "2MPZ_FULL_wspec_coma_complete.fits"
fname_mask = "mask_v3.fits"
colors = ["JCORR", "HCORR", "KCORR", "W1MCORR",
          "W2MCORR", "BCALCORR", "RCALCORR", "ICALCORR"]
cat = fits.open(fname_data)[1].data  # size: 928352
# z-cutoff at 0.1
cat = cat[(cat["ZPHOTO"] >= 0.05) & (cat["ZPHOTO"] <= 0.1)]  # size: 460301
# mask out galactic plane
mask = hp.read_map(fname_mask, dtype=float)
mask = Rotator(coord=["G", "C"]).rotate_map_alms(mask, use_pixel_weights=False)
nside = hp.npix2nside(mask.size)
ipix = hp.ang2pix(nside, cat["SUPRA"], cat["SUPDEC"], lonlat=True)
cat = cat[mask[ipix] > 0.5]  # size: 360164
# cross-match
xcat = cat[np.where(cat["ZSPEC"] != -999.)[0]]  # size: 141552
##

# spectroscopic sample weights
weights, _ = DIR_weights(xcat, cat, colors, save="out/weights")

# entire sample
step = 0.01  # bin width
bins = np.arange(0, 1+step, step=step)
prefix = "out/DIR"
Nz, z_mid = nz_from_weights(xcat, weights, bins=bins,
                            save=prefix, full_output=True)

## jackknives ##

# Given the cross-mathced catalogue, we almost cetrainly cannot split
# it into ``jk["num"]`` equally-sized bins, so we would have to assign
# weights to each jackknife. We overcome this by finding the divisor
# of the size of the catalogue, which is closest to ``jk["num"]``.

N_jk = 100  # approximate number of JKs
div = np.array(divisors(len(xcat)))
N_jk = div[np.abs(div-N_jk).argmin()]  # effective number of JKs
jk_size = len(xcat)/N_jk
print("Jackknife size:\t%d" % jk_size)
print("# of jackknives:\t×%d" % N_jk)
print("Catalogue size:\t=%d" % len(xcat))
# shuffle indices
idx = np.arange(len(xcat))
np.random.shuffle(idx)

Nz_jk = []
for i in range(N_jk):
    pre_jk = prefix + "_jk%s" % i
    res, _ = nz_from_weights(xcat, weights,
                             indices=idx[i::N_jk], bins=bins,
                             save=pre_jk, full_output=True)
    Nz_jk.append(res)

# jackknives load
# Nz_jk = [np.load(prefix+"_jk%s.npz"%i)["nz_arr"] for i in range(N_jk)]

# width
w, dw = [], []
z_avg = np.average(z_mid, weights=Nz)
fitfunc = lambda Nz, width: width_func(z_mid, Nz, width, z_avg)
for N in tqdm(Nz_jk):
    popt, pcov = curve_fit(fitfunc, N, Nz, p0=[1.], bounds=(0.8, 1.2))
    w.append(popt[0])
    dw.append(np.sqrt(pcov.squeeze()))
w, dw = np.array(w), np.array(dw)
