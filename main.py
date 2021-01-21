from tqdm import tqdm
import numpy as np
from astropy.io import fits
from scipy.optimize import curve_fit
from DIR import DIR_weights, nz_from_weights, width_func

# global parameters
fname = "2MPZ_FULL_wspec_coma_complete.fits"
cat = fits.open(fname)[1].data
xcat = cat[np.where(cat["ZSPEC"] != -999.)[0]]
cols = ["JCORR", "HCORR", "KCORR", "W1MCORR",
        "W2MCORR", "BCALCORR", "RCALCORR", "ICALCORR"]

# spectroscopic sample weights
weights, _ = DIR_weights(fname, cols, save="out/weights")

# entire sample
step = 0.01  # bin width
bins = np.arange(0, 1+step, step=step)
prefix = "out/DIR"
Nz, z_mid = nz_from_weights(xcat, weights, bins=bins, save=prefix, full_output=True)

# jackknives create
num_jk = 1000
jk={"do":True,"thin":0.01,"jk_id":None,"replace":False}
Nz_jk = []
for jk_id in tqdm(range(num_jk)):
    jk["jk_id"] = jk_id
    res, _ = nz_from_weights(xcat, weights, bins=bins,
                             save=prefix, full_output=True,
                             jk=jk)
    Nz_jk.append(res)

# # jackknives load
# Nz_jk = []
# for jk_id in tqdm(range(num_jk)):
#     f = np.load(prefix+"_jk%s.npz" % jk_id)
#     Nz_jk.append(f["nz_arr"])

# width
w, dw = [], []
z_avg = np.average(z_mid, weights=Nz)
fitfunc = lambda Nz, width: width_func(z_mid, Nz, width, z_avg)
for N in tqdm(Nz_jk):
    popt, pcov = curve_fit(fitfunc, N, Nz, p0=[1.], bounds=(0.8, 1.2))
    w.append(popt[0])
    dw.append(np.sqrt(pcov.squeeze()))
w, dw = np.array(w), np.array(dw)

print("Width mean: ", w.mean())
print("Width std: ", w.std())
print("Error mean: ", dw.mean())
print("JK sample size: ", w.size)
