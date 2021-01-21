from tqdm import tqdm
import numpy as np
from astropy.io import fits
from DIR import DIR_weights, nz_from_weights

fname = "2MPZ_FULL_wspec_coma_complete.fits"
cat = fits.open(fname)[1].data
xcat = cat[np.where(cat["ZSPEC"] != -999.)[0]]
cols = ["JCORR", "HCORR", "KCORR", "W1MCORR",
        "W2MCORR", "BCALCORR", "RCALCORR", "ICALCORR"]

weights, _ = DIR_weights(fname, cols, save="out/weights")

# entire sample
bins = np.arange(0, 1, step=0.01)
prefix = "out/DIR"
Nz, z_mid = nz_from_weights(xcat, weights, bins=bins, save=prefix, full_output=True)


jk={"num":0,"thin":0.01,"jk_id":None,"replace":False}
for jk_id in tqdm(range(1000)):
    jk["jk_id"] = jk_id
    nz_from_weights(xcat, weights, bins=bins, save=prefix, jk=jk)
