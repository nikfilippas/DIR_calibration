from tqdm import tqdm
import numpy as np
from astropy.io import fits
from DIR import DIR_weights, do_jk

fname = "2MPZ_FULL_wspec_coma_complete.fits"
cat = fits.open(fname)[1].data
xcat = cat[np.where(cat["ZSPEC"] != -999.)[0]]
cols = ["JCORR", "HCORR", "KCORR", "W1MCORR",
        "W2MCORR", "BCALCORR", "RCALCORR", "ICALCORR"]

weights, _ = DIR_weights(fname, cols, save="out/weights")

bins = np.arange(0, 1, step=0.01)
for jk_id in tqdm(range(1000)):
    do_jk(xcat, weights, bins=bins, thin=0.99, jk_id=jk_id, save="out/")
