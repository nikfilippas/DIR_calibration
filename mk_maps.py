"""
Create maps from N(z) catalogues.
"""
import healpy as hp
import numpy as np
from DIR import DIR_cross_match

# global
nside = 512
fname_mask = "data/mask_v3.fits"
hp.disable_warnings()
mask = hp.ud_grade(hp.read_map(fname_mask), nside)

## 2MPZ ##
print("2MPZ")
fname_data = "data/maps/2MPZ_FULL_wspec_coma_complete.fits"
q = DIR_cross_match(fname_data)
q.remove_galplane(fname_mask, "L", "B")
q.cutoff("ZPHOTO", [-1.00, 0.10])
q.cutoff("ZSPEC", -999)
xcat = q.cat_fid

ipix = hp.ang2pix(nside, xcat["L"], xcat["B"], lonlat=True)
npix = hp.nside2npix(nside)
ngal = np.bincount(ipix, minlength=npix)
n_mean = np.sum(ngal*mask) / np.sum(mask)
delta = mask*(ngal/n_mean - 1)
hp.write_map("data/map_2mpz.fits", delta, overwrite=True)


## WIxSC ##
zbins = ["wisc%d" % b for b in range(1, 6)]
fname_data = "data/maps/wiseScosPhotoz160708.csv"
for i, zbin in enumerate(zbins):
    print("WIxSC bin %d" % i)
    fname = fname_data.split(".")[0] + "_bin%d.csv" % i
    q = DIR_cross_match(fname)
    q.remove_galplane(fname_mask, "l", "b")
    cat = q.cat_fid
    q.cutoff("Zspec", -999)
    xcat = q.cat_fid

    ipix = hp.ang2pix(nside, xcat["l"], xcat["b"], lonlat=True)
    npix = hp.nside2npix(nside)
    ngal = np.bincount(ipix, minlength=npix)
    n_mean = np.sum(ngal*mask) / np.sum(mask)
    delta = mask*(ngal/n_mean - 1)
    hp.write_map("data/map_%s.fits" % zbin, delta, overwrite=True)
