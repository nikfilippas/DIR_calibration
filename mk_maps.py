"""
Create maps from N(z) catalogues.
"""
import healpy as hp
import numpy as np
from DIR import xref

# global
nside = 512
fname_mask = "data/maps/mask_v3.fits"
hp.disable_warnings()
mask = hp.ud_grade(hp.read_map(fname_mask), nside)
is_delta_g = False

## WIxSC ##
zbins = ["wisc%d" % b for b in range(1, 6)]
fname_data = "data/cats/wiseScosPhotoz160708.csv"
for i, zbin in enumerate(zbins):
    print("WIxSC bin %d" % (i+1))
    fname = fname_data.split(".")[0] + "_bin%d.csv" % (i+1)
    q = xref(fname)
    q.remove_galplane(fname_mask, "l", "b")
    cat = q.cat_fid

    ipix = hp.ang2pix(nside, cat["l"], cat["b"], lonlat=True)
    npix = hp.nside2npix(nside)
    ngal = np.bincount(ipix, minlength=npix)
    if is_delta_g:
        n_mean = np.sum(ngal*mask) / np.sum(mask)
        ngal = mask*(ngal/n_mean - 1)
        zbin = "Delta_g_" + zbin
    hp.write_map("data/maps/map_%s.fits" % zbin, ngal, overwrite=True)


## 2MPZ ##
zbin = "2mpz"
print("2MPZ")
fname_data = "data/cats/2MPZ_FULL_wspec_coma_complete.fits"
q = xref(fname_data)
q.remove_galplane(fname_mask, "L", "B")

# fiducial cut z < 0.10
q.cutoff("ZPHOTO", [-1.00, 0.10])
cat = q.cat_fid
ipix = hp.ang2pix(nside, cat["L"], cat["B"], lonlat=True)
npix = hp.nside2npix(nside)
ngal = np.bincount(ipix, minlength=npix)
if is_delta_g:
    n_mean = np.sum(ngal*mask) / np.sum(mask)
    ngal = mask*(ngal/n_mean - 1)
    zbin = "Delta_g_" + zbin
hp.write_map("data/maps/map_%s.fits" % zbin, ngal, overwrite=True)

# compare with no z-cut
cat = q.cat  # original cat without photo-z cuts
ipix = hp.ang2pix(nside, cat["l"], cat["b"], lonlat=True)
npix = hp.nside2npix(nside)
ngal = np.bincount(ipix, minlength=npix)
if is_delta_g:
    n_mean = np.sum(ngal*mask) / np.sum(mask)
    delta = mask*(ngal/n_mean - 1)
    zbin = "Delta_g_" + zbin
hp.write_map("data/maps/map_%s_full.fits" % zbin, ngal, overwrite=True)
