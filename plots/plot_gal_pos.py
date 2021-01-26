"""
Plot the positions of the cross-mathced galaxies in the sky.
"""
import os
os.chdir("..")
import healpy as hp
import matplotlib.pyplot as plt
from matplotlib.cm import gray, bwr_r
from DIR import DIR_cross_match

# sample selection
fname_data = "2MPZ_FULL_wspec_coma_complete.fits"
fname_mask = "mask_v3.fits"
q = DIR_cross_match(fname_data)  # size: 928352
q.remove_galplane(fname_mask, "SUPRA", "SUPDEC")  # size: 716055
q.cutoff("ZPHOTO", [0.05, 0.10])  # size: 360164
q.cutoff("ZSPEC", -999)  # size: 141552
xcat = q.cat_fid

# plot
z_spec, z_phot = xcat["ZSPEC"], xcat["ZPHOTO"]
ra, dec = xcat["SUPRA"], xcat["SUPDEC"]
dz = z_spec - z_phot
thin = 50
plt.figure(num="mollview", figsize=(12,7))
hp.mollview(q.mask, fig="mollview", title="",
            notext=True, xsize=2000,
            cmap=gray, cbar=False)
alphas = z_spec[::thin] / z_spec.mean()
transp = dz[::thin]/(2*dz.std()) + 2*dz.std()
colors = bwr_r(transp)
# colors[:,-1] = alphas
hp.projscatter(ra[::thin], dec[::thin], coord="C", lonlat=True,
               color=colors, s=3)
hp.graticule(coord="C", color="y", ls=":")
plt.savefig("img/2MPZ_zspec_positions.pdf", bbox_inches="tight")
