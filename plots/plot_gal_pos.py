"""
Plot the positions of the cross-mathced galaxies in the sky.
"""
import os
os.chdir("..")
import healpy as hp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import gray, cividis
from DIR import DIR_cross_match

# sample selection
fname_data = "data/2MPZ_FULL_wspec_coma_complete.fits"
fname_mask = "data/mask_v3.fits"
q = DIR_cross_match(fname_data)
q.remove_galplane(fname_mask, "L", "B")
q.cutoff("ZPHOTO", [-1.00, 0.10])
q.cutoff("ZSPEC", -999)
xcat = q.cat_fid

# plot
z_spec, z_phot = xcat["ZSPEC"], xcat["ZPHOTO"]
lon, lat = xcat["L"], xcat["B"]
dz = z_spec - z_phot
thin = 1
Z = dz[::thin]

plt.figure(num="mollview", figsize=(20,10))
hp.mollview(q.mask, fig="mollview", title="",
            notext=True, xsize=2000,
            cmap=gray, cbar=False)

pp = np.percentile(Z, [5, 95])
# clip delta-z
Z[Z < pp[0]] = pp[0]
Z[Z > pp[1]] = pp[1]
# reduce to unity
norm = Z/(2*np.abs(Z).max()) + 0.5
colors = cividis(norm)
hp.projscatter(lon[::thin], lat[::thin], lonlat=True,
               color=colors, s=2)
hp.graticule(coord="G", color="r", ls=":")
plt.savefig("img/2MPZ_zspec_positions.pdf", bbox_inches="tight")
