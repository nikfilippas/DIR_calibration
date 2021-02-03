"""
Plot histogram of the distribution of z_spec-z_phot.
"""
import os
os.chdir("..")
import numpy as np
import matplotlib.pyplot as plt
from DIR import xref

# sample selection
fname_data = "data/cats/2MPZ_FULL_wspec_coma_complete.fits"
fname_mask = "data/maps/mask_v3.fits"
q = xref(fname_data)
q.remove_galplane(fname_mask, "L", "B")
q.cutoff("ZPHOTO", [-1.00, 0.10])
q.cutoff("ZSPEC", -999)
xcat = q.cat_fid

# plot
z_spec, z_phot = xcat["ZSPEC"], xcat["ZPHOTO"]
lon, lat = xcat["L"], xcat["B"]
dz = z_spec - z_phot

fig, ax = plt.subplots()
step = 0.002
bins = np.arange(dz.min(), dz.max()+step, step=step)
N, _, _ = ax.hist(dz, bins=bins, histtype="step", color="k", lw=3)
ax.axvline(np.median(dz), c="r", ls=":", lw=2)
ax.set_xlabel(r"$\Delta z=z_{\rm{sp}}-z_{\rm{ph}}$", fontsize=14)
ax.set_ylabel("number of galaxies", fontsize=14)
zrange = bins[np.where(N > N.max()/100)[0].take([0, -1])]
ax.set_xlim(zrange[0], zrange[1]+step)
fig.savefig("img/dz_hist.pdf", bbox_inches="tight")
