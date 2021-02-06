"""
Contour plot of the photo-z to spec-z matching of the galaxies.
"""
# move to parent dir
import os, sys
THIS_PATH = os.path.dirname(os.path.realpath(__file__))
NEW_PATH = "/".join(THIS_PATH.split("/")[:-1])
os.chdir(NEW_PATH)
sys.path.append(NEW_PATH)
#########################
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import inferno
from DIR import xref

# sample selection
fname_data = "data/cats/2MPZ_FULL_wspec_coma_complete.fits"
fname_mask = "data/mmaps/ask_v3.fits"
q = xref(fname_data)
q.remove_galplane(fname_mask, "L", "B")
q.cutoff("ZPHOTO", [-1.00, 0.10])
q.cutoff("ZSPEC", -999)
xcat = q.cat_fid

# plot params
z_phot, z_spec = xcat["ZPHOTO"], xcat["ZSPEC"]
rng = [z_phot.min(), z_phot.max()]
nbins = 40

## Plot
# setup
fig, ax = plt.subplots(figsize=(9, 7))
ax.set_xlabel(r"$ z_{\rm{phot}} $", fontsize=14)
ax.set_ylabel(r"$ z_{\rm{spec}} $", fontsize=14)
ax.set_xlim(rng[0], rng[1])
ax.set_ylim(rng[0], rng[1])
fig.tight_layout()
# 2d-histogram
Z, X, Y, mg = ax.hist2d(z_phot, z_spec, bins=nbins,
                        range=[rng, rng], cmap=inferno)
X = 0.5*(X[:-1] + X[1:])
Y = 0.5*(Y[:-1] + Y[1:])
# honeycomb histogram
ax.hexbin(z_phot, z_spec, gridsize=nbins, extent=rng*2, cmap=inferno)
# y=x ref line
ax.plot(np.arange(2), c="darkred", ls="--", lw=1)
# isocontours
ax.contour(X, Y, Z.T, levels=[68, 95], colors="white", alpha=0.7)
# colourbar
cb = fig.colorbar(mg, ax=ax)
cb.set_label("number of galaxies", fontsize=14)
# etc
ax.set_xlim(0.0, 0.1)
ax.set_ylim(0.0, 0.1)
ax.set_aspect(1)
fig.tight_layout()
fig.savefig("img/zph_zsp.pdf", bbox_inches="tight")
