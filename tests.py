import numpy as np
import healpy as hp
from healpy.rotator import Rotator
import matplotlib.pyplot as plt
from astropy.io import fits

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


## Plots
fig, ax = plt.subplots(1, 1, figsize=(10, 6))
ax.grid(ls=":")
ax.set_xlim(0.0, 0.2)
ax.set_xlabel("z", fontsize=16)
ax.set_ylabel("N(z)", fontsize=16)
fig.tight_layout()

# compare to Maciek's N(z)
m1 = np.loadtxt("/home/nick/Desktop/yxgxk/data/dndz/2MPZ_bin1.txt")
m2 = np.loadtxt("/home/nick/Desktop/yxgxk/data/dndz/2MPZ_v2_bin1.txt")
ax.plot(*m1.T, ls="--", label="2mpz_v1")
ax.plot(*m2.T, ls="-.", label="2mpz_v2")

# jackknives
jk_id = 0
while True:
    try:
        f = np.load("out/DIR_jk%s.npz" % jk_id)
        nz, z_jk = f["nz_arr"], f["z_arr"]
        ax.plot(z_jk, nz, "gray", alpha=0.07)
        jk_id += 1
    except FileNotFoundError:
        break

# catalogues
bins = np.arange(0, 0.2, 0.005)  # redshift bin edges
pz_spec, _ = np.histogram(xcat["ZSPEC"], bins=bins, density=True)
pz_phot, _ = np.histogram(xcat["ZPHOTO"], bins=bins, density=True)
z_mid = 0.5*(bins[:-1] + bins[1:])
plt.plot(z_mid, pz_phot, "navy", lw=2, alpha=0.4, label="photometric")
plt.plot(z_mid, pz_spec, "orange", lw=2, label="spectroscopic")
# DIR
f_DIR = np.load("out/DIR.npz")
plt.plot(f_DIR["z_arr"], f_DIR["nz_arr"], "green", lw=2, label="DIR")

ax.legend(loc="upper right", fontsize=12)
fig.savefig("2mpz_comparison.pdf", bbox_inches="tight")


# galaxy positions
from matplotlib.cm import gray, bwr_r
z_spec, z_phot = xcat["ZSPEC"], xcat["ZPHOTO"]
ra, dec = xcat["SUPRA"], xcat["SUPDEC"]
dz = z_spec - z_phot
thin = 50
plt.figure(num="mollview", figsize=(12,7))
hp.mollview(mask, fig="mollview", title="",
            notext=True, xsize=2000,
            cmap=gray, cbar=False)
alphas = z_spec[::thin] / z_spec.mean()
transp = dz[::thin]/(2*dz.std()) + 2*dz.std()
colors = bwr_r(transp)
# colors[:,-1] = alphas
hp.projscatter(ra[::thin], dec[::thin], coord="C", lonlat=True,
               color=colors, s=3)
hp.graticule(coord="C", color="y", ls=":")
plt.savefig("2MPZ_zspec_positions.pdf", bbox_inches="tight")


# z_phot VS z_spec
from matplotlib.cm import inferno
rng = [z_phot.min(), z_phot.max()]

fig, ax = plt.subplots()
ax.set_xlabel(r"$ z_{\rm{phot}} $", fontsize=14)
ax.set_ylabel(r"$ z_{\rm{spec}} $", fontsize=14)
fig.tight_layout()
Z, X, Y, mg = ax.hist2d(z_phot, z_spec, bins=40, range=[rng, rng], cmap=inferno)
X = 0.5*(X[:-1] + X[1:])
Y = 0.5*(Y[:-1] + Y[1:])
ax.contour(X, Y, Z, levels=[68, 95], colors="white", alpha=0.7)
ax.plot(np.arange(2), c="darkred", ls="--", lw=1)
cb = fig.colorbar(mg, ax=ax)
cb.set_label("number of galaxies", fontsize=14)


#####
nbins = 40
fig, ax = plt.subplots(figsize=(9, 7))
ax.set_xlabel(r"$ z_{\rm{phot}} $", fontsize=14)
ax.set_ylabel(r"$ z_{\rm{spec}} $", fontsize=14)
ax.set_xlim(rng[0], rng[1])
ax.set_ylim(rng[0], rng[1])

Z, X, Y, mg = ax.hist2d(z_phot, z_spec, bins=40, range=[rng, rng], cmap=inferno)
X = 0.5*(X[:-1] + X[1:])
Y = 0.5*(Y[:-1] + Y[1:])
ax.hexbin(z_phot, z_spec, gridsize=nbins, extent=rng*2, cmap=inferno)
ax.plot(np.arange(2), c="darkred", ls="--", lw=1)
ax.contour(X, Y, Z, levels=[68], colors="white", alpha=0.7)
cb = fig.colorbar(mg, ax=ax)
cb.set_label("number of galaxies", fontsize=14)
ax.set_aspect(1)
fig.tight_layout()
fig.savefig("z_phot_z_spec_comparison.pdf", bbox_inches="tight")
