from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits

cat = fits.open("2MPZ_FULL_wspec_coma_complete.fits")[1].data
xcat = cat[np.where(cat["ZSPEC"] != -999.)[0]]
cols = ["JCORR", "HCORR", "KCORR", "W1MCORR",
        "W2MCORR", "BCALCORR", "RCALCORR", "ICALCORR"]

bins = np.arange(0, 1, 0.005)  # redshift bin edges
pz_spec, _ = np.histogram(xcat["ZSPEC"], bins=bins, density=True)
pz_phot, _ = np.histogram(xcat["ZPHOTO"], bins=bins, density=True)


## plots
fig, ax = plt.subplots(1, 1, figsize=(10, 6))
fig.tight_layout()

# compare to Maciek's N(z)
m1 = np.loadtxt("/home/nick/Desktop/yxgxk/data/dndz/2MPZ_bin1.txt")
m2 = np.loadtxt("/home/nick/Desktop/yxgxk/data/dndz/2MPZ_v2_bin1.txt")
ax.plot(*m1.T, ls="--", label="2mpz_v1")
ax.plot(*m2.T, ls="-.", label="2mpz_v2")

# jackknives
for jk_id in tqdm(range(1000)):
    f = np.load("out/jk%s.npz" % jk_id)
    nz, z_jk = f["nz_arr"], f["z_arr"]
    ax.plot(z_jk, nz, "gray", alpha=0.005)

z_mid = 0.5*(bins[:-1] + bins[1:])
plt.plot(z_mid, pz_phot, "navy", lw=2, label="photometric")
plt.plot(z_mid, pz_spec, "orange", lw=2, label="spectroscopic")
f_DIR = np.load("out/DIR.npz")
plt.plot(f_DIR["z_arr"], f_DIR["nz_arr"], "green", lw=2, label="DIR")

ax.legend(loc="upper right", fontsize=12)
ax.grid(ls=":")
ax.set_xlim(0.0, 0.3)
ax.set_xlabel("z", fontsize=16)
ax.set_ylabel("N(z)", fontsize=16)
fig.savefig("2mpz_comparison.pdf", bbox_inches="tight")


# galaxy positions
import healpy as hp
from matplotlib.cm import gray, bwr_r
z_spec, z_phot = xcat["ZSPEC"], xcat["ZPHOTO"]
ra, dec = xcat["SUPRA"], xcat["SUPDEC"]
dz = z_spec - z_phot
eq2gal = (+96.3, -60.1, 0)  # galactic to equatorial rotation
thin = 50
data = hp.read_map("/home/nick/Desktop/yxgxk/data/maps/HFI_SkyMap_545_2048_R2.02_full.fits", dtype=float)
plt.figure(num="mollview", figsize=(12,7))
hp.mollview(data, fig="mollview", title="", notext=True, xsize=2000,
            norm="log", cmap=gray, cbar=False,
            rot=eq2gal, coord="G")
alphas = z_spec[::thin] / z_spec.mean()
transp = dz[::thin]/(2*dz.std()) + 2*dz.std()
colors = bwr_r(transp)
# colors[:,-1] = alphas
hp.projscatter(ra[::thin], dec[::thin], coord="C",
               color=colors, s=2)
hp.graticule(coord="C", color="y", ls="")
plt.savefig("2MPZ_zspec_positions.pdf", bbox_inches="tight")
