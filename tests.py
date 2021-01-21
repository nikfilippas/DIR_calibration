from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from DIR import DIR

fname ="2MPZ_FULL_wspec_coma_complete.fits"
cat = fits.open(fname)[1].data
xcat = cat[np.where(cat["ZSPEC"] != -999.)[0]]
cols = ["JCORR", "HCORR", "KCORR", "W1MCORR", "W2MCORR", "BCALCORR", "RCALCORR", "ICALCORR"]

d = DIR(fname, cols)
d.get_nz(save=True)
weights = d.weights
bins = np.arange(0, 1, 0.001)  # redshift bin edges
pz_dir, _ = np.histogram(xcat["ZSPEC"], bins=bins, density=True, weights=weights)
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
plt.plot(z_mid, pz_dir, "green", lw=2, label="DIR")


ax.legend(loc="upper right", fontsize=12)
ax.grid(ls=":")
ax.set_xlim(0.0, 0.3)
ax.set_xlabel("z", fontsize=16)
ax.set_ylabel("N(z)", fontsize=16)
fig.savefig("2mpz_comparison.pdf", bbox_inches="tight")
