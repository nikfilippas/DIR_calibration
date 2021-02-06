"""
Plot the following redshift distributions and the associated errors:
    - 2MPZ v1 (Gaussian)
    - 2MPZ v2 (Lorentzian)
    - 2MPZ full photo-z
    - 2MPZ full spec-z
    - 2MPZ spec-z DIR
"""
# move to parent dir
import os, sys
THIS_PATH = os.path.dirname(os.path.realpath(__file__))
NEW_PATH = "/".join(THIS_PATH.split("/")[:-1])
os.chdir(NEW_PATH)
sys.path.append(NEW_PATH)
#########################
import matplotlib.pyplot as plt
import numpy as np
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
fig, (ax, sub) = plt.subplots(2, 1, sharex=True, figsize=(10, 8),
                              gridspec_kw={"height_ratios": [4, 1]})
ax.grid(ls=":")
sub.grid(ls=":")
ax.set_xlim(0.0, 0.2)
ax.set_ylabel("N(z)", fontsize=16)
sub.set_ylabel(r"$|\Delta z_{\rm{jk}}|$", fontsize=16)
sub.set_yscale("log")
sub.axhline(0, ls=":")
sub.set_xlabel("z", fontsize=16)
fig.tight_layout()

# compare to Maciek's N(z)
m1 = np.loadtxt("data/dndz/2MPZ_bin1.txt")
m2 = np.loadtxt("data/dndz/2MPZ_v2_bin1.txt")
ax.plot(*m1.T, ls="--", label="2mpz_v1")
ax.plot(*m2.T, ls="-.", label="2mpz_v2")

f = np.load("out/DIR_2mpz.npz")
z_mid, Nz = f["z_arr"], f["nz_arr"]

# jackknives
diff_sq = 0
jk_id = 0
while True:
    try:
        f = np.load("out/DIR_2mpz_jk%s.npz" % jk_id)
        nz, z_jk = f["nz_arr"], f["z_arr"]
        # ax.plot(z_jk, nz, "gray", alpha=0.07)
        diff = nz-Nz
        sub.plot(z_jk, np.abs(diff), "gray", alpha=0.1)
        diff_sq += diff**2
        jk_id += 1
    except FileNotFoundError:
        break

# catalogues
step = np.round(z_mid[1] - z_mid[0], 4)
bins = np.arange(0, 1+step, step)
pz_spec, _ = np.histogram(xcat["ZSPEC"], bins=bins, density=True)
pz_phot, _ = np.histogram(xcat["ZPHOTO"], bins=bins, density=True)
ax.plot(z_mid, pz_phot, "navy", lw=2, alpha=0.4, label="photometric")
ax.plot(z_mid, pz_spec, "orange", lw=2, label="spectroscopic")
# DIR
err = np.sqrt(jk_id/(jk_id-1) * diff_sq)
ax.errorbar(z_mid, Nz, yerr=err,
            fmt="green", lw=2, label="DIR")

ax.legend(loc="upper right", fontsize=12)
fig.savefig("img/2mpz_comparison.pdf", bbox_inches="tight")
