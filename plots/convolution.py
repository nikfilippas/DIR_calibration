import os
os.chdir("..")
import numpy as np
from scipy.integrate import simps
import matplotlib.pyplot as plt
from DIR import DIR_cross_match

# sample selection
fname_data = "2MPZ_FULL_wspec_coma_complete.fits"
fname_mask = "mask_v3.fits"
q = DIR_cross_match(fname_data)  # size: 928352
q.remove_galplane(fname_mask, "SUPRA", "SUPDEC")  # size: 716055
q.cutoff("ZPHOTO", [0.05, 0.10])  # size: 360164
q.cutoff("ZSPEC", -999)  # size: 141552
xcat = q.cat_fid

# DIR sample
f = np.load("out/DIR.npz")
z_mid, Nz = f["z_arr"], f["nz_arr"]
step = np.round(z_mid[1] - z_mid[0], 4)
bins = np.arange(0, 1+step, step)

# photo sample
pz_phot, _ = np.histogram(xcat["ZPHOTO"], bins=bins, density=True)

# mask uninteresting regions
mask = (z_mid > 0.01) & (z_mid < 0.25)
z_mid = z_mid[mask]
Nz = Nz[mask]
pz_phot = pz_phot[mask]

# gaussian
gauss = lambda x, x0, s: 1/(s*2*np.pi) * np.exp(-0.5*((x-x0)/s)**2)
z_eff = np.append(np.append(z_mid[::-1], [0]), z_mid)
Nz_gauss = gauss(z_eff, 0, 0.014)

# convolution
Nz_conv = np.convolve(Nz_gauss, pz_phot, "full")
Nz_conv /= simps(Nz_conv, dx=step)
Nz_conv = Nz_conv[len(z_mid):]

# plot
fig, ax = plt.subplots()
ax.set_xlim(0, 0.25)
ax.set_xlabel("z", fontsize=16)
ax.set_ylabel("N(z)", fontsize=16)
ax.grid(ls=":")
fig.tight_layout()

ax.plot(z_mid, pz_phot, "navy", lw=1, label="photometric")
ax.plot(z_eff, Nz_gauss, "orange", lw=1, label="gauss kernel")
ax.plot(z_mid, Nz_conv[:len(z_mid)], "r", lw=2, label="photo conv. gauss")
ax.plot(z_mid, Nz, "k", lw=2, ls="--", label="DIR")
ax.legend(loc="upper right")
ax.savefig("img/convolution.pdf", bbox_inches="tight")
