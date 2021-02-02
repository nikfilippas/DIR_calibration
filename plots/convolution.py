"""
Show that the convolution of the photo-z sample with a
Gaussian and a Lorentzian roughly reproduces the redshift
distributions in 1805.11525.
"""
import os
os.chdir("..")
import numpy as np
from scipy.integrate import simps
import matplotlib.pyplot as plt
from DIR import DIR_cross_match
from funcs import Likelihood

# sample selection
fname_data = "data/2MPZ_FULL_wspec_coma_complete.fits"
fname_mask = "data/mask_v3.fits"
q = DIR_cross_match(fname_data)
q.remove_galplane(fname_mask, "L", "B")
q.cutoff("ZPHOTO", [-1.00, 0.10])
q.cutoff("ZSPEC", -999)
xcat = q.cat_fid

# DIR sample
f = np.load("out/DIR.npz")
z_mid, Nz = f["z_arr"], f["nz_arr"]
step = np.round(z_mid[1] - z_mid[0], 4)
bins = np.arange(0, 1+step, step)

# smooth DIR sample
Nzi = Likelihood(z_mid, Nz).Nzi

# photo sample
pz_phot, _ = np.histogram(xcat["ZPHOTO"], bins=bins, density=True)

# mask uninteresting regions
mask = (z_mid > 0.01) & (z_mid < 0.25)
z_mid = z_mid[mask]
Nz = Nz[mask]
pz_phot = pz_phot[mask]

# gaussian
gauss = lambda x, x0, s: 1/(s*np.sqrt(2*np.pi)) * np.exp(-0.5*((x-x0)/s)**2)
mu, sigma = z_mid.mean(), 0.014
Nz_gauss = gauss(z_mid, mu, sigma)

# lorentzian
lorentz = lambda x, a, s: (1 + x**2/(2*a*s**2))**(-a)
a, s = 2.93, 0.012
Nz_lor = lorentz(z_mid-mu, a, s)
Nz_lor /= simps(Nz_lor, x=z_mid)

## convolution
Nz_conv_g = np.convolve(pz_phot, Nz_gauss, "same")
Nz_conv_g /= simps(Nz_conv_g, x=z_mid)

Nz_conv_l = np.convolve(pz_phot, Nz_lor, "same")
Nz_conv_l /= simps(Nz_conv_l, x=z_mid)

# normalise convolving funcs
Nz_gauss /= Nz_gauss.max()/Nz.max()
Nz_lor /= Nz_lor.max()/Nz.max()


# compare to Maciek's N(z)
m1 = np.loadtxt("data/2MPZ_bin1.txt")
m2 = np.loadtxt("data/2MPZ_v2_bin1.txt")

# plot
fig, ax = plt.subplots(figsize=(10, 6))
ax.set_xlim(0, 0.25)
ax.set_xlabel("z", fontsize=16)
ax.set_ylabel("N(z)", fontsize=16)
ax.grid(ls=":")
fig.tight_layout()

ax.plot(*m1.T, ls="--", label="2mpz_v1")
ax.plot(*m2.T, ls="-.", label="2mpz_v2")

ax.plot(z_mid, pz_phot, "navy", lw=2, label="photometric")
ax.plot(z_mid, Nz, "k", lw=2, ls="-", label="DIR")
ax.plot(z_mid, Nzi(z_mid), lw=2, ls=":", label="DIR smoothed")

ax.plot(z_mid, Nz_gauss, "r--", lw=1, label="gauss kernel")
ax.plot(z_mid, Nz_lor, "g--", lw=1, label="lorentz kernel")

ax.plot(z_mid, Nz_conv_g, "r", lw=2, label="photo $\\bigotimes$ gauss")
ax.plot(z_mid, Nz_conv_l, "g", lw=2, label="photo $\\bigotimes$ lorentz")

ax.legend(loc="upper right", ncol=2)
fig.savefig("img/convolution.pdf", bbox_inches="tight")
