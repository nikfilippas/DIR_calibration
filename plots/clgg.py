"""
Compare clgg of the g-maps produced with new N(z)
with the ones used in 1909.09102.
"""
import os
os.chdir("..")
import healpy as hp
import numpy as np
import matplotlib.pyplot as plt

#hp.disable_warnings()

mask_fname = "data/maps/mask_v3.fits"
tmpz_old = "data/maps/2mpz_05_01_512.fits"
wisc1_old = "data/maps/2dstarsub_WISC_cleaned_public.bin_0.1_z_0.15.Pix512.fits"
wisc2_old = "data/maps/2dstarsub_WISC_cleaned_public.bin_0.15_z_0.2.Pix512.fits"
wisc3_old = "data/maps/2dstarsub_WISC_cleaned_public.bin_0.2_z_0.25.Pix512.fits"
wisc4_old = "data/maps/2dstarsub_WISC_cleaned_public.bin_0.25_z_0.3.Pix512.fits"
wisc5_old = "data/maps/2dstarsub_WISC_cleaned_public.bin_0.3_z_0.35.Pix512.fits"
tmpz_new = "data/maps/map_2mpz.fits"
tmpz_new_full = "data/maps/map_2mpz_full.fits"
wisc1_new = "data/maps/map_wisc1.fits"
wisc2_new = "data/maps/map_wisc2.fits"
wisc3_new = "data/maps/map_wisc3.fits"
wisc4_new = "data/maps/map_wisc4.fits"
wisc5_new = "data/maps/map_wisc5.fits"


old = [tmpz_old, wisc1_old, wisc2_old, wisc3_old, wisc4_old, wisc5_old]
new = [tmpz_new, wisc1_new, wisc2_new, wisc3_new, wisc4_new, wisc5_new]

mask = hp.read_map(mask_fname)
nside = hp.npix2nside(mask.size)

fig, ax = plt.subplots(2, 3, figsize=(17, 9))
[a.set_ylabel(r"$C_{\ell}$", fontsize=16) for a in ax[:, 0]]
[a.set_xlabel(r"$\ell$", fontsize=16) for a in ax[1]]
ax = ax.flatten()
fig.tight_layout()

for i, (a, o, n) in enumerate(zip(ax, old, new)):
    print("z-bin %d" % i)
    map_o = hp.ud_grade(hp.read_map(o), nside)*mask
    map_n = hp.ud_grade(hp.read_map(n), nside)*mask

    cl_old = hp.anafast(map_o)
    cl_new = hp.anafast(map_n)
    l = np.arange(1, cl_old.size+1)

    a.loglog(l, cl_old, "r.", label="old map")
    a.loglog(l, cl_new, "g.", label="new map")
    if i == 0:
        cl = hp.anafast(hp.ud_grade(hp.read_map(tmpz_new_full), nside)*mask)
        a.loglog(l, cl, "y.", alpha=0.4, label="new map z no photo-z cuts")
        a.legend(loc="upper right")
    a.legend(loc="upper right")

fig.savefig("img/clgg.pdf", bbox_inches="tight")
