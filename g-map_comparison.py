"""
Compare g-maps produced with new N(z) with the ones used in 1909.09102.
"""
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
wisc1_new = "data/maps/map_wisc1.fits"
wisc2_new = "data/maps/map_wisc2.fits"
wisc3_new = "data/maps/map_wisc3.fits"
wisc4_new = "data/maps/map_wisc4.fits"
wisc5_new = "data/maps/map_wisc5.fits"


old = [tmpz_old, wisc1_old, wisc2_old, wisc3_old, wisc4_old, wisc5_old]
new = [tmpz_new, wisc1_new, wisc2_new, wisc3_new, wisc4_new, wisc5_new]


for i, (o, n) in enumerate(zip(old, new)):
    cl_old = hp.anafast(hp.read_map(o))
    cl_new = hp.anafast(hp.read_map(n))
    l = np.arange(1, cl_old.size+1)

    plt.loglog(l, cl_old, "r.", label="old")
    plt.loglog(l, cl_new, "g.", label="new")
    plt.legend(loc="upper right")
    plt.savefig("tests/anafast_%d.pdf" % i)
    plt.close("all")
