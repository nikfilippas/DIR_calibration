"""
Cross-reference the WIxSC photo-z's and spec-z's, and save them.
"""
import pandas as pd
import numpy as np

fname_data_phot = "data/wiseScosPhotoz160708.csv"
fname_data_spec = "data/zSpec-comp-WIxSC.csv"

print("Loading photo-z & spec-z catalogues...")
f_ph = pd.read_csv(fname_data_phot)
f_sp = pd.read_csv(fname_data_spec)

print("Cross-referencing catalogues...")
f_x = pd.merge(f_ph, f_sp,
               left_on=["ra", "dec"],
               right_on=["ra_WISE", "dec_WISE"],
               how="left",
               validate="1:1").fillna(-999)

print("Writing full catalogue to file...")
f_x.to_csv(fname_data_phot.split(".")[0] + "_xref.csv")

print("Splitting the z-bins...")
zbins = [(0.10, 0.15),
         (0.15, 0.20),
         (0.20, 0.25),
         (0.25, 0.30),
         (0.30, 0.35)]

for i, zbin in enumerate(zbins):
    fname = fname_data_phot.split(".")[0] + "_bin%d.csv" % (i+1)
    z = f_x.zPhoto_Corr
    idx = np.where((z > zbin[0]) & (z < zbin[1]))[0]
    f_x.iloc[idx].to_csv(fname)
    print("  wisc%d" % (i+1))

print("Output saved in %s/." % fname_data_phot.split("/")[0])
