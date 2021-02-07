"""
Cross-reference the WIxSC photo-z's and spec-z's, and save them.
"""
import pandas as pd
import numpy as np

fname_data_phot = "data/cats/wiseScosPhotoz160708.csv"
fname_data_spec = "data/cats/zSpec-comp-WIxSC.csv"

print("Loading photo-z & spec-z catalogues...")
f_ph = pd.read_csv(fname_data_phot)
# saving memory
get_rid = ["wiseX", "wiseID", "scosID", "cx", "cy", "cz", "htmID", "ebv", "zPhoto_ANN", "fromAllSky"]
f_ph.drop(inplace=True, columns=get_rid)
f_sp = pd.read_csv(fname_data_spec)
get_rid = ["W1c", "W2c", "Bcc", "Rcc", "w1sigmpro", "w2sigmpro", "errB", "errR", "zCorr"]
f_sp.drop(inplace=True, columns=get_rid)

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
    print("  wisc%d" % (i+1))
    fname = fname_data_phot.split(".")[0] + "_bin%d.csv" % (i+1)
    z = f_x.zPhoto_Corr
    idx = np.where((z > zbin[0]) & (z <= zbin[1]))[0]
    f_x.iloc[idx].to_csv(fname)

print("Output saved in %s/." % fname_data_phot.split("/")[0])
