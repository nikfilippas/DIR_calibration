"""
Compute and save the weights used in the photo-z
calibration of the 2MPZ and WIxSC bins.
"""
from DIR import xref, weights

# global
fname_mask = "data/maps/mask_v3.fits.gz"

## 2MPZ ##
fname_data = "data/cats/2MPZ_FULL_wspec_coma_complete.fits.gz"
colors = ["JCORR", "HCORR", "KCORR", "W1MCORR",
          "W2MCORR", "BCALCORR", "RCALCORR", "ICALCORR"]
print("2MPZ")
# sample selection
q = xref(fname_data)
q.remove_galplane(fname_mask, "L", "B")
q.cutoff("ZPHOTO", [-1.00, 0.10])
cat = q.cat_fid  # photo-z sample
q.cutoff("ZSPEC", -999)
xcat = q.cat_fid  # spec-z sample
print(len(xcat) / len(cat))
# spectroscopic sample weights
weights(xcat, cat, colors, verbose=True, save="out/weights_2mpz")


## WIxSC ##
fname_data = "data/cats/wiseScosPhotoz160708.csv"
colors = ["w1mCorr", "w2mCorr", "bCalCorr", "rCalCorr"]

for i in range(1, 6):
    print("WIxSC bin %d" % i)
    # sample selection
    fname = fname_data.split(".")[0] + "_bin%d.csv" % i
    q = xref(fname)
    q.remove_galplane(fname_mask, "l", "b")
    cat = q.cat_fid  # photo-z sample
    q.cutoff("Zspec", -999)
    xcat = q.cat_fid  # spec-z sample
    print(len(xcat) / len(cat))
    # spectroscopic sample weights
    weights(xcat, cat, colors, verbose=True, save="out/weights_wisc%d" % i)
