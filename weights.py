"""
Compute and save the weights used in the photo-z
calibration of the 2MPZ and WIxSC bins.
"""
from DIR import DIR_cross_match, DIR_weights

## 2MPZ ##
fname_data = "data/2MPZ_FULL_wspec_coma_complete.fits"
fname_mask = "data/mask_v3.fits"
colors = ["JCORR", "HCORR", "KCORR", "W1MCORR",
          "W2MCORR", "BCALCORR", "RCALCORR", "ICALCORR"]
# sample selection
q = DIR_cross_match(fname_data)
q.remove_galplane(fname_mask, "L", "B")
q.cutoff("ZPHOTO", [-1.00, 0.10])
cat = q.cat_fid  # photo-z sample
q.cutoff("ZSPEC", -999)
xcat = q.cat_fid  # spec-z sample
# spectroscopic sample weights
weights = DIR_weights(xcat, cat, colors, verbose=True,
                         save="out/weights_2mpz")


## WIxSC ##
fname_data = "data/wiseScosPhotoz160708.csv"
fname_mask = "data/mask_v3.fits"
colors = ["W1c", "W2c", "Bcc", "Rcc"]

for i in range(1, 6):
    print("WIxSC bin %d" % i)
    # sample selection
    fname = fname_data.split(".")[0] + "_bin%d.csv" % i
    q = DIR_cross_match(fname)
    q.remove_galplane(fname_mask, "l", "b")
    cat = q.cat_fid  # photo-z sample
    q.cutoff("Zspec", -999)
    xcat = q.cat_fid  # spec-z sample
    # spectroscopic sample weights
    print("Finding weights...")
    weights = DIR_weights(xcat, cat, colors, verbose=True,
                          save="out/weights_wisc%d" % i)
