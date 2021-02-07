# DIR Calibration

This repository uses tools to perform weighted direct calibration (DIR) to the 2MPZ galaxy catalogue and the WISExSuperCOSMOS (WIxSC) catalogue, which is split in five redshift ranges.

## Running the pipeline
1. To run this pipeline you must have catalogues containing galaxy IDs (or coordinates), and spectroscopic and photometric redshifts for 2MPZ and WIxSC.
2. `wisc_xref` performs cross-referencing of the entire WIxSC catalogue, and splits it into 5 catalogues containing the galaxies of each redshift bin (to save loading time).
3. `weights.py` used the spectroscopic redshifts as the training set to perform DIR to the entire sample of galaxies in each z-bin.
4. `nz.py` uses the saved weights to calculate and save the DIR-calibrated redshift distributions as `numpy.array`s in `out/DIR_*.npz`. It also performs jackknives.
5. `width.py` uses the saved N(z)'s to calculate priors (mean and standard deviation) on the `width (w)` parameter used in `nikfilippas/yxg` and `nikfilippas/yxgxk` to modify the redshift distributions.

## Other scripts
6. `mk_dndz` uses the saved weights to calculate and save *smoothed* DIR-calibrated redshifts (as opposed to `nz.py` which calculates raw distributions) in `data/dndz/*DIR.txt`; ready to be used in modelling of N(z)'s.
7. `mk_maps.py` uses the catalogues to create maps of galaxy overdensity (`\Delta_g`) in each redshift bin.
8. `DIR.py` contains the functions to perform DIR calibration.
9. `funcs.py` contains other useful functions used in the pipeline.

## Credit
If you find this pipeline useful, please include a link to this repo in your work.
