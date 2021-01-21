import numpy as np
from scipy.spatial import cKDTree
from astropy.io import fits
from sklearn.neighbors import NearestNeighbors


def DIR_weights(fname, cols, N_nearest=20, tree_leaf=30, save=None):
    """
    DIR calibration

    Performs weighted direct calibration (DIR) to a sample of photometric
    redshifts using a subsample of spectroscopic redshifts. Calibration is
    performed using nearest neighbours in colour space.

    Arguments
    ---------
        fname : str
            The target filename of the FITS file containing the catalogue.
        cols : list
            A list of strings giving the names of the colours used in the
            colour space hyper-rectangle.
        N_nearest : int
            Number of nearest neighbours to query.
        tree_leaf : int
            Number of elements in a leaf before the trees use brute force.
        save : str
            Save the output in an `.npz` file using the specified filename.
            Default: `None`, i.e. no save.

    Returns
    -------
        weights : `numpy.ndarray`
            Weights of the calibration galaxies in the catalogue.
        idx : `numpy.ndarray`
            Indices of the calibration galaxies in the catalogue.
    """
    cat = fits.open(fname)[1].data
    xcat = cat[np.where(cat["ZSPEC"] != -999.)[0]]
    photo_sample = np.column_stack([cat[col] for col in cols])
    tree = cKDTree(photo_sample, leafsize=tree_leaf)
    # set-up samples
    train_sample = np.column_stack([xcat[col] for col in cols])
    # neighbours and distances
    NN = NearestNeighbors(n_neighbors=N_nearest, algorithm="kd_tree",
                          leaf_size=tree_leaf, metric="euclidean")
    distances, _ = NN.fit(train_sample).kneighbors(train_sample)
    distances = np.amax(distances, axis=1)
    # tree lookup
    num_photoz = np.array([len(tree.query_ball_point(t, d+1e-6))
                           for t, d in zip(train_sample, distances)])
    weights = len(train_sample)/len(photo_sample) * num_photoz/N_nearest
    idx = xcat["TWOMASSID"]
    # output handling
    if save is not None:
        np.savez(save, indices=idx, weights=weights)
    return weights, idx


def nz_from_weights(xcat, weights, bins="auto", save=None, full_output=False,
          jk={"num":0,"thin":0.01,"jk_id":None,"replace":False}):
    """
    Peform jackknives

    Arguments
    ---------
        xcat : `astropy.io.fits.fitsrec.FITS_rec`
            Catalogue of the cross-matched calibration galaxies.
        weights : `numpy.ndarray`
            Array containing the weights of the calibration galaxies.
        bins : array-like
            Redshift bins to sample probability distribution function.
            Default: "auto" for automatic creation of the bins.
        save : str
            Save the probability density function in an `.npz` file.
        full_output : bool
            Whether to return calculated redshift distributions.
        jk : dict
            Jackknife config:
                num : int
                    Number of jackknives. Defaults: 0 (use entire sample).
                thin : float
                    Thin out redshift training sample by factor,
                    rounded to nearest integer.
                jk_id : int
                    Jackknife id number.
                replace : bool
                    Sample with replacement (bootstrap).


    Returns
    -------
        Nz : `numpy.ndarray`
            The redshift probability density function.
        z_mid : `numpy.ndarray`
            Midpoints of the redshift bins.
    """
    # input handling
    if (save is not None) and (type(jk["jk_id"]) not in [int, float]):
        raise ValueError("Jackknife ID should be a number.")
    if bins == "auto":
        bins = np.arange(0, 1, step=0.001)
    z_spec = xcat["ZSPEC"]
    # jackknife
    if jk["num"] != 0:
        idx = np.random.choice(np.arange(len(xcat)),
                               size=int(np.around(len(xcat)*(1-jk["thin"]))),
                               replace=jk["replace"])
        z_spec, weights = z_spec[idx], weights[idx]
    Nz, z_mid = nz_from_weights(z_spec, weights, bins=bins)
    # output handling
    if save is not None:
        if save[-1] != "/": save += "_"
        if jk["num"] != 0: save += "jk%s" % jk[""]
        np.savez(save, z_arr=z_mid, nz_arr=Nz)
    if full_output:
        return Nz, z_mid
