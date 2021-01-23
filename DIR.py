import warnings
import numpy as np
from scipy.spatial import cKDTree
from sklearn.neighbors import NearestNeighbors
from scipy.interpolate import interp1d
from scipy.integrate import simps


def DIR_weights(xcat, cat, cols, N_nearest=20, tree_leaf=30, save=None):
    """
    DIR calibration

    Performs weighted direct calibration (DIR) to a sample of photometric
    redshifts using a subsample of spectroscopic redshifts. Calibration is
    performed using nearest neighbours in colour space.

    Arguments
    ---------
        xcat : ``numpy.record``
            The cross-matched catalogue (training galaxy sample).
        cat : ``numpy.record``
            The full catalogue (photo-z sample).
            colour space hyper-rectangle.
        N_nearest : int
            Number of nearest neighbours to query.
        tree_leaf : int
            Number of elements in a leaf before the trees use brute force.
        save : str
            Save the output in an `.npz` file using the specified filename.
            Default: ``None``, i.e. no save.

    Returns
    -------
        weights : ``numpy.ndarray``
            Weights of the calibration galaxies in the catalogue.
        idx : ``numpy.ndarray``
            Indices of the calibration galaxies in the catalogue.
    """
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


def nz_from_weights(xcat, weights, indices=None, bins="auto",
                    save=None, full_output=False):
    """
    Calculate redshift probability density function.

    Arguments
    ---------
        xcat : ``numpy.record``
            Catalogue of the cross-matched calibration galaxies.
        weights : ``numpy.ndarray``
            Array containing the weights of the calibration galaxies.
        indices : `array_like`
            Indices of ``xcat`` and ``weights`` sub-sample.
            Default: ``None``; use the entire sample.
        bins : `array_like`
            Redshift bins to sample probability distribution function.
            Default: ``auto`` for automatic creation of the bins.
        save : str
            Save the probability density function in an `.npz` file.
        full_output : bool
            Whether to return calculated redshift distributions.

    Returns
    -------
        Nz : ``numpy.ndarray``
            The redshift probability density function.
        z_mid : ``numpy.ndarray``
            Midpoints of the redshift bins.
    """
    # input handling
    if (save is None) and (full_output is False):
        raise ValueError("Either `save` or `full_output` must specify behaviour.")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if bins == "auto":  # FutureWarning
            bins = np.arange(0, 1, step=0.001)
    z_spec = xcat["ZSPEC"]

    # sub-sample according to indices
    if indices is not None:
        z_spec = z_spec[indices]
        weights = weights[indices]

    Nz, _ = np.histogram(z_spec, bins=bins, density=True, weights=weights)

    # output handling
    z_mid = 0.5*(bins[:-1] + bins[1:])
    if save is not None:
        np.savez(save, z_arr=z_mid, nz_arr=Nz)
    if full_output:
        return Nz, z_mid


def width_func(z, Nz, width=1, z_avg=None, normed=True):
    """
    Stretch the redshift distribution using the width parameter.

    .. math::
        p(z) \\propto p_{\\rm{fid}}\\left( \\langle z \\rangle +
                                \\frac{z - \\langle z \\rangle}{w} \\right),

    where :math:`p_{\\rm{fid}}` is the input distribution,
    :math:`\\langle z \\rangle` is the mean redshift (or anchor point), and
    :math:`w` is the width parameter.

    Arguments
    ---------
        z : ``numpy.ndarray``
            Sampled redshifts of the redshift distribution.
        Nz : ``numpy.ndarray``
            Redshift probability density.
        width : float
            Factor to stretch the distribution by.
        z_avg : float
            Custom redshift anchor point to stretch the distribution.
            Default: ``None``; use the distribution mean value.
        normed : bool
            Re-normalise the modified redhisft distribution.

    Returns
    -------
        Nz_w : ``numpy.ndarray``
            Modified redshift distribution.
    """
    nzf = interp1d(z, Nz, kind="cubic", bounds_error=False, fill_value=0)
    if z_avg is None:
        z_avg = np.average(z, weights=Nz)
    Nz_w = nzf(z_avg + (z-z_avg)/width)
    if normed:
        Nz_w /= simps(Nz_w, x=z)
    return Nz_w
