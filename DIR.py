import warnings
import healpy as hp
from healpy.rotator import Rotator
import pandas as pd
import numpy as np
from astropy.io import fits
from scipy.spatial import cKDTree
from sklearn.neighbors import NearestNeighbors


class DIR_cross_match(object):
    """
    Perform cross-matching operations on a catalogue.

    Arguments
    ---------
        fname_data : ``str``
            Path to data.
    """
    def __init__(self, fname_data):
        self.is_fits = (fname_data.split(".")[-1] == "fits")
        self.is_csv = (fname_data.split(".")[-1] == "csv")
        #hp.disable_warnings()
        if self.is_fits:
            self.cat = fits.open(fname_data)[1].data
        elif self.is_csv:
            self.cat = pd.read_csv(fname_data)
        self.cat_fid = self.cat

    def remove_galplane(self, fname_mask, lon_name, lat_name, mode="G"):
        """
        Given a galactic plane mask, remove the galactic plane.

        Arguments
        ---------
            fname_mask : ``str``
                Path to galactic plane mask.
            lon_name : ``str``
                Name of column containing longitude coordinate.
            lat_name : ``str``
                Name of column containing latitude coordinate.
            mode : ``str``
                Base coordiname system to use {'G', 'C'}. Default: 'C'.
        """
        self.mask = hp.read_map(fname_mask, dtype=float)
        if mode != "G":
            self.mask = Rotator(coord=["G", mode]).rotate_map_alms(self.mask,
                                                    use_pixel_weights=False)
        self.nside = hp.npix2nside(self.mask.size)
        ipix = hp.ang2pix(self.nside,
                          self.cat[lon_name],
                          self.cat[lat_name],
                          lonlat=True)
        self.cat = self.cat[self.mask[ipix] > 0.5]
        self.cat_fid = self.cat

    def cutoff(self, col, vals):
        """
        Create a subsample of the main catalogue.

        Arguments
        ---------
            col : ``str``
                Column name according to which the subsample is created.
            vals : ``float`` or array_like
                If a single value is passed, it cuts off elements with that
                value. If two values are passed, it treats them as boundaries.
        """
        vals = np.atleast_1d(vals)
        # set up indexer if table is Pandas DataFrame
        cat_fid = self.cat_fid.iloc if self.is_csv else self.cat_fid
        if len(vals) == 1:
            self.cat_fid = cat_fid[np.where(self.cat_fid[col] != vals[0])[0]]
        elif len(vals) == 2:
            self.cat_fid = cat_fid[(self.cat_fid[col] >= vals[0]) &
                                   (self.cat_fid[col] <= vals[1])]
        else:
            raise ValueError("Argument `vals` should contain 1 or 2 cutoff values.")


def DIR_weights(xcat, cat, cols, N_nearest=20, tree_leaf=30, save=None, verbose=False):
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
        N_nearest : ``int``
            Number of nearest neighbours to query.
        tree_leaf : ``int``
            Number of elements in a leaf before the trees use brute force.
        save : ``str``
            Save the output in an `.npz` file using the specified filename.
            Default: ``None``, i.e. no save.
        verbose : ``bool``
            Increase verbosity in kd-tree build and query.

    Returns
    -------
        weights : ``numpy.ndarray``
            Weights of the calibration galaxies in the catalogue.
        idx : ``numpy.ndarray``
            Indices of the calibration galaxies in the catalogue.
    """
    photo_sample = np.column_stack([cat[col] for col in cols])
    if verbose: print("Building tree...")
    tree = cKDTree(photo_sample, leafsize=tree_leaf,
                   balanced_tree=False)
    # set-up samples
    train_sample = np.column_stack([xcat[col] for col in cols])
    # neighbours and distances
    if verbose: print("Finding neighbours...")
    NN = NearestNeighbors(n_neighbors=N_nearest, algorithm="kd_tree",
                          leaf_size=tree_leaf, metric="euclidean")
    distances, _ = NN.fit(train_sample).kneighbors(train_sample)
    distances = np.amax(distances, axis=1)
    # tree lookup
    if verbose: print("Tree lookup...")
    num_photoz = np.array([len(tree.query_ball_point(t, d+1e-6))
                           for t, d in zip(train_sample, distances)])
    weights = len(train_sample)/len(photo_sample) * num_photoz/N_nearest
    # output handling
    if save is not None:
        if verbose: print("Saving weights...")
        np.savez(save, weights=weights)
    return weights


def nz_from_weights(xcat, weights, bins="auto", z_col=None,
                    indices=None, save=None, full_output=False):
    """
    Calculate redshift probability density function.

    Arguments
    ---------
        xcat : ``numpy.record``
            Catalogue of the cross-matched calibration galaxies.
        weights : ``numpy.ndarray``
            Array containing the weights of the calibration galaxies.
        bins : `array_like`
            Redshift bins to sample probability distribution function.
            Default: ``auto`` for automatic creation of the bins.
        z_col : ``str``
            Name of column in `xcat` containing spectroscopic redshifts.
        indices : `array_like`
            Indices of ``xcat`` and ``weights`` sub-sample.
            Default: ``None``; use the entire sample.
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
    z_spec = xcat[z_col]

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
