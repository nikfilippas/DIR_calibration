"""
Useful functions used in weighted direct calibration.
Likelihood code in this script adapted from @damonge.
"""
import numpy as np
from scipy.interpolate import interp1d
from scipy.integrate import simps
from scipy.signal import savgol_filter
from sympy import divisors


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


def nearest_divisor(num, total, mode="nearest"):
    """
    Finds number nearest to ``num`` which is a divisor of ``total``.

    Parameters
    ----------
    num : int
        Target number.
    total : int
        Length of array to be split into equally sized parts.
    mode : str
        How the number will move {'nearest', 'high', 'low'}.

    Returns
    -------
    int
        Divisor of ``total`` nearest to ``num``.

    Examples
    --------
    It works when the nearest divisor is smaller:

    >>> nearest_divisor(3, 10)
    2

    It works when the nearest divisor is larger:

    >>> nearest_divisor(4, 15)
    5

    If there are two equidistant nearest divisors
    it returns the lowest of the two:

    >>> nearest_divisor(2, 9) == nearest_divisor(2, 9, "low") == 1
    True

    But note:

    >>> nearest_divisor(2, 9, "higher")
    3
    """
    if total % num == 0:  # number is already a divisor
        return num
    div = np.array(divisors(total))
    if mode == "nearest":
        idx = np.abs(div-num).argmin()
    elif mode == "high":
        idx = np.where((div-num) > 0)[0][0]
    elif mode == "low":
        idx = np.where((div-num) < 0)[0][-1]
    else:
        raise ValueError("mode not recognised")
    return div[idx]


class Likelihood(object):
    """
    Likelihood of the width parameter.

    Smooth the N(z) data using a Savgol filter
    and use the interpolated template to create and
    minimise the likelihood probability.

    Arguments
    ---------
        z : ``numpy.ndarray``
            Redshift array.
        Nz : ``numpy.ndarray``
            DIR probability distribution function.
        dNz : ``numpy.ndarray``
            Error of N(z).
    """
    def __init__(self, z, Nz, dNz, window_size=21):
        # create smooth distribution to serve as template
        Nz_smooth = savgol_filter(Nz, window_size, 2)
        self.Nzi = interp1d(z, Nz_smooth, kind="cubic",
                            bounds_error=False, fill_value=0)

        # cut to range of redshifts we actually care about
        zz = np.linspace(z.min(), z.max(), 1000)
        Nzz = self.Nzi(zz)
        zrange = zz[np.where(Nzz > Nzz.max()/500)[0]].take([0, -1])
        mask = (z >= zrange[0]) & (z <= zrange[1])
        mask = (z < 0.25) & (z > 0.01)
        self.z = z[mask]
        self.Nz = Nz[mask]
        self.dNz = dNz[mask]  # actual errors
        # if the error is zero, make it very large
        # (effectively removing those points from the chi^2)
        self.err = self.dNz
        self.err[self.err <= 0] = 1e16

        self.Nz_smooth = Nz_smooth[mask]
        self.z_mean = np.average(self.z, weights=self.Nz)

    def chi2(self, w):
        """Compute the chi square, given a width ``w``."""
        Nzw = self.Nzi(self.z_mean + (self.z-self.z_mean)/w)
        return np.sum(((self.Nz-Nzw)/self.err)**2)

    def prob(self, prior=[0.98, 1.02]):
        """Calculate the probability."""
        ws = np.linspace(prior[0], prior[1], 1000)
        wprob = np.exp(-0.5*np.array([self.chi2(w) for w in ws]))
        wprob /= np.sum(wprob)

        self.w_mean = np.sum(wprob * ws)
        self.dw = np.sqrt(np.sum(wprob * (ws - self.w_mean)**2))
        return self.w_mean, self.dw
