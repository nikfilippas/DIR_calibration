"""
Useful functions used in weighted direct calibration.
"""
import numpy as np
from scipy.interpolate import interp1d
from scipy.integrate import simps
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
