"""
Useful functions used in weighted direct calibration.
Likelihood code in this script adapted from @damonge.
"""
import numpy as np
from scipy.interpolate import interp1d
from scipy.integrate import simps
from scipy.optimize import root_scalar
from scipy.signal import savgol_filter
from sympy import divisors


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
    def __init__(self, z, Nz, dNz=None):
        self.z = z
        self.Nz = Nz
        self.dNz = dNz
        self.smooth()
        self.Nzi = interp1d(self.z, self.Nz_smooth, kind="cubic",
                            bounds_error=False, fill_value=0)
        self.z_mean = np.average(self.z, weights=self.Nz)

    def smooth(self, q_int=[0.1, 99.5]):
        """
        Smooth filtering with optimal window size.

        Start from a window size of 5 and check if
        function is monotonically increasing up to <z>
        and then monotonically decreasing.

        Arguments
        ---------
            q_int : Sequence of 2 floats
                Mask off region outside percentiles of the distribution.
                (Used when the distribution has very long tails.)
        """
        # cutoff uninteresting region of redshift distribution
        q = percentile(self.z, q_int, weights=self.Nz)
        idx = np.where((self.z >= q[0]) & (self.z <= q[1]))[0]
        self.z = self.z[idx]
        self.Nz = self.Nz[idx]
        if self.dNz is not None:
            self.dNz = self.dNz[idx]
        # determine window size
        window = int(15*np.log10(self.z.size) // 2 * 2 + 1)
        self.Nz_smooth = savgol_filter(self.Nz, window, polyorder=3)
        self.norm = simps(self.Nz_smooth, self.z)

    def chi2(self, w):
        """Compute the chi square, given a width ``w``."""
        # if the error is zero, make it very large
        # (effectively removing those points from the chi^2)
        err = self.dNz.copy()
        err[err <= 0] = 1e16
        Nzw = self.Nzi(self.z_mean + (self.z-self.z_mean)/w)
        Nzw *= self.norm/simps(Nzw, self.z)
        return np.sum(((self.Nz_smooth-Nzw)/err)**2)

    def prob(self, prior=[0.98, 1.02]):
        """Calculate the probability."""
        ws = np.linspace(prior[0], prior[1], 1000)
        wprob = np.exp(-0.5*np.array([self.chi2(w) for w in ws]))
        wprob /= np.sum(wprob)
        self.ws, self.wprob = ws, wprob
        self.w_mean = np.sum(wprob * ws)
        self.dw = np.sqrt(np.sum(wprob * (ws - self.w_mean)**2))
        return self.w_mean, self.dw


def percentile(a, q, weights=None):
    """Wrapper of ``numpy.percentile`` that handles weights."""
    if weights is None:
        return np.percentile(a, q)
    q = np.atleast_1d(q)/100
    sums = [simps(weights[:i], a[:i]) for i in range(1, len(a))]
    sums = np.append([0], sums)
    interp = [interp1d(a, sums-qi) for qi in q]
    roots = [root_scalar(I, bracket=[a.min(), a.max()]).root for I in interp]
    if len(q) == 1:
        return roots[0]
    else:
        return np.array(roots)



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

    Given the cross-mathced catalogue, we almost cetrainly cannot split
    it into ``jk["num"]`` equally-sized bins, so we would have to assign
    weights to each jackknife. We overcome this by finding the divisor
    of the size of the catalogue, which is closest to ``jk["num"]``.

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

    Example usecase:
    >>> N_jk = 100
    >>> N_jk = nearest_divisor(N_jk, len(xcat))  # effective number of JKs
    >>> print("Jackknife size:\t%d" % (len(xcat)/N_jk))
    >>> print("# of jackknives:\t×%d" % N_jk)
    >>> print("Catalogue size:\t=%d" % len(xcat))
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
