"""
Likelihood code adapted from @damonge.
"""
import numpy as np
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt

# Read data
d = np.load("data/test1.npz")
z, Nz, errNz = d['z'], d['nz'], d['err']

# Create smoothed distribution to serve as template
Nz_smooth = savgol_filter(Nz, 21, 2)
Nzi = interp1d(z, Nz_smooth, fill_value=0, bounds_error=False, kind='cubic')

# Cut to range of redshifts we actually care about
mask = (z < 0.25) & (z > 0.01)
z = z[mask]
Nz = Nz[mask]
errNz = errNz[mask]
Nz_smooth = Nz_smooth[mask]

plt.figure()
plt.plot(z, Nz_smooth, 'r-')
plt.errorbar(z, Nz, yerr=errNz, fmt='k.')
plt.xlabel('$z$')
plt.ylabel('$N(z)$')


# Compute mean redshift
z_mean = np.sum(z*Nz)/np.sum(Nz)

# If the error is zero, make it very large
# (effectively removing those points from the chi^2)
errNz[errNz<=0] = 1E10


def chi2(w):
    Nzw = Nzi(z_mean + (z-z_mean)/w)
    return np.sum(((Nz-Nzw)/errNz)**2)


# Compute w likelihood
ws = np.linspace(0.98, 1.02, 500)
wprob = np.exp(-0.5*np.array([chi2(w) for w in ws]))
wprob /= np.sum(wprob)

# Compute mean and error from pdf
w_mean = np.sum(ws*wprob)
w_error = np.sqrt(np.sum((ws-w_mean)**2*wprob))

plt.figure()
plt.plot(ws, wprob)
plt.xlim([w_mean-5*w_error,w_mean+5*w_error])
plt.xlabel('$w_z$')
plt.ylabel('$p(w_z)$')
print(f"w = {w_mean} +- {w_error}")
