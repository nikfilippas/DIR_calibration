"""
Testing behaviour of ``numpy.convolve``.
"""
import os
os.chdir("..")
import numpy as np
from scipy.integrate import simps
import matplotlib.pyplot as plt

# define an x-array
x = np.linspace(0, 1, 1000)

# define and create a gaussian
# centred at the mean of the x-array  <-- important
# so that convolution is symmetric
# --------------------------------
# no need to normalise it
gauss= lambda x, m, s: 1/(s*np.sqrt(2*np.pi)) * np.exp(-0.5*((x-m)/s)**2)
m = x.mean()
s = 0.014
gs = gauss(x, m, s)

# create a very narrow top-hat (almost delta)
th = np.zeros_like(x)
# no need to normalise it so for better viewing
# we set the top to align with the other functions
th[(x >= 0.20) & (x <= 0.21)] = gs.max()


# convolve
# normalise, because after all it's a pdf
cv = np.convolve(th, gs, "same")
cv /= simps(cv, x=x)

fig, ax = plt.subplots()
ax.set_title("convolution example")
ax.plot(x, th, "k")
ax.plot(x, gs, "b")
ax.plot(x, cv, "r")
ax.set_ylim(0, 30)
# fig.savefig("img/convolution_example.pdf", bbox_inches="tight")
