import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import lognorm

gmax=.000333
gmin=.000000333

w_max = 0.3

w = 0.1
gp = w/w_max * (gmax - gmin) + gmin
gn = gmin

fig = plt.figure()

ax1 = fig.add_subplot(4, 1, 1)
ax2 = fig.add_subplot(4, 1, 2)
ax3 = fig.add_subplot(4, 1, 3)
ax4 = fig.add_subplot(4, 1, 4)
# Calculate the first four moments:

s = 0.8
mean, var, skew, kurt = lognorm.stats(s, moments='mvsk')
# Display the probability density function (pdf):

# x = np.linspace(lognorm.ppf(0.01, s),
#                 lognorm.ppf(0.99, s), 100)
# ax1.plot(x, lognorm.pdf(x, s),
#         'r-', lw=5, alpha=0.6, label='lognorm pdf')
# rv = lognorm(s)
# ax1.plot(x, rv.pdf(x), 'k-', lw=2, label='frozen pdf')
#
# vals = lognorm.ppf([0.001, 0.5, 0.999], s)
# np.allclose([0.001, 0.5, 0.999], lognorm.cdf(vals, s))

rp = lognorm.rvs(s, size=1000)
ax1.hist(gp*rp, bins=999, density=True, histtype='stepfilled', alpha=0.2)
# ax1.legend(loc='best', frameon=False)
ax1.set_title("{}".format(str(np.var(rp*gp))))

rn = lognorm.rvs(s, size=1000)
ax2.hist(gn*rn, bins=999, density=True, histtype='stepfilled', alpha=0.2)
ax2.set_title("{}".format(str(np.var(rn*gn))))

g = gp * rp - gn * rn
ax3.hist(g, bins=999, density=True, histtype='stepfilled', alpha=0.2)
ax3.set_title("{}".format(np.var(g)))

w_t = g * w_max / (gmax - gmin)
ax4.hist(w_t, bins=999, density=True, histtype='stepfilled', alpha=0.2)
ax4.set_title("{}".format(np.var(w_t)))

plt.tight_layout()
plt.savefig("crossbar weight distribution.pdf")
plt.show()

