import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--num-mixture", type=int, required=False, default=3)
parser.add_argument("--random-seed", type=int, required=False, default=1234)
args = parser.parse_args()

import numpy as np
np.set_printoptions(3, suppress=True)
np.random.seed(args.random_seed)

xlim = (-10., 10.)
ylim = (-10., 10.)

X = np.concatenate([
  np.random.multivariate_normal(
    mean=(5.,5.), cov=[[1.,-3.],[-3.,1.]], size=20),
  np.random.multivariate_normal(
    mean=(-1.,2.), cov=[[2.,1.],[1.,1.]], size=20),  
  np.random.multivariate_normal(
    mean=(-5.,-5.), cov=[[3.,0.],[0.,3.]], size=20) 
])

import sklearn.mixture
gm = sklearn.mixture.GaussianMixture(n_components=3, random_state=0)
gm.fit(X)

xs = np.linspace(xlim[0], xlim[1], 60)
ys = np.linspace(ylim[0], ylim[1], 60)
xs, ys = np.meshgrid(xs, ys)
xys = np.concatenate([np.expand_dims(e, -1) for e in [xs, ys]], axis=-1)

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches

fig = plt.figure(figsize=(9.6,4.8))
ax_2d = fig.add_subplot(1, 2, 1)
ax_2d.set_xlim(*xlim)
ax_2d.set_ylim(*ylim)
ax_3d = fig.add_subplot(1, 2, 2, projection='3d')

import scipy.stats
for i, (m, cv) in enumerate(zip(gm.means_, gm.covariances_)):
  ax_2d.scatter(X[:,0], X[:,1])

  def get_ellipse(_m, _cv, scale=20):
    a, b, _, c = _cv.flatten()
    w = (a+c)/2. + np.sqrt(np.square((a-c)/2.)+np.square(b))
    h = (a+c)/2. - np.sqrt(np.square((a-c)/2.)+np.square(b))
    if b == 0:
      angle = 0 if a >= c else np.pi/2.
    else:
      angle = np.arctan2(w-a, b)
    angle *= 180./np.pi
    w = np.sqrt(w*scale)
    h = np.sqrt(h*scale)
    return matplotlib.patches.Ellipse(xy=_m, width=w, height=h, angle=angle, fill=False)

  el = get_ellipse(m, cv)
  ax_2d.add_patch(el)
  ax_2d.scatter(m[0], m[1], color='red') # center

zs = np.exp(gm.score_samples(xys.reshape(-1, 2)))
zs = zs.reshape((60, 60))

colors = matplotlib.cm.jet(plt.Normalize(zs.min(), zs.max())(zs))
rcount, ccount, _ = colors.shape
suf = ax_3d.plot_surface(xs, ys, zs,
  rcount=50, ccount=50, facecolors=colors, shade=False)
suf.set_facecolor((0,0,0,0))
  
plt.savefig('output.png')

