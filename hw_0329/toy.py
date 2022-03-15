import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--num-mixture", type=int, required=True)
parser.add_argument("--random-seed", type=int, required=False, default=1234)
args = parser.parse_args()

import numpy as np
np.set_printoptions(3, suppress=True)
np.random.seed(args.random_seed)

X = np.random.normal(size=(20, 2)) # TODO

import sklearn.mixture
gm = sklearn.mixture.GaussianMixture(n_components=3, random_state=0)
gm.fit(X)

xs = np.linspace(-3, 3, 60)
ys = np.linspace(-3, 4, 60)
xs, ys = np.meshgrid(xs, ys)
xys = np.concatenate([np.expand_dims(e, -1) for e in [xs, ys]], axis=-1)

import scipy.stats
for i, (m, cv) in enumerate(zip(gm.means_, gm.covariances_)):
  rv = scipy.stats.multivariate_normal(mean=m, cov=cv)
  zs = rv.pdf(xys)

  import matplotlib
  import matplotlib.pyplot as plt
  fig = plt.figure(figsize=(9.6,4.8))
  ax_2d = fig.add_subplot(1, 2, 1) 
  ax_3d = fig.add_subplot(1, 2, 2, projection='3d')

  ax_2d.scatter(X[:,0], X[:,1])
  #colors = matplotlib.cm.viridis(plt.Normalize(zs.min(), zs.max()))
  #rcount, ccount, _ = colors.shape
  suf = ax_3d.plot_surface(xs, ys, zs,
    cmap='jet', shade=False, edgecolor='black', linewidth=1)
    #rcount=rcount, ccout=ccout, facecolors=colors, shade=False)
  suf.set_facecolor((0,0,0,0))
    #rstride=1, cstride=1, shade=False, cmap='jet', linewidth=1,
    #edgecolors='#000000')
  plt.savefig('output2_{}.png'.format(i))

