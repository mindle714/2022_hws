import numpy as np
np.set_printoptions(3, suppress=True)

import scipy.fftpack
import matplotlib.pyplot as plt

N = 8
a = np.random.uniform(size=N)
  
W = np.zeros([N, N], dtype='float')
for n in range(N):
  for k in range(N):
    W[n][k] = 2 * np.cos(np.pi/N*(n+.5)*k)
	
print(scipy.fftpack.dct(a, type=2))
e = np.matmul(a, W)
print(e)

fig = plt.figure()

ax = fig.add_subplot(1, 1, 1)
ax.set_axis_off()
ax.imshow(np.matmul(W[0, np.newaxis].T, W[0, np.newaxis]), cmap='gray')

plt.savefig('tmp1.png')
