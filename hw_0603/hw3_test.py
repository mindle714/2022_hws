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

W = np.zeros([N, N], dtype='float')
for n in range(N):
  for k in range(N):
    W[n][k] = np.sqrt(2./N) * np.cos(np.pi/N*(n+.5)*k)
	
print(scipy.fftpack.dct(a, type=2, norm='ortho'))
e = np.matmul(a, W)
e[0] /= np.sqrt(2.)
print(e)

fig = plt.figure()

ax = fig.add_subplot(1, 1, 1)
ax.set_axis_off()
ax.imshow(W, cmap='gray')

plt.savefig('dct.png')

fig = plt.figure()

ax = fig.add_subplot(1, 1, 1)
ax.set_axis_off()
ax.imshow(np.matmul(W.T, W), cmap='gray')

plt.savefig('dct2.png')

fig = plt.figure()

ax = fig.add_subplot(1, 1, 1)
ax.set_axis_off()
ax.imshow(np.matmul(W, W.T), cmap='gray')

plt.savefig('dct2_v2.png')

from scipy.fftpack import fft, dct, idct

dct_opt=dict(norm='ortho')

def _dct(e):
  # return cv2.dct(e/255.)
  return dct(dct(e, axis=0, **dct_opt), axis=1, **dct_opt)

def _idct(e):
  # return cv2.idct(e/255.)
  return idct(idct(e, axis=0, **dct_opt), axis=1, **dct_opt)

fig = plt.figure()

ax = fig.add_subplot(1, 1, 1)
ax.set_axis_off()
ax.imshow(_dct(np.eye(8)), cmap='gray')

plt.savefig('dct3.png')
