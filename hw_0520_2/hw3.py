import numpy as np
from scipy.fftpack import fft, dct, idct
import cv2
np.set_printoptions(precision=2)

_in = np.array([
  [10, 11, 12, 11, 12, 13, 12, 11],
  [10, -10, 8, -7, 8, -8, 7, -7]
])

print(dct(_in[0], axis=0, norm='ortho'))
print(dct(_in[1], axis=0, norm='ortho'))
print(dct(_in.reshape([16]), axis=0, norm='ortho'))

import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.add_subplot(3,1,1)
ax.plot(np.arange(8), dct(_in[0], axis=0, norm='ortho'))
ax = fig.add_subplot(3,1,2)
ax.plot(np.arange(8), dct(_in[1], axis=0, norm='ortho'))
ax = fig.add_subplot(3,1,3)
ax.plot(np.arange(16), dct(_in.reshape([16]), axis=0, norm='ortho'))
plt.savefig('hw3.png')
