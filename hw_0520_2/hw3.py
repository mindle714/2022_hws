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
