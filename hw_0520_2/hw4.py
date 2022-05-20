import numpy as np
from scipy.fftpack import fft, dct, idct
import cv2
np.set_printoptions(precision=2)

def _dct(e):
  # return cv2.dct(e/255.)
  return dct(dct(e, axis=0, **dct_opt), axis=1, **dct_opt)

def _idct(e):
  # return cv2.idct(e/255.)
  return idct(idct(e, axis=0, **dct_opt), axis=1, **dct_opt)

#dct_type=1
#dct_opt=dict(type=dct_type, norm='ortho')
dct_opt=dict(norm='ortho')

_in = np.array([
  [164, 157, 209, 149, 132, 182, 129, 142],
  [202, 165, 206, 145, 125, 192, 130, 129],
  [198, 154, 198, 164, 122, 205, 132, 107],
  [165, 190, 221, 174, 139, 195, 123, 106],
  [195, 229, 225, 177, 127, 182, 139, 101],
  [232, 229, 183, 155, 179, 209, 145, 114],
  [234, 190, 146, 199, 227, 220, 149, 120],
  [214, 127, 165, 231, 229, 183, 148, 187]
])

_din = _dct(_in)
print(_din)

_in_shift = _in - 128.
_din_shift = _dct(_in_shift)
print(_din_shift)

_quant = np.array([
  [16, 11, 10, 16, 24, 40, 51, 61],
  [12, 12, 14, 19, 26, 58, 60, 55],
  [14, 13, 16, 24, 40, 57, 69, 56],
  [14, 17, 22, 29, 51, 87, 80, 62],
  [18, 22, 37, 56, 68, 109, 103, 77],
  [24, 35, 55, 64, 81, 104, 113, 92],
  [49, 64, 78, 87, 103, 121, 120, 101],
  [72, 92, 95, 98, 112, 100, 103, 99]
])

_din_shift_quant = np.round(_din_shift / _quant)
print(_din_shift_quant)

_din_shift_quant_deq = _din_shift_quant * _quant
print(_din_shift_quant_deq)

_in_shift_quant_deq = _idct(_din_shift_quant_deq)
print(_in_shift_quant_deq)

_in_shift_des = _in_shift_quant_deq + 128.
print(_in_shift_des)

print(_in_shift_des - _in)
print(np.sqrt(np.sum((_in_shift_des - _in)**2)))

import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.imshow(_in / 255., cmap='gray')
plt.savefig('in.png')

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.imshow(_in_shift_des / 255., cmap='gray')
plt.savefig('in_shift_des.png')
