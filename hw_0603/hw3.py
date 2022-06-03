import numpy as np
from scipy.fftpack import fft, dct, idct
import cv2
import imageio as iio
np.set_printoptions(precision=2)

img_lena = iio.imread("lena.tif")

#dct_type=1
#dct_opt=dict(type=dct_type, norm='ortho')
dct_opt=dict(norm='ortho')

def _dct(e):
  # return cv2.dct(e/255.)
  return dct(dct(e, axis=0, **dct_opt), axis=1, **dct_opt)

def _idct(e):
  # return cv2.idct(e/255.)
  return idct(idct(e, axis=0, **dct_opt), axis=1, **dct_opt)

'''
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
'''

def zonal_dct_blk(img, ratio):
  remains = int((img.shape[0] * img.shape[1]) * ratio)

  _in_shift = img - 128.
  _din_shift = _dct(_in_shift)

  # _din_shift_quant = np.round(_din_shift / _quant)
  # _din_shift_quant_deq = _din_shift_quant * _quant
  _din_shift_quant_deq = np.zeros_like(_din_shift)

  processed = 0
  cur_pos = (0, 0)
  diag_up = False

  while processed < remains:
    cur_x, cur_y = cur_pos
    _din_shift_quant_deq[cur_x, cur_y] = _din_shift[cur_x, cur_y]

    if cur_x == 0 and cur_y % 2 == 0:
      cur_pos = (0, cur_y + 1)
      diag_up = False

    elif cur_x % 2 == 1 and cur_y == 0:
      cur_pos = (cur_x + 1, 0)
      diag_up = True

    else:
      if diag_up:
        cur_pos = (cur_x - 1, cur_y + 1)
      else:
        cur_pos = (cur_x + 1, cur_y - 1)

    processed += 1

  _in_shift_quant_deq = np.round(_idct(_din_shift_quant_deq))

  _in_shift_des = _in_shift_quant_deq + 128.
  return _in_shift_des

def do_dct_compress(img, blk_func, ratio=0.25, blk_shape=(4,4)):
  assert img.shape[0] % blk_shape[0] == 0
  assert img.shape[1] % blk_shape[1] == 0
  
  img_comp = np.zeros_like(img)
  img_comp_level = np.zeros_like(img)

  for blk_i in range(img.shape[0] // blk_shape[0]):
    for blk_j in range(img.shape[1] // blk_shape[1]):
      blk_img = img[blk_i * blk_shape[0] : (blk_i + 1) * blk_shape[0],
                    blk_j * blk_shape[1] : (blk_j + 1) * blk_shape[1]]

      blk_img_comp = blk_func(blk_img, ratio)
      img_comp[blk_i * blk_shape[0] : (blk_i + 1) * blk_shape[0],
               blk_j * blk_shape[1] : (blk_j + 1) * blk_shape[1]] = blk_img_comp

  return img_comp

import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.add_subplot(1, 4, 1)
ax.set_axis_off()
ax.imshow(img_lena, cmap='gray')

ax = fig.add_subplot(1, 4, 2)
ax.set_axis_off()
ax.imshow(do_dct_compress(img_lena, zonal_dct_blk, ratio=0.25), cmap='gray')

ax = fig.add_subplot(1, 4, 3)
ax.set_axis_off()
ax.imshow(do_dct_compress(img_lena, zonal_dct_blk, ratio=0.125), cmap='gray')

ax = fig.add_subplot(1, 4, 4)
ax.set_axis_off()
ax.imshow(do_dct_compress(img_lena, zonal_dct_blk, ratio=0.0625), cmap='gray')

plt.savefig('viz.png')
