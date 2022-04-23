import numpy as np
import imageio as iio
import matplotlib.pyplot as plt

img = iio.imread("lena.png")

def quant(e, _min=-255, _max=255, level=2**4):
  step = (_max - _min) / level
  qvals = np.linspace(_min, _max, level)

  # TODO
  if (e + step / 2) < qvals[0]: return qvals[0]
  for qval in qvals:
    if (e + step / 2) < qval: return qval
  return qvals[-1]

def dpcm(img, wa, wb, wc):
  pred_img = np.zeros_like(img)

  for ridx in range(img.shape[0]):
    for cidx in range(img.shape[1]):
      a = pred_img[ridx][cidx-1] if cidx > 0 else 0
      b = pred_img[ridx-1][cidx] if ridx > 0 else 0
      c = pred_img[ridx-1][cidx+1] if (ridx > 0 and cidx < (img.shape[1]-1)) else 0

      pred = wa * a + wb * b + wc * c
      quant_err = quant(img[ridx][cidx] - pred)
      pred_img[ridx][cidx] = pred + quant_err # TODO

  return pred_img

fig = plt.figure()

ax = fig.add_subplot(1, 3, 1)
ax.set_axis_off()
ax.imshow(img, cmap='gray')

ax = fig.add_subplot(1, 3, 2)
ax.set_axis_off()
ax.imshow(dpcm(img, 0.97, 0., 0.), cmap='gray')

ax = fig.add_subplot(1, 3, 3)
ax.set_axis_off()
ax.imshow(dpcm(img, 0.75, -0.5, 0.75), cmap='gray')

plt.savefig('dpcm.png')
