import numpy as np
import imageio as iio
import matplotlib.pyplot as plt

img = iio.imread("lena.png")

def quant(e, _min=-255, _max=255, level=2**4):
  step = (_max - _min) / level
  qvals = np.linspace(_min, _max, level)

  if (e + step / 2) < qvals[0]: return qvals[0]
  for qval in qvals:
    if (e + step / 2) < qval: return qval
  return qvals[-1]

def dpcm_enc(img, wa, wb, wc):
  pred_img = np.zeros_like(img)
  quant_error = np.zeros_like(img)

  for ridx in range(img.shape[0]):
    for cidx in range(img.shape[1]):
      have_a = (cidx > 0)
      have_b = (ridx > 0)
      have_c = (ridx > 0 and cidx < (img.shape[1]-1))

      a = pred_img[ridx][cidx-1] if have_a else 0
      b = pred_img[ridx-1][cidx] if have_b else 0
      c = pred_img[ridx-1][cidx+1] if have_c else 0

      pred = wa * a + wb * b + wc * c
      qerr = quant(img[ridx][cidx] - pred)
      quant_error[ridx][cidx] = qerr
      pred_img[ridx][cidx] = pred + qerr

  return quant_error

def dpcm_dec(quant_error, wa, wb, wc):
  pred_img = np.zeros_like(quant_error)

  for ridx in range(quant_error.shape[0]):
    for cidx in range(quant_error.shape[1]):
      have_a = (cidx > 0)
      have_b = (ridx > 0)
      have_c = (ridx > 0 and cidx < (quant_error.shape[1]-1))

      a = pred_img[ridx][cidx-1] if have_a else 0
      b = pred_img[ridx-1][cidx] if have_b else 0
      c = pred_img[ridx-1][cidx+1] if have_c else 0

      pred = wa * a + wb * b + wc * c
      pred_img[ridx][cidx] = pred + quant_error[ridx][cidx]

  return pred_img

fig = plt.figure()

ax = fig.add_subplot(2, 3, 1)
ax.set_axis_off()
ax.imshow(img, cmap='gray')

ax = fig.add_subplot(2, 3, 2)
ax.set_axis_off()
dpcm_qerr = dpcm_enc(img, 0.97, 0., 0.)
dpcm_img = dpcm_dec(dpcm_qerr, 0.97, 0., 0.)
ax.imshow(dpcm_img, cmap='gray')

ax = fig.add_subplot(2, 3, 5)
ax.set_axis_off()
ax.imshow(-(dpcm_img - img), cmap='gray')

ax = fig.add_subplot(2, 3, 3)
ax.set_axis_off()
dpcm_qerr2 = dpcm_enc(img, 0.75, -0.5, 0.75)
dpcm_img2 = dpcm_dec(dpcm_qerr2, 0.75, -0.5, 0.75)
ax.imshow(dpcm_img2, cmap='gray')

ax = fig.add_subplot(2, 3, 6)
ax.set_axis_off()
ax.imshow(-(dpcm_img2 - img), cmap='gray')

plt.savefig('dpcm.png')
