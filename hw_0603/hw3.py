import numpy as np
from scipy.fftpack import fft, dct, idct
import imageio as iio
import matplotlib.pyplot as plt
np.set_printoptions(precision=2, suppress=True)

def rmse(img1, img2):
  return np.sqrt(np.mean((img1 - img2) ** 2))

def psnr(img1, img2):
  _rmse = np.sqrt(np.mean((img1 - img2) ** 2))
  return 20 * np.log10(255 / _rmse)

img_lena = iio.imread("lena.tif")
dct_opt = dict(norm='ortho')

def _dct(e):
  return dct(dct(e, axis=0, **dct_opt), axis=1, **dct_opt)

def _idct(e):
  return idct(idct(e, axis=0, **dct_opt), axis=1, **dct_opt)
    
quant_4x4 = np.array([
  [16, 11, 10, 16],
  [12, 12, 14, 19],
  [14, 13, 16, 24],
  [14, 17, 22, 29]
])
    
quant_8x8 = np.array([
  [16, 11, 10, 16, 24, 40, 51, 61],
  [12, 12, 14, 19, 26, 58, 60, 55],
  [14, 13, 16, 24, 40, 57, 69, 56],
  [14, 17, 22, 29, 51, 87, 80, 62],
  [18, 22, 37, 56, 68, 109, 103, 77],
  [24, 35, 55, 64, 81, 104, 113, 92],
  [49, 64, 78, 87, 103, 121, 120, 101],
  [72, 92, 95, 98, 112, 100, 103, 99]
])

do_sample = False
dct_blk_sample = None

def zonal_dct_blk(img, ratio, do_quant=False):
  global dct_blk_sample

  if img.shape == (4,4):
    _quant = quant_4x4
  elif img.shape == (8,8):
    _quant = quant_8x8
  else:
    assert False, "shape of the block must be either (4,4) or (8,8)"

  remains = int((img.shape[0] * img.shape[1]) * ratio)

  _in_shift = img - 128.
  _din_shift = _dct(_in_shift)

  zigzag_mask = np.zeros_like(_din_shift)

  processed = 0
  cur_pos = (0, 0)
  diag_up = False

  # zig-zag scan to mask out triangular zonal area
  while processed < remains:
    cur_x, cur_y = cur_pos
    zigzag_mask[cur_x, cur_y] = 1.

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

  if do_sample:
    dct_blk_sample = zigzag_mask

  if do_quant:
    _din_shift_quant = np.round((zigzag_mask * _din_shift) / _quant)
    _din_shift_quant_deq = _din_shift_quant * _quant
  else:
    _din_shift_quant_deq = zigzag_mask * _din_shift

  _in_shift_quant_deq = np.round(_idct(_din_shift_quant_deq))

  _in_shift_des = _in_shift_quant_deq + 128.
  return _in_shift_des

def thres_dct_blk(img, ratio, do_quant=False):
  global dct_blk_sample

  if img.shape == (4,4):
    _quant = quant_4x4
  elif img.shape == (8,8):
    _quant = quant_8x8
  else:
    assert False, "shape of the block must be either (4,4) or (8,8)"

  remains = int((img.shape[0] * img.shape[1]) * ratio)

  _in_shift = img - 128.
  _din_shift = _dct(_in_shift)

  thres = np.sort(np.abs(_din_shift.flatten()))[::-1][remains]
  thres_mask = np.where(np.abs(_din_shift) > thres,
    np.ones_like(_din_shift), np.zeros_like(_din_shift))

  if do_sample:
    dct_blk_sample = thres_mask
  
  if do_quant:
    _din_shift_quant = np.round((thres_mask * _din_shift) / _quant)
    _din_shift_quant_deq = _din_shift_quant * _quant
  else:
    _din_shift_quant_deq = thres_mask * _din_shift

  _in_shift_quant_deq = np.round(_idct(_din_shift_quant_deq))

  _in_shift_des = _in_shift_quant_deq + 128.
  return _in_shift_des

def do_dct_compress(img, blk_func, ratio=0.25, blk_shape=(4,4), do_quant=False):
  global do_sample
  assert img.shape[0] % blk_shape[0] == 0
  assert img.shape[1] % blk_shape[1] == 0
  
  img_comp = np.zeros_like(img)
  img_comp_level = np.zeros_like(img)

  for blk_i in range(img.shape[0] // blk_shape[0]):
    for blk_j in range(img.shape[1] // blk_shape[1]):

      do_sample = False
      if blk_i == (img.shape[0] // blk_shape[0]) // 2 + 1 and \
         blk_j == (img.shape[1] // blk_shape[1]) // 2 + 1:
        do_sample = True   

      blk_img = img[blk_i * blk_shape[0] : (blk_i + 1) * blk_shape[0],
                    blk_j * blk_shape[1] : (blk_j + 1) * blk_shape[1]]

      blk_img_comp = blk_func(blk_img, ratio, do_quant=do_quant)
      img_comp[blk_i * blk_shape[0] : (blk_i + 1) * blk_shape[0],
               blk_j * blk_shape[1] : (blk_j + 1) * blk_shape[1]] = blk_img_comp

  return img_comp

def plot_result(name, blk_func, do_quant, disp_err=False):
  title_opt = dict(fontsize=5)
  fig = plt.figure()

  ax = fig.add_subplot(4, 4, 1)
  ax.set_axis_off()
  ax.imshow(img_lena, cmap='gray')
  ax.set_title('original', **title_opt)
  
  for ratio, idx in zip([0.25, 0.125, 0.0625], [2, 3, 4]):
    ax = fig.add_subplot(4, 4, idx)
    ax.set_axis_off()

    img_comp = do_dct_compress(img_lena, blk_func,
      ratio=ratio, blk_shape=(4,4), do_quant=do_quant)

    if not disp_err:
      ax.imshow(img_comp, cmap='gray')
    else:
      ax.imshow(img_comp - img_lena, cmap='gray')

    ax.set_title('block=(4,4), ratio={}\nrmse={:.2f}, psnr={:.2f}'.format(
      ratio, rmse(img_lena, img_comp), psnr(img_lena, img_comp)), **title_opt)
    
    ax = fig.add_subplot(4, 4, idx + 4)
    ax.set_axis_off()
    ax.pcolormesh(dct_blk_sample, edgecolors='k', cmap='gray', linewidth=1)
    ax.invert_yaxis()

  ax = fig.add_subplot(4, 4, 9)
  ax.set_axis_off()
  ax.imshow(img_lena, cmap='gray')
  ax.set_title('original', **title_opt)

  for ratio, idx in zip([0.25, 0.125, 0.0625], [10, 11, 12]):
    ax = fig.add_subplot(4, 4, idx)
    ax.set_axis_off()
    
    img_comp = do_dct_compress(img_lena, blk_func,
      ratio=ratio, blk_shape=(8,8), do_quant=do_quant)

    if not disp_err:
      ax.imshow(img_comp, cmap='gray')
    else:
      ax.imshow(img_comp - img_lena, cmap='gray')

    ax.set_title('block=(8,8), ratio={}\nrmse={:.2f}, psnr={:.2f}'.format(
      ratio, rmse(img_lena, img_comp), psnr(img_lena, img_comp)), **title_opt)
    
    ax = fig.add_subplot(4, 4, idx + 4)
    ax.set_axis_off()
    ax.pcolormesh(dct_blk_sample, edgecolors='k', cmap='gray', linewidth=1)
    ax.invert_yaxis()

  fig.tight_layout()
  plt.savefig('{}.png'.format(name))

#plot_result('zonal', zonal_dct_blk, False)
#plot_result('thres', thres_dct_blk, False)

plot_result('zonal_quant', zonal_dct_blk, True)
plot_result('thres_quant', thres_dct_blk, True)

plot_result('zonal_quant_err', zonal_dct_blk, True, True)
plot_result('thres_quant_err', thres_dct_blk, True, True)
