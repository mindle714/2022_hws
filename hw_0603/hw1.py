import imageio as iio
import numpy as np
import matplotlib.pyplot as plt

def rmse(img1, img2):
  return np.sqrt(np.mean((img1 - img2) ** 2))

def psnr(img1, img2):
  _rmse = np.sqrt(np.mean((img1 - img2) ** 2))
  return 20 * np.log10(255 / _rmse)

img_lena = iio.imread("lena.tif")
img_mandrill = iio.imread("smandril.tif")

def do_btc(img, blk_func, blk_shape=(4,4)):
  assert img.shape[0] % blk_shape[0] == 0
  assert img.shape[1] % blk_shape[1] == 0

  img_comp = np.zeros_like(img)
  img_comp_level = np.zeros_like(img)

  for blk_i in range(img.shape[0] // blk_shape[0]):
    for blk_j in range(img.shape[1] // blk_shape[1]):
      blk_img = img[blk_i * blk_shape[0] : (blk_i + 1) * blk_shape[0],
                    blk_j * blk_shape[1] : (blk_j + 1) * blk_shape[1]]

      blk_img_comp_level, a, b = blk_func(blk_img)
      blk_img_comp = np.where(blk_img_comp_level == 1,
              np.ones_like(blk_img) * b, np.ones_like(blk_img) * a)

      img_comp_level[blk_i * blk_shape[0] : (blk_i + 1) * blk_shape[0],
                     blk_j * blk_shape[1] : (blk_j + 1) * blk_shape[1]] = blk_img_comp_level
      img_comp[blk_i * blk_shape[0] : (blk_i + 1) * blk_shape[0],
               blk_j * blk_shape[1] : (blk_j + 1) * blk_shape[1]] = blk_img_comp

  return img_comp_level, img_comp
  
def sbtc_blk(img):
  m = img.shape[0] * img.shape[1]

  img = img.astype(np.int64)
  img_mean = np.mean(img)
  img_var = np.mean(img**2) - (img_mean**2)
  img_std = np.sqrt(img_var)

  q = np.sum(img > img_mean)
  a = img_mean - img_std * np.sqrt(q / (m - q))
  b = img_mean + img_std * np.sqrt((m - q) / q)
  #print("a[{}] b[{}]".format(a, b))

  img_comp = 1 * (img > img_mean) + 0 * (1 - (img > img_mean))
  return img_comp, int(a), int(b)

def ambtc_blk(img):
  m = img.shape[0] * img.shape[1]

  img = img.astype(np.int64)
  img_mean = np.mean(img)
  alpha = np.mean(np.abs(img - img_mean))

  q = np.sum(img > img_mean)
  a = img_mean - (m * alpha) / (2 * (m - q))
  b = img_mean + (m * alpha) / (2 * q)
  #print("a[{}] b[{}]".format(a, b))

  img_comp = 1 * (img > img_mean) + 0 * (1 - (img > img_mean))
  return img_comp, int(a), int(b)

def plot_result(name, 
                img_lena, img_lena_comp_level, img_lena_comp,
                img_mandrill, img_mandrill_comp_level, img_mandrill_comp):

  fig = plt.figure()

  ax = fig.add_subplot(2, 4, 1)
  ax.set_axis_off()
  ax.imshow(img_lena, cmap='gray')

  ax = fig.add_subplot(2, 4, 2)
  ax.set_axis_off()
  ax.imshow(img_lena_comp, cmap='gray')

  ax = fig.add_subplot(2, 4, 3)
  ax.set_axis_off()
  ax.imshow(img_lena_comp_level, cmap='gray')

  ax = fig.add_subplot(2, 4, 4)
  ax.set_axis_off()
  ax.imshow(img_lena_comp - img_lena, cmap='gray')

  ax = fig.add_subplot(2, 4, 5)
  ax.set_axis_off()
  ax.imshow(img_mandrill, cmap='gray')

  ax = fig.add_subplot(2, 4, 6)
  ax.set_axis_off()
  ax.imshow(img_mandrill_comp, cmap='gray')

  ax = fig.add_subplot(2, 4, 7)
  ax.set_axis_off()
  ax.imshow(img_mandrill_comp_level, cmap='gray')

  ax = fig.add_subplot(2, 4, 8)
  ax.set_axis_off()
  ax.imshow(img_mandrill_comp - img_mandrill, cmap='gray')

  print("{} lena RMSE[{}] PSNR[{}]".format(
      name,
      rmse(img_lena, img_lena_comp),
      psnr(img_lena, img_lena_comp)))
  print("{} mandrill RMSE[{}] PSNR[{}]".format(
      name,
      rmse(img_mandrill, img_mandrill_comp),
      psnr(img_mandrill, img_mandrill_comp)))

  plt.savefig('{}.png'.format(name))

img_lena_comp_level, img_lena_comp = do_btc(img_lena, sbtc_blk)
img_mandrill_comp_level, img_mandrill_comp = do_btc(img_mandrill, sbtc_blk)

plot_result('sbtc',
            img_lena, img_lena_comp_level, img_lena_comp,
            img_mandrill, img_mandrill_comp_level, img_mandrill_comp)

img_lena_comp_level, img_lena_comp = do_btc(img_lena, ambtc_blk)
img_mandrill_comp_level, img_mandrill_comp = do_btc(img_mandrill, ambtc_blk)

plot_result('ambtc',
            img_lena, img_lena_comp_level, img_lena_comp,
            img_mandrill, img_mandrill_comp_level, img_mandrill_comp)
