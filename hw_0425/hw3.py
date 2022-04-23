import numpy as np
import imageio as iio
import matplotlib.pyplot as plt

img = iio.imread("lena.png")
diff_img = img.flatten() - np.concatenate([[0], img.flatten()[:-1]], 0)
diff_img = diff_img.reshape(img.shape)
diff_trunc_img = np.clip(diff_img, -16, 16)

from huffman import *
from collections import Counter

def get_freq(e):
  freq = dict(Counter([str(k) for k in e.flatten()]))
  freq = sorted(freq.items(), key=lambda x: x[1], reverse=True)
  return freq

img_freq = get_freq(img)
diff_img_freq = get_freq(diff_img)
diff_trunc_img_freq = get_freq(diff_trunc_img)

img_code = huffman_code_tree(make_tree(img_freq))
diff_img_code = huffman_code_tree(make_tree(diff_img_freq))
diff_trunc_img_code = huffman_code_tree(make_tree(diff_trunc_img_freq))

def get_avg_len(freqs, code):
  avg_len = []
  for sym, freq in freqs:
    avg_len.append(freq * len(code[sym]))
  avg_len = sum(avg_len) / sum([e[1] for e in freqs])
  return avg_len

avg_img_len = get_avg_len(img_freq, img_code)
avg_diff_img_len = get_avg_len(diff_img_freq, diff_img_code)
avg_diff_trunc_img_len = get_avg_len(diff_trunc_img_freq, diff_trunc_img_code)

print("{} {} {}".format(
  8/avg_img_len, 8/avg_diff_img_len, 8/avg_diff_trunc_img_len))

fig = plt.figure(figsize=(19.2, 4.8))

ax = fig.add_subplot(1, 3, 1)
ax.hist(img.flatten(), density=True, bins=60)
ax.set_title("compression ratio: {}".format(8/avg_img_len))

ax_diff = fig.add_subplot(1, 3, 2)
ax_diff.hist(diff_img.flatten(), density=True, bins=60)
ax_diff.set_title("compression ratio: {}".format(8/avg_diff_img_len))

ax_diff_trunc = fig.add_subplot(1, 3, 3)
ax_diff_trunc.set_xlim(ax_diff.get_xlim())
ax_diff_trunc.hist(diff_trunc_img.flatten(), density=True, bins=8)
ax_diff_trunc.set_title("compression ratio: {}".format(8/avg_diff_trunc_img_len))

plt.savefig('hist.png')
