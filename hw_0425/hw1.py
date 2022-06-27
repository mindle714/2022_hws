import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--image", type=str, required=True)
args = parser.parse_args()

import numpy as np
import imageio as iio
import matplotlib.pyplot as plt
from collections import Counter

def get_entropy(e):
  freq = dict(Counter([str(k) for k in e.flatten()]))
  prob = np.array([e[1] for e in list(freq.items())])
  prob = prob / np.sum(prob)

  return sum([-p*np.log2(p) for p in prob])

def get_bits(e):
  return [(e>>i) % 2 for i in range(8)][::-1]

def conv_gray(e):
  bits = get_bits(e)
  gray_bits = [bits[0]]
  for i in range(1, 8):
    gray_bits.append((bits[i-1] != bits[i]).astype('int'))

  return sum([gray_bits[i] * (2**(7-i)) for i in range(8)])

base = ".".join(args.image.split(".")[:-1])
#with iio.imopen(args.image, "ri", 
#  plugin=None, format_hint=None, legacy_mode=True) as f:
with iio.imopen(args.image, "ri", 
  legacy_mode=True) as f:
  img = f.read()
  print(img[0:3,0:3])

with iio.imopen(args.image, "ri", 
  legacy_mode=False) as f:
  img = f.read()
  print(img[0:3,0:3])

iio.imwrite("{}_orig.png".format(base), img)
#img = iio.imread(args.image)
#with iio.imopen(args.image, "r") as f:
#  img = f.read()
gray_img = conv_gray(img)

bit_planes = get_bits(img)
gray_planes = get_bits(gray_img)

fig = plt.figure(figsize=(19.2, 4.8))

for i in range(8):
  bit_plane = bit_planes[i]
  gray_plane = gray_planes[i]

  bit_entropy = get_entropy(bit_plane)
  gray_entropy = get_entropy(gray_plane)

  ax = fig.add_subplot(2, 8, i+1)
  ax.set_axis_off()
  ax.imshow(bit_plane * 255, cmap='gray')

  ax_pos = ax.get_position()
  fig.text(ax_pos.x0, ax_pos.y0-0.05,
    "bit {} entropy {:.4f}".format(i, bit_entropy))
  
  ax_gray = fig.add_subplot(2, 8, i+9)
  ax_gray.set_axis_off()
  ax_gray.imshow(gray_plane * 255, cmap='gray')
  
  ax_gray_pos = ax_gray.get_position()
  fig.text(ax_gray_pos.x0, ax_gray_pos.y0-0.05,
    "gray {} entropy {:.4f}".format(i, gray_entropy))

plt.savefig('{}_gray.png'.format(base))
