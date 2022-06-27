import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--ref', type=str, required=True, help='reference jpeg csv')
parser.add_argument('--hyp', type=str, required=True, help='hypothesis model csv')
parser.add_argument('--output', type=str, required=True, help='output png')
args = parser.parse_args()

import os
import numpy as np
from imageio import imread
import matplotlib.pyplot as plt

line = True

lstm_ssim = np.genfromtxt(args.hyp, delimiter=',')
lstm_ssim = lstm_ssim[:, :-1]
if line:
    lstm_ssim = np.mean(lstm_ssim, axis=0)
    lstm_bpp = np.arange(1, 17) / 192 * 24
    plt.plot(lstm_bpp, lstm_ssim, label='LSTM', marker='o')
else:
    lstm_bpp = np.stack([np.arange(1, 17) for _ in range(24)]) / 192 * 24
    plt.scatter(
        lstm_bpp.reshape(-1), lstm_ssim.reshape(-1), label='LSTM', marker='o')

jpeg_ssim = np.genfromtxt(args.ref, delimiter=',')
jpeg_ssim = jpeg_ssim[:, :-1]
if line:
    jpeg_ssim = np.mean(jpeg_ssim, axis=0)

jpeg_bpp = np.array([
    os.path.getsize('jpeg/kodim{:02d}/{:02d}.jpg'.format(i, q)) * 8 /
    (imread('jpeg/kodim{:02d}/{:02d}.jpg'.format(i, q)).size // 3)
    for i in range(1, 25) for q in range(1, 16)
]).reshape(24, 15)

if line:
    jpeg_bpp = np.mean(jpeg_bpp, axis=0)
    plt.plot(jpeg_bpp, jpeg_ssim, label='JPEG', marker='x')
else:
    plt.scatter(
        jpeg_bpp.reshape(-1), jpeg_ssim.reshape(-1), label='JPEG', marker='x')

plt.xlim(0., 2.)
plt.ylim(0.7, 1.0)
plt.xlabel('bit per pixel')
plt.ylabel('MS-SSIM')
plt.legend()
#plt.show()
plt.savefig(args.output)
