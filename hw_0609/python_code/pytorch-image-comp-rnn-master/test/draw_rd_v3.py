import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--ref', type=str, required=False,
    default='/home/hejung/2022_hws/hw_0609/python_code/pytorch-image-comp-rnn-master/cal/jpeg_ssim.csv')
parser.add_argument('--hyps', nargs='+', default=[])
parser.add_argument('--labels', nargs='+', default=[])
parser.add_argument('--output', type=str, required=True)
parser.add_argument('--iter', type=int, required=False, default=16)
args = parser.parse_args()

assert len(args.hyps) == len(args.labels)
assert len(args.hyps) > 0

import os
import numpy as np
from imageio import imread
import matplotlib.pyplot as plt

fig = plt.figure(figsize=(9.6, 19.2))
ax = fig.add_subplot(2, 1, 1)

for hyp, label in zip(args.hyps, args.labels):
    lstm_ssim = np.genfromtxt(hyp, delimiter=',')
    lstm_ssim = lstm_ssim[:, :-1]
    lstm_ssim = np.mean(lstm_ssim, axis=0)[:args.iter]
    lstm_bpp = np.arange(1, lstm_ssim.shape[0]+1) / 192 * 24
    ax.plot(lstm_bpp, lstm_ssim, label=label, marker='o')

jpeg_ssim = np.genfromtxt(args.ref, delimiter=',')
jpeg_ssim = jpeg_ssim[:, :-1]
jpeg_ssim = np.mean(jpeg_ssim, axis=0)

jpeg_bpp = np.array([
    os.path.getsize('jpeg/kodim{:02d}/{:02d}.jpg'.format(i, q)) * 8 /
    (imread('jpeg/kodim{:02d}/{:02d}.jpg'.format(i, q)).size // 3)
    for i in range(1, 25) for q in range(1, 16)
]).reshape(24, 15)

jpeg_bpp = np.mean(jpeg_bpp, axis=0)
ax.plot(jpeg_bpp, jpeg_ssim, label='JPEG', marker='x')

ax.set_title("RD curve on 24 Kodak Photo CD Dataset")
ax.set_xlim(0., 1.5)
ax.set_ylim(0.7, 1.0)
ax.set_xlabel('bit per pixel')
ax.set_ylabel('MS-SSIM')
ax.legend()

ax = fig.add_subplot(2, 1, 2)

for hyp, label in zip(args.hyps, args.labels):
    lstm_ssim = np.genfromtxt(hyp, delimiter=',')
    lstm_ssim = lstm_ssim[[4, 10, 17, 18, 19], :]
    lstm_ssim = lstm_ssim[:, :-1]
    lstm_ssim = np.mean(lstm_ssim, axis=0)[:args.iter]
    lstm_bpp = np.arange(1, lstm_ssim.shape[0]+1) / 192 * 24
    ax.plot(lstm_bpp, lstm_ssim, label=label, marker='o')

jpeg_ssim = np.genfromtxt(args.ref, delimiter=',')
jpeg_ssim = jpeg_ssim[[4, 10, 17, 18, 19], :]
jpeg_ssim = jpeg_ssim[:, :-1]
jpeg_ssim = np.mean(jpeg_ssim, axis=0)

jpeg_bpp = np.array([
    os.path.getsize('jpeg/kodim{:02d}/{:02d}.jpg'.format(i, q)) * 8 /
    (imread('jpeg/kodim{:02d}/{:02d}.jpg'.format(i, q)).size // 3)
    for i in range(1, 25) for q in range(1, 16)
]).reshape(24, 15)

jpeg_bpp = np.mean(jpeg_bpp, axis=0)
ax.plot(jpeg_bpp, jpeg_ssim, label='JPEG', marker='x')

ax.set_title("RD curve on index {4, 10, 17, 18, 19} of the Kodak dataset")
ax.set_xlim(0., 1.5)
ax.set_ylim(0.7, 1.0)
ax.set_xlabel('bit per pixel')
ax.set_ylabel('MS-SSIM')
ax.legend()

plt.savefig(args.output)
