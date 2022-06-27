import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--ref', type=str, required=False,
    default='/home/hejung/2022_hws/hw_0609/python_code/pytorch-image-comp-rnn-master/cal/jpeg_ssim.csv')
parser.add_argument('--hyps', nargs='+', default=[])
parser.add_argument('--labels', nargs='+', default=[])
parser.add_argument('--output', type=str, required=True)
parser.add_argument('--iter', type=int, required=False, default=16)
parser.add_argument('--subset', action='store_true')
args = parser.parse_args()

assert len(args.hyps) == len(args.labels)
assert len(args.hyps) > 0

import os
import numpy as np
from imageio import imread
import matplotlib.pyplot as plt

for hyp, label in zip(args.hyps, args.labels):
    lstm_ssim = np.genfromtxt(hyp, delimiter=',')
    if args.subset:
        lstm_ssim = lstm_ssim[[4, 10, 17, 18, 19], :]
    lstm_ssim = lstm_ssim[:, :-1]
    lstm_ssim = np.mean(lstm_ssim, axis=0)[:args.iter]
    lstm_bpp = np.arange(1, lstm_ssim.shape[0]+1) / 192 * 24
    plt.plot(lstm_bpp, lstm_ssim, label=label, marker='o')

jpeg_ssim = np.genfromtxt(args.ref, delimiter=',')
if args.subset:
    jpeg_ssim = jpeg_ssim[[4, 10, 17, 18, 19], :]
jpeg_ssim = jpeg_ssim[:, :-1]
jpeg_ssim = np.mean(jpeg_ssim, axis=0)

jpeg_bpp = np.array([
    os.path.getsize('jpeg/kodim{:02d}/{:02d}.jpg'.format(i, q)) * 8 /
    (imread('jpeg/kodim{:02d}/{:02d}.jpg'.format(i, q)).size // 3)
    for i in range(1, 25) for q in range(1, 16)
]).reshape(24, 15)

jpeg_bpp = np.mean(jpeg_bpp, axis=0)
plt.plot(jpeg_bpp, jpeg_ssim, label='JPEG', marker='x')

plt.xlim(0., 1.5)
plt.ylim(0.7, 1.0)
plt.xlabel('bit per pixel')
plt.ylabel('MS-SSIM')
plt.legend()
#plt.show()
plt.savefig(args.output)
