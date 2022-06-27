import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--hyps', nargs='+', default=[])
parser.add_argument('--labels', nargs='+', default=[])
parser.add_argument('--output', type=str, required=True)
args = parser.parse_args()

assert len(args.hyps) == len(args.labels)
assert len(args.hyps) > 0

import os
import numpy as np
from imageio import imread
import matplotlib.pyplot as plt

#fig = plt.figure(figsize=(19.2, 4.8))
fig = plt.figure()

for idx, (hyp, label) in enumerate(zip(args.hyps, args.labels)):
    _imfull = []
    for _subidx in [0, 3, 6, 9, 12, 15]: 
        _im = imread(os.path.join(hyp, '{}.png'.format('%02d' % _subidx)))
        _imfull.append(_im)

    ax = fig.add_subplot(len(args.hyps), 1, idx+1)
    ax.set_axis_off()
    ax.set_title(label)

    _imfull = np.concatenate(_imfull, axis=1)
    ax.imshow(_imfull)

plt.savefig(args.output)
