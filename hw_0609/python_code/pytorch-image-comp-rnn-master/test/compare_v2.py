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
    _im = imread(os.path.join(hyp, '15.png'))

    ax = fig.add_subplot(1, len(args.hyps), idx+1)
    ax.set_axis_off()
    ax.set_title(label)

    ax.imshow(_im)

plt.savefig(args.output)
