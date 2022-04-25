import numpy as np
import imageio as iio
import matplotlib.pyplot as plt

from huffman import *
from collections import Counter

def get_freq(e):
  freq = dict(Counter([str(k) for k in e.flatten()]))
  freq = sorted(freq.items(), key=lambda x: x[1], reverse=True)
  return freq

seq = np.array([6,2,1,4,2,1,2,3,5,2,12,6,3,6,7,11,6,8,10,11,7,13,14,6,6,14,9,13])
freq = get_freq(seq)
freq_dict = dict(freq)
code = huffman_code_tree(make_tree(freq))

bl = []
for k in range(1,15):
  k = str(k)
  print("{}:{} {} {}/{}".format(k,code[k], len(code[k]), freq_dict[k],sum([e[1] for e in freq])))
  bl.append(len(code[k])*(freq_dict[k]/sum([e[1] for e in freq])))
print(sum(bl))
