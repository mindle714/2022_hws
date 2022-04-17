import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--wav-list", type=str, required=True) 
parser.add_argument("--trans-list", type=str, required=True) 
parser.add_argument("--num-chunks", type=int, required=False, default=100)
parser.add_argument("--samp-len", type=int, required=False, default=14*16000)
parser.add_argument("--output", type=str, required=True) 
args = parser.parse_args()

import os
import sys
import json
from stat import S_IREAD, S_IRGRP, S_IROTH, S_IWUSR

args_file = os.path.join(args.output, "ARGS")
if os.path.isdir(args.output):
  msg = 'directory {} exists. Do you want to proceed?'.format(args.output)
  cont = input("%s (y/N) " % msg).lower() == 'y'
  if not cont: sys.exit(0)
  if os.path.isfile(args_file):
    os.chmod(args_file, S_IWUSR|S_IREAD)

os.makedirs(args.output, exist_ok=True)
wav_list = [e.strip() for e in open(args.wav_list).readlines()]
trans_list = [e.strip() for e in open(args.trans_list).readlines()]
assert len(wav_list) == len(trans_list)

vocab = {'<blank>': 0}; vidx_list = []
max_trans = 0

for trans in trans_list:
  trans = trans.replace("<NOISE>", "").strip()
  trans = " ".join(trans.split())

  vidx = []
  for e in trans:
    if e not in vocab:
      vocab[e] = len(vocab)
    vidx.append(vocab[e])
  
  vidx_list.append(vidx)
  max_trans = max(len(vidx), max_trans)

args.trans_len = max_trans
with open(os.path.join(args.output, "vocab"), "w") as f:
  for k in vocab:
    f.write("{}\t{}\n".format(k, vocab[k]))

import warnings
import tensorflow as tf
import numpy as np
import soundfile
import tqdm

num_chunks = min(len(wav_list), args.num_chunks)
writers = [tf.io.TFRecordWriter(os.path.join(
    args.output, "train-{}.tfrecord".format(idx))) for idx in range(num_chunks)]

chunk_idx = 0; chunk_lens = [0 for _ in range(num_chunks)]
ignored = 0

for idx, (_pcm, vidx) in tqdm.tqdm(enumerate(zip(wav_list, vidx_list)), total=len(wav_list)):
  pcm, _ = soundfile.read(_pcm)

  samp_len = args.samp_len
  if pcm.shape[0] > samp_len:
    ignored += 1
    continue
  
  pcm_len = pcm.shape[0]
  trans_len = len(vidx)

  pcm = np.concatenate([pcm,
    np.zeros(samp_len - pcm.shape[0], dtype=pcm.dtype)], 0)
  trans = np.concatenate([vidx,
    np.zeros(max_trans - len(vidx), dtype=np.int32)])

  pcm_feat = tf.train.Feature(float_list=tf.train.FloatList(value=pcm))
  pcm_len_feat = tf.train.Feature(int64_list=tf.train.Int64List(value=[pcm_len]))

  trans_feat = tf.train.Feature(int64_list=tf.train.Int64List(value=trans))
  trans_len_feat = tf.train.Feature(int64_list=tf.train.Int64List(value=[trans_len]))

  feats = {
    'pcm': pcm_feat, 'pcm_len': pcm_len_feat, 
    'trans': trans_feat, 'trans_len': trans_len_feat
  }

  ex = tf.train.Example(features=tf.train.Features(feature=feats))
  writers[chunk_idx].write(ex.SerializeToString())

  chunk_lens[chunk_idx] += 1
  chunk_idx = (chunk_idx+1) % num_chunks

for writer in writers:
  writer.close()

args.chunk_lens = chunk_lens
args.ignored = ignored

with open(args_file, "w") as f:
  f.write(" ".join([sys.executable] + sys.argv))
  f.write("\n")
  f.write(json.dumps(vars(args)))
os.chmod(args_file, S_IREAD|S_IRGRP|S_IROTH)

import shutil
for origin in [os.path.abspath(__file__)]:
  shutil.copy(origin, args.output)
