import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--train-list", type=str, required=True) 
parser.add_argument("--vocab", type=str, required=True) 
parser.add_argument("--num-chunks", type=int, required=False, default=100)
parser.add_argument("--samp-len", type=int, required=False, default=8192)
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
with open(args_file, "w") as f:
  f.write(json.dumps(vars(args)))
os.chmod(args_file, S_IREAD|S_IRGRP|S_IROTH)

train_list = [e.strip() for e in open(args.train_list).readlines()]
vocab = {e.strip():idx for idx, e in enumerate(open(args.vocab).readlines())}

import warnings
import tensorflow as tf
import soundfile
import tqdm

num_chunks = min(len(train_list), args.num_chunks)
writers = [tf.io.TFRecordWriter(os.path.join(
    args.output, "train-{}.tfrecord".format(idx))) for idx in range(num_chunks)]

chunk_idx = 0
for idx, _list in tqdm.tqdm(enumerate(train_list), total=len(train_list)):
  if len(_list.split()) != 2:
    warnings.warn("failed to parse {} at line {}".format(_list, idx))
    continue

  spk, wav = _list.split()
  if spk not in vocab:
    warnings.warn("vocab {} missing {}".format(args.vocab, spk))
    continue

  pcm, sr = soundfile.read(wav)
  if pcm.shape[0] < args.samp_len:
    warnings.warn("too short to pack {} at line {}".format(wav, idx))
    continue

  hop_len = args.samp_len//2
  for pcm_idx in range((pcm.shape[0]-args.samp_len)//hop_len):
    _pcm = pcm[pcm_idx*hop_len : pcm_idx*hop_len+args.samp_len]

    pcm_feat = tf.train.Feature(float_list=tf.train.FloatList(value=_pcm))
    spk_feat = tf.train.Feature(int64_list=tf.train.Int64List(value=[vocab[spk]]))
    feats = {'pcm': pcm_feat, 'speaker': spk_feat}

    ex = tf.train.Example(features=tf.train.Features(feature=feats))
    writers[chunk_idx].write(ex.SerializeToString())
    chunk_idx = (chunk_idx+1) % num_chunks

for writer in writers:
  writer.close()
