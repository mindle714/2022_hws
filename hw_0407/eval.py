import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--ckpt", type=str, required=True) 
parser.add_argument("--eval-list", type=str, required=True) 
args = parser.parse_args()

import os
import sys

expdir = os.path.abspath(os.path.dirname(args.ckpt))
sys.path.insert(0, expdir)

import model
if os.path.dirname(model.__file__) != expdir:
  sys.exit("model is loaded from {}".format(model.__file__))

import json
exp_args = os.path.join(expdir, "ARGS")
with open(exp_args, "r") as f:
  jargs = json.loads(f.readlines()[-1])
  vocab = {e.strip():idx for idx, e in enumerate(open(jargs["vocab"]).readlines())}

  with open(os.path.join(jargs["tfrec"], "ARGS")) as f2:
    jargs2 = json.loads(f2.readlines()[-1])
    samp_len = int(jargs2["samp_len"])

import numpy as np
m = model.tdnn(len(vocab))
_ = m((np.zeros((1, samp_len), dtype=np.float32), np.zeros((1, 1), dtype=np.int32)))

import tensorflow as tf
ckpt = tf.train.Checkpoint(m)
ckpt.read(args.ckpt)

import warnings
import soundfile
import tqdm

evals = [e.strip() for e in open(args.eval_list, "r").readlines()]

pcount = 0
for idx, _eval in tqdm.tqdm(enumerate(evals), total=len(evals)):
  if len(_eval.split()) != 2:
    warnings.warn("failed to parse {} at line {}".format(_eval, idx))
    continue

  spk, wav = _eval.split()
  if spk not in vocab:
    warnings.warn("vocab {} missing {}".format(args.vocab, spk))
    continue

  pcm, sr = soundfile.read(wav)
  out = m((np.expand_dims(pcm, 0), None))
  pcount += int(np.argmax(out) == vocab[spk])

print("overall pass {}/{}({:.3f}%)".format(
  pcount, len(evals), float(pcount)/(len(evals))*100))
