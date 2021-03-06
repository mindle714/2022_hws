import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--ckpt", type=str, required=False,
  default="exps/base_cont/model-16000.ckpt") 
parser.add_argument("--wav-list", type=str, required=False,
  default="/home/speech/wsj1/dev93.list") 
parser.add_argument("--save-result", action="store_true") 
parser.add_argument("--beam-size", type=int, required=False, default=1)
parser.add_argument("--arpa", type=str, required=False, default=None)
parser.add_argument("--lm-weight", type=int, required=False, default=0.1)
args = parser.parse_args()

if args.arpa is not None:
  import arpa
  import trie

  lm = arpa.loadf(args.arpa)[0]
  prefix_trie = trie.trie(lm.vocabulary())

import os
import sys

expdir = os.path.abspath(os.path.dirname(args.ckpt))
sys.path.insert(0, expdir)
expname = expdir.split("/")[-1]
epoch = os.path.basename(args.ckpt).replace(".", "-").split("-")[1]

import model
if os.path.dirname(model.__file__) != expdir:
  sys.exit("model is loaded from {}".format(model.__file__))

import json
exp_args = os.path.join(expdir, "ARGS")
with open(exp_args, "r") as f:
  jargs = json.loads(f.readlines()[-1])

  tfrec_path = jargs["tfrec"]
  with open(os.path.join(tfrec_path, "ARGS")) as f2:
    jargs2 = json.loads(f2.readlines()[-1])
    samp_len = int(jargs2["samp_len"])
    vocab = {int(k[1]):k[0] for k in \
      [e.split("\t") for e in open(os.path.join(tfrec_path, "vocab")).readlines()]}

import numpy as np
m = model.birnn(len(vocab))
_in = np.zeros((1, samp_len), dtype=np.float32)
_ = m((_in, np.zeros((1,1)), None, np.zeros((1,1))), training=False)

import tensorflow as tf
ckpt = tf.train.Checkpoint(model=m)
ckpt.read(args.ckpt)

import warnings
import soundfile
import tqdm
import time

wav_list = [e.strip() for e in open(args.wav_list, "r").readlines()]

outfile = "{}-{}.eval".format(expname, epoch)
if args.beam_size > 0:
  outfile = "{}-{}-b{}.eval".format(expname, epoch, args.beam_size)
  if args.arpa is not None:
    outfile = "{}-{}-b{}-lm.eval".format(expname, epoch, args.beam_size)

with open(outfile, "w") as f:
  pcount = 0; snr_tot = 0

  for idx, _wav in enumerate(wav_list):
    pcm, _ = soundfile.read(_wav)
    pcm = np.expand_dims(pcm, 0).astype(np.float32)

    def pad(pcm, mod=8):
      if pcm.shape[-1] % mod != 0:
        pcm = np.concatenate([pcm, np.zeros((1, mod-pcm.shape[-1]%mod))], -1)
        return pcm
      return pcm

    hyp = m((pad(pcm), np.zeros((1,1)), None, np.zeros((1,1))), training=False)

    def greedy(hyp):
      truns = []; prev = 0
      for idx in hyp:
        if idx != prev:
          if prev != 0: truns.append(prev)
        prev = idx
      if prev != 0: truns.append(prev)
      return ''.join([vocab[e] for e in truns])
    
    if args.beam_size < 1:
      maxids = np.argmax(np.squeeze(hyp, 0), -1)
      f.write(greedy(maxids) + "\n")
      f.flush()

    else:
      def softmax(x):
        z = x - np.max(x, -1, keepdims=True)
        num = np.exp(z)
        denom = np.sum(num, -1, keepdims=True)
        return num / denom

      def _lm_prob(cidxs, max_ctx=3, max_prefs=10):
        if args.arpa is None: return 1.

        cwords = ''.join([vocab[e] for e in cidxs])
        cwords = cwords.split()
        if len(cwords) == 0: return 1.

        num_prefs = trie.expand_prefix(prefix_trie, cwords[-1], max_prefs)
        denom_prefs = trie.expand_prefix(prefix_trie, cwords[-1][:-1], max_prefs) \
          if len(cwords[-1]) > 1 else []

        num_cwords = [(' '.join(cwords[:-1]) + ' ' + e).strip() for e in num_prefs]
        num_cwords = [' '.join(e.split()[:max_ctx]) for e in num_prefs]
        num_cwords = [e for e in num_cwords if e != '']
        
        denom_cwords = [(' '.join(cwords[:-1]) + ' ' + e).strip() for e in denom_prefs]
        denom_cwords = [' '.join(e.split()[:max_ctx]) for e in denom_prefs]
        denom_cwords = [e for e in denom_cwords if e != '']

        num = sum([lm.p(e) for e in num_cwords]) if len(num_cwords) > 0 else 0.
        denom = sum([lm.p(e) for e in denom_cwords]) if len(denom_cwords) > 0 else 1.
        
        return (num / denom) ** args.lm_weight

      lm_elapsed = 0
      def lm_prob(cidxs):
        global lm_elapsed

        start = time.time()
        res = _lm_prob(cidxs)
        end = time.time()

        lm_elapsed += (end - start)
        return res

      logits = softmax(np.squeeze(hyp, 0))
      beams = [((1., 0.), [])]

      for t in range(logits.shape[0]):
        new_beams = []
        lm_elapsed = 0
        start = time.time()

        for bidx in range(len(beams)):
          (bprob, nbprob), y = beams[bidx]
          (c_bprob, c_nbprob), c_y = ((0., 0.), y)

          if len(y) > 0:
            ye = y[-1]
            c_nbprob = nbprob * logits[t][ye] # repeat last

            for e in beams:
              if y[:-1] == e[1]:
                # blank is needed to append "b" to "~b"
                prefix_prob = e[0][0] if (len(e[1]) > 0 and ye == e[1][-1]) else sum(e[0])
                c_nbprob += prefix_prob * logits[t][ye] * lm_prob(c_y) # expand from prefix
           
          c_bprob = (bprob + nbprob) * logits[t][0] # blank -> 0
          new_beams.append(((c_bprob, c_nbprob), c_y))

          for k in range(1, logits[t].shape[-1]):
            (c_bprob, c_nbprob), c_y = ((0., 0.), y + [k])
            prefix_prob = bprob if (len(y) > 0 and k == y[-1]) else bprob + nbprob 

            c_nbprob = prefix_prob * logits[t][k] * lm_prob(c_y) 
            new_beams.append(((c_bprob, c_nbprob), c_y))

        beams = sorted(new_beams, 
          key=lambda e: sum(e[0])/(((5+len(e[1]))**0.5)/(6**0.5)), reverse=True)
        beams = beams[:args.beam_size]
        end = time.time()

      f.write(''.join([vocab[e] for e in beams[0][1]]) + "\n")
      f.flush()
