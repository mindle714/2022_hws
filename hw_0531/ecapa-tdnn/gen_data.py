import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--train-list", type=str, required=True) 
parser.add_argument("--noise-list", type=str, required=False, default=None)
parser.add_argument("--min-snr", type=int, required=False, default=10)
parser.add_argument("--max-snr", type=int, required=False, default=10)
parser.add_argument("--vocab", type=str, required=True)
parser.add_argument("--num-chunks", type=int, required=False, default=100)
parser.add_argument("--samp-len", type=int, required=False, default=8192)
parser.add_argument("--no-remainder", action='store_true')
parser.add_argument("--output", type=str, required=True) 
parser.add_argument("--apply-jointb", action='store_true') 
parser.add_argument("--mixup-multiplier", type=int, required=False, default=1) 
parser.add_argument("--apply-cutmix", action='store_true') 
args = parser.parse_args()

if args.apply_jointb:
  import jointbilatFil
  import librosa

  def magnitude(e): return np.abs(e)
  def phase(e): return np.arctan2(e.imag, e.real)
  def polar(mag, phase):
    return mag * (np.cos(phase) + np.sin(phase) * 1j)

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

train_list = [e.strip() for e in open(args.train_list).readlines()]
vocab = {e.strip():idx for idx, e in enumerate(open(args.vocab).readlines())}

import random
random.shuffle(train_list)

if args.noise_list is not None:
  noise_list = [e.strip() for e in open(args.noise_list).readlines()]
  assert len(train_list) == len(noise_list)

import warnings
import tensorflow as tf
import multiprocessing
import numpy as np
import copy
import soundfile
import tqdm

def add_noise(pcm, noise, snr_db):
  ns_pcm = copy.deepcopy(pcm)

  if args.apply_jointb:
    f_orig = librosa.stft(ns_pcm)
    m_orig = magnitude(f_orig)
    m_orig = librosa.amplitude_to_db(m_orig)

  pcm_en = np.mean(ns_pcm**2)
  noise_en = np.maximum(np.mean(noise**2), 1e-9)
  snr_en = 10.**(snr_db/10.)

  noise *= np.sqrt(pcm_en / (snr_en * noise_en))
  ns_pcm += noise
  noise_pcm_en = np.maximum(np.mean(ns_pcm**2), 1e-9)
  ns_pcm *= np.sqrt(pcm_en / noise_pcm_en)

  if args.apply_jointb:
    f_ns = librosa.stft(ns_pcm)
    m_ns = magnitude(f_ns); ph_ns = phase(f_ns)
    m_ns = librosa.amplitude_to_db(m_ns)

    m_ns_new = jointbilatFil.jointBilateralFilter(
      np.expand_dims(m_ns, -1), np.expand_dims(m_orig, -1))
    m_ns_new = np.squeeze(m_ns_new, -1)
    m_ns_new = librosa.db_to_amplitude(m_ns_new)

    ns_pcm = librosa.istft(polar(m_ns_new, ph_ns), length=ns_pcm.shape[0])

  return ns_pcm

def get_feat(_pcm, _spk, _samp_len, noise, snr_db): 
  _ref = copy.deepcopy(_pcm)
  if noise is not None:
    _pcm = add_noise(_pcm, noise, snr_db)

  '''
  pcm_feat = tf.train.Feature(float_list=tf.train.FloatList(value=_pcm))
  spk_feat = tf.train.Feature(int64_list=tf.train.Int64List(value=[vocab[_spk]]))

  feats = {'pcm': pcm_feat, 'speaker': spk_feat}
  ex = tf.train.Example(features=tf.train.Features(feature=feats))
  return ex.SerializeToString()
  '''
  return {'pcm': _pcm, 'speaker': [vocab[_spk]]}

def get_feats(pcm, _spk, noise, snr_db): 
  exs = []
  num_seg = max((pcm.shape[0] - samp_len) // hop_len + 1, 0)

  for pcm_idx in range(num_seg):
    _pcm = pcm[pcm_idx*hop_len: pcm_idx*hop_len+samp_len]
    _noise = None
    if noise is not None:
      _noise = noise[pcm_idx*hop_len: pcm_idx*hop_len+samp_len]

    ex = get_feat(_pcm, _spk, samp_len, _noise, snr_db)
    exs.append(ex)

  rem_len = pcm[num_seg*hop_len:].shape[0]
  if (not args.no_remainder) and rem_len > 0:
    def pad(_in):
      return np.concatenate([_in,
        np.zeros(samp_len-_in.shape[0], dtype=_in.dtype)], 0)

    _pcm = pad(pcm[num_seg*hop_len:])
    _noise = None
    if noise is not None:
      _noise = pad(noise[num_seg*hop_len:])

    ex = get_feat(_pcm, _spk, rem_len, _noise, snr_db)
    exs.append(ex)

  return exs

num_chunks = min(len(train_list), args.num_chunks)
writers = [tf.io.TFRecordWriter(os.path.join(
    args.output, "train-{}.tfrecord".format(idx))) for idx in range(num_chunks)]

chunk_idx = 0; chunk_lens = [0 for _ in range(num_chunks)]
hop_len = args.samp_len//2; samp_len = args.samp_len
num_process = 8

for bidx in tqdm.tqdm(range(len(train_list)//num_process+1)):
  blist = train_list[bidx*num_process:(bidx+1)*num_process]
  if len(blist) == 0: break

  blist_spk = [e.split()[0] for e in blist]
  blist_pcm = [e.split()[1] for e in blist]

  pcms = [soundfile.read(e)[0] for e in blist_pcm]
  spks = blist_spk
  snr_dbs = np.random.uniform(args.min_snr, args.max_snr, len(blist))
  
  noises = [None for _ in range(len(blist))]
  if args.noise_list is not None:
    nlist = noise_list[bidx*num_process:(bidx+1)*num_process]
    noises = [soundfile.read(e)[0] for e in nlist]
  
    for nidx in range(len(noises)):
      pcm = pcms[nidx]; noise = noises[nidx]
      if pcm.shape[0] >= noise.shape[0]:
        noise = np.repeat(noise, (pcm.shape[0]//noise.shape[0]+1))
        noise = noise[:pcm.shape[0]]
      else:
        pos = np.random.randint(0, noise.shape[0]-pcm.shape[0]+1)
        noise = noise[pos:pos+pcm.shape[0]]
      noises[nidx] = noise

  with multiprocessing.Pool(num_process) as pool:
    exs = pool.starmap(get_feats, zip(pcms, spks, noises, snr_dbs))
  
  exs_flat = []
  while True:
    ignored = 0
    for idx in range(len(exs)):
      ex = exs[idx]
      if len(ex) == 0:
        ignored += 1
        continue
      exs_flat.append(ex.pop(0))
    if ignored == len(exs): break

  for idx in range(len(exs_flat)//args.mixup_multiplier):
    _exs = exs_flat[idx*args.mixup_multiplier:(idx+1)*args.mixup_multiplier]
    _exs_spk = sum([e['speaker'] for e in _exs], [])

    if len(_exs_spk) == len(set(_exs_spk)):
      mixup_weights = np.random.uniform(size=args.mixup_multiplier)
      mixup_weights /= np.sum(mixup_weights)

      if not args.apply_cutmix:
        _pcm = sum([e['pcm'] * w for e, w in zip(_exs, mixup_weights)])

      else:
        _pcms = []; pcm_pos = 0
        for idx, (e, w) in enumerate(zip(_exs, mixup_weights)):
          _len = int(args.samp_len * w)
          if idx == (args.mixup_multiplier - 1):
            _len = args.samp_len - pcm_pos

          epcm = e['pcm'][pcm_pos:pcm_pos+_len]
          _pcms.append(epcm)
          pcm_pos += _len

        _pcm = np.concatenate(_pcms, 0)

      pcm_feat = tf.train.Feature(float_list=tf.train.FloatList(value=_pcm))
      spks_feat = tf.train.Feature(int64_list=tf.train.Int64List(value=_exs_spk))
      mixup_feat = tf.train.Feature(float_list=tf.train.FloatList(value=mixup_weights))

      feats = {'pcm': pcm_feat, 'speakers': spks_feat, 'mixup_weights': mixup_feat}
      ex = tf.train.Example(features=tf.train.Features(feature=feats))
      _ex = ex.SerializeToString()

      writers[chunk_idx].write(_ex)
      chunk_lens[chunk_idx] += 1
      chunk_idx = (chunk_idx+1) % num_chunks

for writer in writers:
  writer.close()

args.chunk_lens = chunk_lens

with open(args_file, "w") as f:
  f.write(" ".join([sys.executable] + sys.argv))
  f.write("\n")
  f.write(json.dumps(vars(args)))
os.chmod(args_file, S_IREAD|S_IRGRP|S_IROTH)

import shutil
for origin in [os.path.abspath(__file__), args.train_list]:
  shutil.copy(origin, args.output)
