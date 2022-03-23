import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--tfrec", type=str, required=True) 
parser.add_argument("--vocab", type=str, required=True) 
parser.add_argument("--samp-len", type=int, required=False, default=8192)
parser.add_argument("--output", type=str, required=True) 
args = parser.parse_args()

import parse_data
import types
import sys
import os
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

origins = [val.__spec__.origin for name, val in globals().items() \
  if isinstance(val, types.ModuleType)]
origins = [e for e in origins if e is not None and os.getcwd() in e]

import shutil
for origin in origins:
  shutil.copy(origin, args.output)

import glob
tfrec_list = glob.glob(os.path.join(args.tfrec, "train-*.tfrecord"))
dataset = parse_data.gen_train(tfrec_list)

import functools
import numpy as np
import tensorflow as tf

def mel_filterbank(
    waveforms,
    sample_rate=16000,
    frame_length=25, frame_step=10, fft_length=None,
    window_fn=functools.partial(tf.signal.hann_window, periodic=True),
    lower_edge_hertz=80.0, upper_edge_hertz=7600.0, num_mel_bins=80,
    log_noise_floor=1e-3, apply_mask=True):

  wav_lens = tf.reduce_max(
      tf.expand_dims(tf.range(tf.shape(waveforms)[1]), 0) *
      tf.cast(tf.not_equal(waveforms, 0.0), tf.int32),
      axis=-1) + 1
  frame_length = int(frame_length * sample_rate / 1e3)
  frame_step = int(frame_step * sample_rate / 1e3)
  if fft_length is None:
    fft_length = int(2**(np.ceil(np.log2(frame_length))))

  stfts = tf.signal.stft(
      waveforms,
      frame_length=frame_length,
      frame_step=frame_step,
      fft_length=fft_length,
      window_fn=window_fn,
      pad_end=True)

  stft_lens = (wav_lens + (frame_step - 1)) // frame_step
  masks = tf.cast(tf.less_equal(
      tf.expand_dims(tf.range(tf.shape(stfts)[1]), 0),
      tf.expand_dims(stft_lens, 1)), tf.float32)

  # An energy spectrogram is the magnitude of the complex-valued STFT.
  # A float32 Tensor of shape [batch_size, ?, 257].
  magnitude_spectrograms = tf.abs(stfts)

  # Warp the linear-scale, magnitude spectrograms into the mel-scale.
  num_spectrogram_bins = magnitude_spectrograms.shape[-1]
  linear_to_mel_weight_matrix = (
      tf.signal.linear_to_mel_weight_matrix(
          num_mel_bins, num_spectrogram_bins, sample_rate, lower_edge_hertz,
          upper_edge_hertz))
  mel_spectrograms = tf.tensordot(
      magnitude_spectrograms, linear_to_mel_weight_matrix, 1)
  # Note: Shape inference for tensordot does not currently handle this case.
  mel_spectrograms.set_shape(magnitude_spectrograms.shape[:-1].concatenate(
      linear_to_mel_weight_matrix.shape[-1:]))

  log_mel_sgram = tf.math.log(tf.maximum(log_noise_floor, mel_spectrograms))

  if apply_mask:
    log_mel_sgram *= tf.expand_dims(tf.cast(masks, tf.float32), -1)

  return log_mel_sgram

for data in dataset:
  mel = mel_filterbank(data["pcm"])
  print(mel)
