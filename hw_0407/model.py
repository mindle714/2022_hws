import tensorflow as tf
import functools
import numpy as np

def mel_filterbank(
    pcm, sr=16000,
    frame_length=25, frame_step=10, fft_length=None,
    window_fn=functools.partial(tf.signal.hann_window, periodic=True),
    lower_edge_hertz=80.0, upper_edge_hertz=7600.0, num_mel_bins=24,
    log_noise_floor=1e-3):

  frame_length = int(frame_length * sr / 1e3)
  frame_step = int(frame_step * sr / 1e3)
  if fft_length is None:
    fft_length = int(2**(np.ceil(np.log2(frame_length))))

  stfts = tf.signal.stft(
      pcm,
      frame_length=frame_length,
      frame_step=frame_step,
      fft_length=fft_length,
      window_fn=window_fn,
      pad_end=True)

  # An energy spectrogram is the magnitude of the complex-valued STFT.
  # A float32 Tensor of shape [batch_size, ?, 257].
  magnitude_spectrograms = tf.abs(stfts)

  # Warp the linear-scale, magnitude spectrograms into the mel-scale.
  num_spectrogram_bins = magnitude_spectrograms.shape[-1]
  linear_to_mel_weight_matrix = (
      tf.signal.linear_to_mel_weight_matrix(
          num_mel_bins, num_spectrogram_bins, sr, lower_edge_hertz,
          upper_edge_hertz))
  mel_spectrograms = tf.tensordot(
      magnitude_spectrograms, linear_to_mel_weight_matrix, 1)
  # Note: Shape inference for tensordot does not currently handle this case.
  mel_spectrograms.set_shape(magnitude_spectrograms.shape[:-1].concatenate(
      linear_to_mel_weight_matrix.shape[-1:]))

  log_mel_sgram = tf.math.log(tf.maximum(log_noise_floor, mel_spectrograms))
  return log_mel_sgram

class tdnn(tf.keras.layers.Layer):
  def __init__(self, vocab, *args, **kwargs):
    super(tdnn, self).__init__(*args, **kwargs)
    self.vocab = vocab

  def build(self, input_shape):
    self.frames = [
      tf.keras.layers.Conv1D(512, 5),
      tf.keras.layers.Conv1D(512, 3, dilation_rate=2),
      tf.keras.layers.Conv1D(512, 3, dilation_rate=3),
      tf.keras.layers.Dense(512),
      tf.keras.layers.Dense(1500)
    ]
    self.frame_bns = [tf.keras.layers.BatchNormalization() \
      for _ in range(len(self.frames))]

    self.segments = [
      tf.keras.layers.Dense(512),
      tf.keras.layers.Dense(512)
    ]
    self.segment_bns = [tf.keras.layers.BatchNormalization() \
      for _ in range(len(self.segments))]

    self.softmax = tf.keras.layers.Dense(self.vocab)

  def call(self, inputs, training=None):
    pcm, ref = inputs
    x = mel_filterbank(pcm)

    if ref is not None:
      ref = tf.squeeze(ref, -1)

    # cmvn; TODO need to consider online cmvn
    # m, v = tf.nn.moments(x, axes=1, keepdims=True)
    # x = (x - m) / tf.math.sqrt(v + 1e-9)

    for frame, bn in zip(self.frames, self.frame_bns):
      x = bn(tf.nn.relu(frame(x)))

    m, v = tf.nn.moments(x, axes=1)
    x = tf.concat([m, tf.math.sqrt(v + 1e-9)], -1)

    for segment, bn in zip(self.segments, self.segment_bns):
      x = bn(tf.nn.relu(segment(x)))

    x = self.softmax(x)

    if ref is not None:
      loss = tf.nn.sparse_softmax_cross_entropy_with_logits(ref, x)
      loss = tf.math.reduce_mean(loss)
      return loss
    
    return x
