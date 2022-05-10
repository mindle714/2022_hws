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
