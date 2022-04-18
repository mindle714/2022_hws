import tensorflow as tf
import functools
import numpy as np
from util import *

tf_sum = tf.math.reduce_sum
tf_expd = tf.expand_dims

class birnn(tf.keras.layers.Layer):
  def __init__(self, vocab, *args, **kwargs):
    super(birnn, self).__init__(*args, **kwargs)

    self.vocab = vocab
    self.sr = 16000
    self.frame_length = 16
    self.frame_step = 8
    self.dim = 500
    self.layer = 5

  def build(self, input_shape):
    self.lstms = [tf.keras.layers.Bidirectional(
      tf.keras.layers.LSTM(self.dim, return_sequences=True)
    ) for _ in range(self.layer)]

    self.post = tf.keras.layers.Dense(self.vocab)

  def call(self, inputs, training=None):
    pcm, pcm_len, ref, ref_len = inputs

    x = spectrogram(pcm, frame_length=self.frame_length,
      frame_step=self.frame_step)

    for lstm in self.lstms:
      x = lstm(x)

    x = self.post(x)

    if ref is not None:
      frame_length = int(self.frame_length * self.sr / 1e3)
      frame_step = int(self.frame_step * self.sr / 1e3)
    
      pcm_len = tf.squeeze(pcm_len, -1)
      ref_len = tf.squeeze(ref_len, -1)

      x_len = (pcm_len - frame_length) // frame_step + 1
      x_len = tf.math.maximum(x_len, 0)

      loss = tf.nn.ctc_loss(ref, x, ref_len, x_len, logits_time_major=False)
      return loss

    return x
