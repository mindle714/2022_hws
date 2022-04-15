import tensorflow as tf
import functools
import numpy as np
from util import *

tf_sum = tf.math.reduce_sum
tf_expd = tf.expand_dims

class birnn(tf.keras.layers.Layer):
  def __init__(self, *args, **kwargs):
    super(birnn, self).__init__(*args, **kwargs)
    self.dim = 500
    self.layer = 5

  def build(self, input_shape):
    self.lstms = [tf.keras.layers.Bidirectional(
      tf.keras.layers.LSTM(self.dim, return_sequences=True)
    ) for _ in range(self.layer)]

  def call(self, inputs, training=None):
    pcm, ref = inputs
    x = mel_filterbank(pcm)

    for lstm in self.lstms:
      x = lstm(x)

    if ref is not None:
      loss = tf.nn.ctc_loss()

    return x
