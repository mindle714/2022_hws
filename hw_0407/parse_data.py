import tensorflow as tf

def _parse_function(ex):
  desc = {
    'pcm': tf.io.FixedLenFeature([8192], tf.float32),
    'speaker': tf.io.FixedLenFeature([1], tf.int64)
  }
  return tf.io.parse_single_example(ex, desc)

def gen_train(tfrec_list, batch_size=16):
  dataset = tf.data.TFRecordDataset(tfrec_list)
  dataset = dataset.map(_parse_function)
  dataset = dataset.batch(batch_size)
  dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
  return dataset
