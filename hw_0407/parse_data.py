import tensorflow as tf

def parse_func(pcm_len):
  def _parse_func(ex):
    desc = {
      'pcm': tf.io.FixedLenFeature([pcm_len], tf.float32),
      'speaker': tf.io.FixedLenFeature([1], tf.int64)
    }
    return tf.io.parse_single_example(ex, desc)
  return _parse_func

def gen_train(tfrec_list, pcm_len, batch_size=16):
  dataset = tf.data.TFRecordDataset(tfrec_list)
  dataset = dataset.map(parse_func(pcm_len))
  dataset = dataset.batch(batch_size)
  dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
  return dataset
