import tensorflow as tf

def parse_func_v1(pcm_len, mix_len):
  def _parse_func_v1(ex):
    desc = {
      'pcm': tf.io.FixedLenFeature([pcm_len], tf.float32),
      'speaker': tf.io.FixedLenFeature([1], tf.int64)
    }
    return tf.io.parse_single_example(ex, desc)
  return _parse_func_v1

def hot_ref_v1(vocab):
  def _hot_ref_v1(ex):
    ref = ex['speaker']
    del ex['speaker']

    ref = tf.squeeze(ref, -1)
    hot_ref = tf.one_hot(ref, vocab)
    ex['target'] = hot_ref

    return ex
  return _hot_ref_v1

def gen_train_v1(tfrec_list, pcm_len, mix_len, vocab, batch_size=16, seed=1234):
  dataset = tf.data.TFRecordDataset(tfrec_list)
  dataset = dataset.shuffle(batch_size, seed=seed, reshuffle_each_iteration=True)
  dataset = dataset.repeat()

  dataset = dataset.map(parse_func_v1(pcm_len, mix_len))
  dataset = dataset.map(hot_ref_v1(vocab))
  dataset = dataset.batch(batch_size)
  dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
  return dataset

def parse_func_v2(pcm_len, mix_len):
  def _parse_func_v2(ex):
    desc = {
      'pcm': tf.io.FixedLenFeature([pcm_len], tf.float32),
      'speakers': tf.io.FixedLenFeature([mix_len], tf.int64),
      'mixup_weights': tf.io.FixedLenFeature([mix_len], tf.float32)
    }
    return tf.io.parse_single_example(ex, desc)
  return _parse_func_v2

def hot_ref_v2(vocab):
  def _hot_ref_v2(ex):
    refs = ex['speakers']
    del ex['speakers']

    weights = ex['mixup_weights']
    del ex['mixup_weights']

    hot_refs = tf.one_hot(refs, vocab) * tf.expand_dims(weights, -1)
    hot_refs = tf.math.reduce_sum(hot_refs, -2) 
    ex['target'] = hot_refs

    return ex
  return _hot_ref_v2

def gen_train_v2(tfrec_list, pcm_len, mix_len, vocab, batch_size=16, seed=1234):
  dataset = tf.data.TFRecordDataset(tfrec_list)
  dataset = dataset.shuffle(batch_size, seed=seed, reshuffle_each_iteration=True)
  dataset = dataset.repeat()

  dataset = dataset.map(parse_func_v2(pcm_len, mix_len))
  dataset = dataset.map(hot_ref_v2(vocab))
  dataset = dataset.batch(batch_size)
  dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
  return dataset
