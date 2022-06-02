import tensorflow as tf
import glob
import soundfile
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

tfrecs = glob.glob(os.path.join("tfrec_mixup2", "train-*.tfrecord"))
data = tf.data.TFRecordDataset(tfrecs)

for idx, rec in enumerate(data.take(3)):
  ex = tf.train.Example()
  ex.ParseFromString(rec.numpy())
  pcm = ex.features.feature['pcm'].float_list.value
  speakers = ex.features.feature['speakers'].int64_list.value
  weights = ex.features.feature['mixup_weights'].float_list.value
  #soundfile.write("view_data_v2_{}.wav".format(idx), pcm, 16000)
  print(speakers)
  print(weights)

import parse_data
train = parse_data.gen_train(tfrecs, 8192, 4)

for idx, rec in enumerate(train.take(3)):
  ex = tf.train.Example()
  print(rec.keys())
  speakers = rec['speakers']
  weights = rec['mixup_weights']
  print(speakers)
  print(weights)

