import tensorflow as tf
import glob
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

tfrecs = glob.glob(os.path.join("si84", "train-*.tfrecord"))
data = tf.data.TFRecordDataset(tfrecs)

for rec in data.take(7):
  ex = tf.train.Example()
  ex.ParseFromString(rec.numpy())
  print(len(ex.features.feature['pcm'].float_list.value))

