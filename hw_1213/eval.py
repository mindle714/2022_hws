import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--batch-size", type=int, required=False, default=32)
parser.add_argument("--ckpt", type=str, required=False, default="hw1_mix_warm3_b16_lr5e-5/model.ckpt")
args = parser.parse_args()

from model import resnet18
import tensorflow as tf

_, model, preprocess_input = resnet18()
model.load_weights(args.ckpt).expect_partial()

from tensorflow.keras import datasets
(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

def preprocess_cifar10(img, label):
  img = tf.image.resize(img, [224, 224])
  img = preprocess_input(img)
  return img, label

train_data = (
  tf.data.Dataset.from_tensor_slices((train_images, train_labels))
  .batch(args.batch_size)
  .map(
    preprocess_cifar10,
    num_parallel_calls=tf.data.AUTOTUNE
  )
)

test_data = (
  tf.data.Dataset.from_tensor_slices((test_images, test_labels))
  .batch(args.batch_size)
  .map(
    preprocess_cifar10,
    num_parallel_calls=tf.data.AUTOTUNE
  )
)
	
model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        	  metrics=['accuracy'])

test_res = model.predict(test_data, verbose=2)
print(test_res)

import matplotlib.pyplot as plt
plt.figure(figsize=(10,10))
plt.imshow(test_images[0])
plt.savefig('test_images0.png')

print(test_res[0])
import numpy as np
np.set_printoptions(suppress=True, precision=5, linewidth=70)
print(tf.nn.softmax(test_res[0], -1))
print(tf.nn.softmax(test_res[0]/2., -1))
print(tf.one_hot(test_labels[0], 10))

import matplotlib.pyplot as plt
plt.figure(figsize=(10,10))
plt.imshow(test_images[1])
plt.savefig('test_images1.png')

print(test_res[1])
import numpy as np
np.set_printoptions(suppress=True)
print(tf.nn.softmax(test_res[1], -1))
print(tf.one_hot(test_labels[1], 10))

#train_res = model.evaluate(train_data, verbose=2)
#print(train_res)

