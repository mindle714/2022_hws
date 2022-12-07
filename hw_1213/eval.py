import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--batch-size", type=int, required=False, default=32)
parser.add_argument("--ckpt", type=str, required=True)
args = parser.parse_args()

from model import resnet18
import tensorflow as tf

model, preprocess_input = resnet18()
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

train_res = model.evaluate(train_data, verbose=2)
print(train_res)

test_res = model.evaluate(test_data, verbose=2)
print(test_res)
