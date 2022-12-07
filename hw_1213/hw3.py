import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--batch-size", type=int, required=False, default=32)
parser.add_argument("--warm-epoch", type=int, required=False, default=0)
parser.add_argument("--epoch", type=int, required=False, default=20)
parser.add_argument("--warm-lr", type=float, required=False, default=1e-3)
parser.add_argument("--lr", type=float, required=False, default=1e-5)
parser.add_argument("--ckpt", type=str, required=False, default="hw1_mix_warm3/model.ckpt")
parser.add_argument("--mix-up", action="store_true")
parser.add_argument("--mix-alpha", type=float, required=False, default=0.2)
parser.add_argument("--train-type", type=str, required=False,
  default="fine-tune", choices=["fine-tune", "lwf"])
args = parser.parse_args()

import matplotlib.pyplot as plt
from model import resnet18
import tensorflow as tf

gpus = tf.config.list_physical_devices('GPU')
if gpus:
  try:
    # Currently, memory growth needs to be the same across GPUs
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Memory growth must be set before GPUs have been initialized
    print(e)

up_model, old_model, preprocess_input = resnet18()
old_model.load_weights(args.ckpt).expect_partial()
x = tf.keras.layers.GlobalAveragePooling2D()(up_model.output)
output = tf.keras.layers.Dense(100)(x)
model = tf.keras.models.Model(inputs=[up_model.input], outputs=[output])

from tensorflow.keras import datasets
_, (test_cifar10_images, test_cifar10_labels) = datasets.cifar10.load_data()
test_cifar10_images = test_cifar10_images.astype("float32")

test_cifar10_data = tf.data.Dataset.from_tensor_slices((test_cifar10_images, test_cifar10_labels))

def preprocess_cifar10(img, label):
  img = tf.image.resize(img, [224, 224])
  img = preprocess_input(img)
  return img, label

test_cifar10_data = (
  test_cifar10_data
  .batch(args.batch_size)
  .map(
    preprocess_cifar10,
    num_parallel_calls=tf.data.AUTOTUNE
  )
)

import numpy as np
import glob
from matplotlib.image import imread

train_dir = "facescrub_train"
test_dir = "facescrub_test"

def load_data(_dir):
  images = []; labels = []
  for idx, face in enumerate(glob.glob("{}/*/".format(_dir))):
    for sample in glob.glob("{}/*.jpg".format(face)):
      img = imread(sample)
      images.append(img); labels.append([idx])

  images = np.array(images)
  labels = np.array(labels)
  return images, labels

(train_images, train_labels), (test_images, test_labels) = load_data(train_dir), load_data(test_dir)
train_images, test_images = train_images.astype("float32"), test_images.astype("float32")

train_data = tf.data.Dataset.from_tensor_slices((train_images, train_labels))
test_data = tf.data.Dataset.from_tensor_slices((test_images, test_labels))

def preprocess_facescrub(img, label):
  img = tf.image.resize(img, [224, 224])
  img = preprocess_input(img)
  return img, label

if args.mix_up:
  import mixup
  train_data, test_data = mixup.mix_dataset(
          train_data, test_data, args.batch_size, args.mix_alpha,
          num_class=100)

else:
  train_data = (
    train_data
    .shuffle(args.batch_size * 100)
    .batch(args.batch_size)
  )

train_data = (
  train_data
  .map(
    preprocess_facescrub,
    num_parallel_calls=tf.data.AUTOTUNE
  )
)

test_data = (
  test_data
  .batch(args.batch_size)
  .map(
    preprocess_facescrub,
    num_parallel_calls=tf.data.AUTOTUNE
  )
)

def compile(model, lr):
  def get_lr_metric(optimizer):
    def lr(y_true, y_pred):
      return optimizer.lr#_decayed_lr(tf.float32)
    return lr

  lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    lr, decay_steps=50000//args.batch_size, decay_rate=0.96, staircase=False)

  opt = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
  lr_metric = get_lr_metric(opt)

  if args.mix_up:
    model.compile(optimizer=opt,
                  loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy', lr_metric])

  else:
    model.compile(optimizer=opt,
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy', lr_metric])

old_model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

if args.warm_epoch > 0:
  up_model.trainable = False
  compile(model, args.warm_lr)
  history = model.fit(train_data, epochs=args.warm_epoch, 
                      validation_data=test_data)
  
if args.epoch > 0:
  up_model.trainable = True
  compile(model, args.lr)
  history = model.fit(train_data, epochs=args.epoch, 
                      validation_data=test_data)

import os
name = 'hw3'
if args.mix_up:
  name = "{}_mix".format(name)
if args.warm_epoch > 0:
  name = "{}_warm{}".format(name, args.warm_epoch)
if args.train_type == "lwf":
  name = "{}_lwf".format(name)

os.makedirs(name, exist_ok=True)
model.save_weights('{}/model.ckpt'.format(name))

train_res = model.evaluate(train_data, verbose=2)
train_acc, train_lr, train_loss = train_res
print(train_res)

test_face_res = model.evaluate(test_data, verbose=2)
test_face_acc, test_lr, test_loss = test_face_res
print(test_face_res)
test_cifar10_res = old_model.evaluate(test_cifar10_data, verbose=2)
test_cifar10_acc, _ = test_cifar10_res
print(test_cifar10_res)

plt.figure()
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label = 'test_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.title('{} {} {}'.format(train_acc, test_face_acc, test_cifar10_acc))
plt.savefig('{}_accuracy.png'.format(name))
plt.clf()

plt.figure()
plt.plot(history.history['loss'], label='loss')
plt.plot(history.history['val_loss'], label = 'test_loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(loc='lower right')
plt.title('{} {}'.format(train_loss, test_loss))
plt.savefig('{}_loss.png'.format(name))
plt.clf()

plt.figure()
plt.plot(history.history['lr'], label='lr')
plt.xlabel('Epoch')
plt.ylabel('Lr')
plt.legend(loc='lower right')
plt.savefig('{}_lr.png'.format(name))