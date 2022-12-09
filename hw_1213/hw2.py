import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--batch-size", type=int, required=False, default=32)
parser.add_argument("--epoch", type=int, required=False, default=20)
parser.add_argument("--lr", type=float, required=False, default=4e-4)
parser.add_argument("--teacher", type=str, required=False, default="hw1_mix_warm3_b16_lr5e-5/model.ckpt")
parser.add_argument("--alpha", type=float, required=False, default=0.9)
parser.add_argument("--temperature", type=float, required=False, default=1)
parser.add_argument("--mix-up", action="store_true")
parser.add_argument("--mix-alpha", type=float, required=False, default=0.2)
parser.add_argument("--suffix", type=str, required=False, default="")
args = parser.parse_args()

import tensorflow as tf
import keras
import distiller

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

from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt

(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

# Normalize pixel values to be between 0 and 1
train_images, test_images = train_images.astype("float32") / 255.0, test_images.astype("float32") / 255.0

train_data = tf.data.Dataset.from_tensor_slices((train_images, train_labels))
test_data = tf.data.Dataset.from_tensor_slices((test_images, test_labels))

if args.mix_up:
  import mixup
  train_data, test_data = mixup.mix_dataset(
          train_data, test_data, args.batch_size, args.mix_alpha)

else:
  train_data = (
    train_data
    .shuffle(args.batch_size * 100)
    .batch(args.batch_size)
  )

test_data = (
  test_data
  .batch(args.batch_size)
)

s_model = models.Sequential()
s_model.add(layers.ZeroPadding2D((1, 1), input_shape=(32, 32, 3)))
s_model.add(layers.Conv2D(32, (5, 5), activation='relu'))
s_model.add(layers.MaxPooling2D((2, 2)))#, strides=(1, 1)))
s_model.add(layers.ZeroPadding2D((1, 1)))
s_model.add(layers.Conv2D(64, (3, 3), activation='relu'))
s_model.add(layers.MaxPooling2D((2, 2)))#, strides=(1, 1)))
s_model.add(layers.ZeroPadding2D((1, 1)))
s_model.add(layers.Conv2D(128, (3, 3), activation='relu'))
s_model.add(layers.MaxPooling2D((2, 2)))#, strides=(1, 1)))

s_model.add(layers.Flatten())
s_model.add(layers.Dense(500, activation='relu'))
s_model.add(layers.Dense(10))

print(s_model.summary())

from model import resnet18

_, t_model, preprocess_input = resnet18()
t_model.load_weights(args.teacher).expect_partial()

def preprocess_cifar10(img):
  img = tf.image.resize(img * 255., [224, 224])
  img = preprocess_input(img)
  return img

model = distiller.Distiller(student=s_model, teacher=t_model,
  teacher_hook=preprocess_cifar10)

def get_lr_metric(optimizer):
    def lr(y_true, y_pred):
        return optimizer.lr#_decayed_lr(tf.float32)
    return lr

lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    args.lr,
    decay_steps=50000//args.batch_size,
    decay_rate=0.96,
    staircase=False)

opt = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
lr_metric = get_lr_metric(opt)
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
if args.mix_up:
  loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=True)

model.compile(optimizer=opt,
              student_loss_fn=loss_fn,
              distillation_loss_fn=tf.keras.losses.KLDivergence(),
              metrics=['accuracy', lr_metric],
              alpha=args.alpha,
              temperature=args.temperature)

history = model.fit(train_data, epochs=args.epoch, 
                    validation_data=test_data,
                    batch_size=args.batch_size)

import os
name = 'hw2'
if args.mix_up:
  name = "{}_mix".format(name)
name = "{}_b{}".format(name, args.batch_size)
name = '{}_{}_{}'.format(name, args.alpha, args.temperature)
if args.suffix != "":
  name = "{}_{}".format(name, args.suffix)

os.makedirs(name, exist_ok=True)
model.save_weights('{}/model.ckpt'.format(name))

train_res = model.evaluate(train_data, verbose=2)
train_acc, train_lr, train_loss = train_res
print(train_res)

test_res = model.evaluate(test_data, verbose=2)
test_acc, test_lr, test_loss = test_res
print(test_res)

plt.figure()
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label = 'test_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.title('{} {}'.format(train_acc, test_acc))
plt.savefig('{}_accuracy.png'.format(name))
plt.clf()

plt.figure()
plt.plot(history.history['student_loss'], label='loss(student)')
plt.plot(history.history['distillation_loss'], label='loss(teacher)')
plt.plot(history.history['loss'], label='loss(student+teacher)')
plt.plot(history.history['val_student_loss'], label = 'test_loss(student)')
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
