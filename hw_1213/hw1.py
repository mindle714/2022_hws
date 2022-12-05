import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--batch-size", type=int, required=False, default=32)
parser.add_argument("--epoch", type=int, required=False, default=30)
parser.add_argument("--lr", type=float, required=False, default=1e-3)
parser.add_argument("--mix-up", action="store_true")
parser.add_argument("--alpha", type=float, required=False, default=0.2)
args = parser.parse_args()

import matplotlib.pyplot as plt

import keras
from classification_models.keras import Classifiers
# from classification_models.tfkeras import Classifiers

ResNet18, preprocess_input = Classifiers.get('resnet18')
model = ResNet18((224, 224, 3), weights='imagenet', include_top=False)
x = keras.layers.GlobalAveragePooling2D()(model.output)
output = keras.layers.Dense(10, activation='softmax')(x)
model = keras.models.Model(inputs=[model.input], outputs=[output])

print(model.summary())

import tensorflow as tf
from tensorflow.keras import datasets
(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

def preprocess_cifar10(img, label):
  img = tf.image.resize(img, [224, 224])
  img = preprocess_input(img)
  return img, label

train_data = (
  tf.data.Dataset.from_tensor_slices((train_images, train_labels))
  .shuffle(args.batch_size * 100)
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
	
model.compile(optimizer=opt,
    	      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        	  metrics=['accuracy', lr_metric])
history = model.fit(train_data, epochs=args.epoch, 
                    validation_data=test_data,
                    batch_size=args.batch_size)

name = "hw1"

plt.figure()
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label = 'test_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.savefig('{}_accuracy.png'.format(name))
plt.clf()

plt.figure()
plt.plot(history.history['loss'], label='loss')
plt.plot(history.history['val_loss'], label = 'test_loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(loc='lower right')
plt.savefig('{}_loss.png'.format(name))
plt.clf()

plt.figure()
plt.plot(history.history['lr'], label='lr')
plt.xlabel('Epoch')
plt.ylabel('Lr')
plt.legend(loc='lower right')
plt.savefig('{}_lr.png'.format(name))

train_res = model.evaluate(train_data, verbose=2)
print(train_res)

test_res = model.evaluate(test_data, verbose=2)
print(test_res)
