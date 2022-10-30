import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--batch-size", type=int, required=False, default=32)
parser.add_argument("--epoch", type=int, required=False, default=10)
parser.add_argument("--lr", type=float, required=False, default=1e-3)
parser.add_argument("--mix-up", action="store_true")
parser.add_argument("--alpha", type=float, required=False, default=0.2)
args = parser.parse_args()

import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt

def sample_beta_distribution(size, concentration_0=0.2, concentration_1=0.2):
    gamma_1_sample = tf.random.gamma(shape=[size], alpha=concentration_1)
    gamma_2_sample = tf.random.gamma(shape=[size], alpha=concentration_0)
    return gamma_1_sample / (gamma_1_sample + gamma_2_sample)

def mix_up(ds_one, ds_two, alpha=0.2):
    # Unpack two datasets
    images_one, labels_one = ds_one
    images_two, labels_two = ds_two
    batch_size = tf.shape(images_one)[0]

    # Sample lambda and reshape it to do the mixup
    l = sample_beta_distribution(batch_size, alpha, alpha)
    x_l = tf.reshape(l, (batch_size, 1, 1, 1))
    y_l = tf.reshape(l, (batch_size, 1))

    # Perform mixup on both images and labels by combining a pair of images/labels
    # (one from each dataset) into one image/label
    images = images_one * x_l + images_two * (1 - x_l)
    labels = labels_one * y_l + labels_two * (1 - y_l)
    return (images, labels)

(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

# Normalize pixel values to be between 0 and 1
train_images, test_images = train_images.astype("float32") / 255.0, test_images.astype("float32") / 255.0

if args.mix_up:
    train_labels = tf.squeeze(tf.one_hot(train_labels, 10), 1)
    test_labels = tf.squeeze(tf.one_hot(test_labels, 10), 1)

    train_pairs_1 = (
        tf.data.Dataset.from_tensor_slices((train_images, train_labels))
        .shuffle(args.batch_size * 100)
        .batch(args.batch_size)
    )

    train_pairs_2 = (
        tf.data.Dataset.from_tensor_slices((train_images, train_labels))
        .shuffle(args.batch_size * 100)
        .batch(args.batch_size)
    )

    train_pairs = tf.data.Dataset.zip((train_pairs_1, train_pairs_2))
    train_pairs = train_pairs.map(
        lambda pair_1, pair_2: mix_up(pair_1, pair_2, alpha=args.alpha),
        num_parallel_calls=tf.data.AUTOTUNE
    )

    test_pairs = tf.data.Dataset.from_tensor_slices((test_images, test_labels)).batch(args.batch_size)

model = models.Sequential()
model.add(layers.ZeroPadding2D((1, 1), input_shape=(32, 32, 3)))
model.add(layers.Conv2D(32, (5, 5), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.ZeroPadding2D((1, 1)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.ZeroPadding2D((1, 1)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Flatten())
model.add(layers.Dense(500, activation='relu'))
model.add(layers.Dense(10))

print(model.summary())

def get_lr_metric(optimizer):
    def lr(y_true, y_pred):
        return optimizer._decayed_lr(tf.float32)
    return lr

lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    args.lr,
    decay_steps=50000//args.batch_size,
    decay_rate=0.96,
    staircase=False)

opt = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
lr_metric = get_lr_metric(opt)

if args.mix_up:
	model.compile(optimizer=opt,
    	          loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
        	      metrics=['accuracy', lr_metric])

	history = model.fit(train_pairs, epochs=args.epoch, 
                        validation_data=test_pairs)
else:
	model.compile(optimizer=opt,
    	          loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        	      metrics=['accuracy', lr_metric])

	history = model.fit(train_images, train_labels, epochs=args.epoch, 
                        validation_data=(test_images, test_labels),
                        batch_size=args.batch_size)

name = "hw1"
if args.mix_up:
    name = "{}_mix".format(name)

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

train_res = model.evaluate(train_images, train_labels, verbose=2)
print(train_res)

test_res = model.evaluate(test_images, test_labels, verbose=2)
print(test_res)
