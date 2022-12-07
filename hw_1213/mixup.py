import tensorflow as tf

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

def mix_dataset(train_data, test_data,
                batch_size, mix_alpha, num_class=10):
  train_data = train_data.map(
    lambda _, label: (_, tf.squeeze(tf.one_hot(label, num_class), 0)))

  train_pairs_1 = (
    train_data
    .shuffle(batch_size * 100)
    .batch(batch_size)
  )

  train_pairs_2 = (
    train_data
    .shuffle(batch_size * 100)
    .batch(batch_size)
  )

  train_pairs = tf.data.Dataset.zip((train_pairs_1, train_pairs_2))
  train_data = train_pairs.map(
    lambda pair_1, pair_2: mix_up(pair_1, pair_2, alpha=mix_alpha),
    num_parallel_calls=tf.data.AUTOTUNE
  )
  
  test_data = test_data.map(
    lambda _, label: (_, tf.squeeze(tf.one_hot(label, num_class), 0)))

  return train_data, test_data

def mix_dataset_lwf(train_data, test_data,
                    batch_size, mix_alpha,
                    ref_model, num_class=10):
  train_data = train_data.map(
    lambda _, label: (_, tf.squeeze(tf.one_hot(label, num_class), 0)))

  train_pairs_1 = (
    train_data
    .shuffle(batch_size * 100)
    .batch(batch_size)
  )

  train_pairs_2 = (
    train_data
    .shuffle(batch_size * 100)
    .batch(batch_size)
  )

  train_pairs = tf.data.Dataset.zip((train_pairs_1, train_pairs_2))
  train_data = train_pairs.map(
    lambda pair_1, pair_2: mix_up(pair_1, pair_2, alpha=mix_alpha),
    num_parallel_calls=tf.data.AUTOTUNE
  ).map(
    lambda img, label:
#      (img, tf.squeeze(ref_model(tf.expand_dims(img, 0)), 0), label),
      (img, (tf.nn.softmax(ref_model(img), -1), label)),
    num_parallel_calls=tf.data.AUTOTUNE
  )
  
  test_data = test_data.map(
    lambda img, label: 
      (img, (tf.squeeze(tf.nn.softmax(ref_model(tf.expand_dims(img, 0)), -1), 0), 
          tf.squeeze(tf.one_hot(label, num_class), 0))))

  return train_data, test_data
