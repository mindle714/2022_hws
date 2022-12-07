import tensorflow as tf
from classification_models.tfkeras import Classifiers

def resnet18(num_class=10):
  ResNet18, preprocess_input = Classifiers.get('resnet18')
  up_model = ResNet18((224, 224, 3), weights='imagenet', include_top=False)
  x = tf.keras.layers.GlobalAveragePooling2D()(up_model.output)
  output = tf.keras.layers.Dense(num_class)(x)
  model = tf.keras.models.Model(inputs=[up_model.input], outputs=[output])
  return up_model, model, preprocess_input
