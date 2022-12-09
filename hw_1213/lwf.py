import tensorflow as tf
import keras

class LWF(keras.Model):
  def __init__(self, new_model, old_model, ref_old_model, old_model_hook=None):
    super(LWF, self).__init__()
    self.new_model = new_model
    self.old_model = old_model
    self.ref_old_model = ref_old_model
    self.old_model_hook = old_model_hook

  def compile(
    self,
    optimizer1,
    optimizer2,
    metrics,
    new_model_loss_fn,
    distillation_loss_fn,
    temperature=2,
  ):
    """ Configure the LWF.

    Args:
      optimizer: Keras optimizer for the new_model weights
      metrics: Keras metrics for evaluation
      new_model_loss_fn: Loss function of difference between new_model
        predictions and ground-truth
      distillation_loss_fn: Loss function of difference between soft
        new_model predictions and soft old_model predictions
      temperature: Temperature for softening probability distributions.
        Larger temperature gives softer distributions.
    """
    super(LWF, self).compile(optimizer=optimizer1, metrics=metrics)
    self.optimizer1 = optimizer1
    self.optimizer2 = optimizer2
    self.new_model_loss_fn = new_model_loss_fn
    self.distillation_loss_fn = distillation_loss_fn
    self.temperature = temperature

  def train_step(self, data):
    x, y = data

    if self.old_model_hook is None:
      ref_old_model_predictions = self.ref_old_model(x, training=False)
    else:
      ref_old_model_predictions = self.ref_old_model(self.old_model_hook(x), training=False)

    with tf.GradientTape() as tape:
      new_model_predictions = self.new_model(x, training=True)
      new_model_loss = self.new_model_loss_fn(y, new_model_predictions)
      '''
    trainable_vars = self.new_model.trainable_variables
    gradients = tape.gradient(new_model_loss, trainable_vars)
    self.optimizer1.apply_gradients(zip(gradients, trainable_vars))
    
    with tf.GradientTape() as tape:
      '''
      old_model_predictions = self.old_model(x, training=True)
      # Compute scaled distillation loss from https://arxiv.org/abs/1503.02531
      # The magnitudes of the gradients produced by the soft targets scale
      # as 1/T^2, multiply them by T^2 when using both hard and soft targets.
      distillation_loss = (
        self.distillation_loss_fn(
          tf.nn.softmax(ref_old_model_predictions / self.temperature, axis=1),
          tf.nn.softmax(old_model_predictions / self.temperature, axis=1),
        )
        * self.temperature**2
      )
      loss = new_model_loss + distillation_loss
    
    trainable_vars = self.old_model.trainable_variables + self.new_model.trainable_variables[-2:]
    gradients = tape.gradient(loss, trainable_vars)
    self.optimizer1.apply_gradients(zip(gradients, trainable_vars))

    # Update the metrics configured in `compile()`.
    self.compiled_metrics.update_state(y, new_model_predictions)

    # Return a dict of performance
    results = {m.name: m.result() for m in self.metrics}
    results.update(
      {"new_model_loss": new_model_loss, "distillation_loss": distillation_loss}
    )
    return results

  def test_step(self, data):
    # Unpack the data
    x, y = data

    # Compute predictions
    y_prediction = self.new_model(x, training=False)

    # Calculate the loss
    new_model_loss = self.new_model_loss_fn(y, y_prediction)

    # Update the metrics.
    self.compiled_metrics.update_state(y, y_prediction)

    # Return a dict of performance
    results = {m.name: m.result() for m in self.metrics}
    results.update({"new_model_loss": new_model_loss})
    return results
