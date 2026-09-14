import collections
import tensorflow as tf
import tensorflow_federated as tff

# Load simulation data.
source, _ = tff.simulation.datasets.emnist.load_data()
def client_data(n):
  return source.create_tf_dataset_for_client(source.client_ids[n]).map(
      lambda e: (tf.reshape(e['pixels'], [-1]), e['label'])
  ).repeat(10).batch(20)

# Pick a subset of client devices to participate in training.
train_data = [client_data(n) for n in range(3)]

# Wrap a Keras model for use with TFF.
keras_model = tf.keras.models.Sequential([
  tf.keras.layers.Dense(
    10, tf.nn.softmax, input_shape=(784,), kernel_initializer='zeros')
])
tff_model = tff.learning.models.functional_model_from_keras(
      keras_model,
      loss_fn=tf.keras.losses.SparseCategoricalCrossentropy(),
      input_spec=train_data[0].element_spec,
      metrics_constructor=collections.OrderedDict(
        accuracy=tf.keras.metrics.SparseCategoricalAccuracy))

# Simulate a few rounds of training with the selected client devices.
trainer = tff.learning.algorithms.build_weighted_fed_avg(
  tff_model,
  client_optimizer_fn=tff.learning.optimizers.build_sgdm(learning_rate=0.1))
state = trainer.initialize()
for _ in range(5):
  result = trainer.next(state, train_data)
  state = result.state
  metrics = result.metrics
  print(metrics['client_work']['train']['accuracy'])