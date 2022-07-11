import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# os.system("cls")

import tensorflow as tf
from tensorflow import keras

n = 10000
x_train = tf.random.uniform(
    [n],
    minval=-10,
    maxval=10,
    dtype=tf.dtypes.float32,
    seed=None,
    name=None
)
noise = tf.random.normal(
    [n],
    mean=2.0,
    stddev=1.5,
    dtype=tf.dtypes.float32,
    seed=None,
    name=None
)
true_w = tf.constant(7.6)
true_b = tf.constant(-3.3)

y_train = x_train * true_w + true_b

model = keras.models.Sequential([
    keras.layers.Dense(1, activation='relu')
])

# loss and optimizer
loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
optim = keras.optimizers.Adam(learning_rate=0.001)
metrics = ["accuracy"]

model.compile(loss=loss, optimizer=optim, metrics=metrics)

# training
batch_size = 64
epochs = 5

model.fit(x_train, y_train, batch_size = batch_size, epochs = epochs, shuffle=True, verbose=2)