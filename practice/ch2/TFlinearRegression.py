import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# os.system("cls")

import tensorflow as tf
from tensorflow import keras

def normalize_fixed(x):
    current_max = tf.math.reduce_max(
    x, axis=None, keepdims=False, name=None
    )
    current_min = tf.math.reduce_min(
    x, axis=None, keepdims=False, name=None
    )
    # current_min, current_max = tf.expand_dims(current_range[:, 0], 1), tf.expand_dims(current_range[:, 1], 1)
    # normed_min, normed_max = tf.expand_dims(normed_range[:, 0], 1), tf.expand_dims(normed_range[:, 1], 1)
    x_normed = (x - current_min) / (current_max - current_min)
    # x_normed = x_normed * (normed_max - normed_min) + normed_min
    return x_normed

n = 1000
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

x_train = normalize_fixed(x_train)
y_train = normalize_fixed(y_train)

model = keras.models.Sequential([
    keras.layers.Input(shape=(1,)),
    keras.layers.Dense(1, activation='relu')
])

# loss and optimizer
loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
optim = keras.optimizers.SGD(learning_rate=0.01, clipnorm=1.)
metrics = ["accuracy"]

model.compile(loss=loss, optimizer=optim, metrics=metrics)

# training
batch_size = 64
epochs = 5

model.fit(x_train, y_train, batch_size = batch_size, epochs = epochs, shuffle=True, verbose=2)

c = 0
for x in model.weights:
    print(str(c) + str(x))
    c = c + 1