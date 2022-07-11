import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# os.system("cls")

import tensorflow as tf

n = 10000
x = tf.random.uniform(
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

###################################################
# Dense: y=wx+b
rows = n
net = tf.keras.layers.Dense(1) # 一个隐藏层，一个神经元
net.build((rows, 1)) # 每个训练数据有1个特征
print("net.w:", net.kernel) # 参数个数
print("net.b:", net.bias) # 和Dense数一样

a = tf.range([12])
print(a)
