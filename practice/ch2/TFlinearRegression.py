import tensorflow as tf

# 定義一個隨機數（純量）
random_float = tf.random.uniform(shape=(10,1)) *10
X = random_float
w = tf.constant([6.7])
b = tf.constant([-3.3])
Y =  w * X + b
# Y = tf.matmul(X, w) # 計算矩陣A和B的乘積
# Y = tf.add(Y, b)    # 計算矩陣A和B的和

# 定義一個有2個元素的零向量
zero_vector = tf.zeros(shape=(2))

# 定義兩個2×2的常量矩陣
A = tf.constant([[1., 2.], [3., 4.]])
B = tf.constant([[5., 6.], [7., 8.]])

optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)
wg = tf.Variable(initial_value=0.)
bg = tf.Variable(initial_value=0.)
variables = [wg, bg]
print(X)

num_epoch = 1000
for e in range(num_epoch):
    # 使用tf.GradientTape()記錄損失函數的梯度資訊
    with tf.GradientTape() as tape:
        y_pred = wg * X + bg
        loss = tf.reduce_sum(tf.square(y_pred - Y)) / 10
    # TensorFlow自動計算損失函數關於自變數（模型參數）的梯度
    grads = tape.gradient(loss, variables)
    # TensorFlow自動根據梯度更新參數
    optimizer.apply_gradients(grads_and_vars=zip(grads, variables))

print(grads)
print(wg, bg)

class Linear(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.dense = tf.keras.layers.Dense(
            units=1,
            activation=None,
            kernel_initializer=tf.zeros_initializer(),
            bias_initializer=tf.zeros_initializer()
        )

    def call(self, input):
        output = self.dense(input)
        return output


# 以下程式碼結構與前一節類似
model = Linear()
optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)
for i in range(1000):
    with tf.GradientTape() as tape:
        y_pred = model(X)      # 呼叫模型 y_pred = model(X) 而不是顯式寫出 y_pred = a * X + b
        loss = tf.reduce_mean(tf.square(y_pred - Y))
        # loss = tf.reduce_sum(tf.square(y_pred - Y)) / 10
    grads = tape.gradient(loss, model.variables)    # 使用 model.variables 這一屬性直接獲得模型中的所有變數
    optimizer.apply_gradients(grads_and_vars=zip(grads, model.variables))
print(model.variables)