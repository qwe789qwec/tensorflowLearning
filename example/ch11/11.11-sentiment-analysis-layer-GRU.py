#!/usr/bin/env python
# encoding: utf-8
"""
@author: HuRuiFeng
@file: 11.11-sentiment-analysis-layer-GRU.py
@time: 2020/2/28 16:25
@desc: 11.11 GRU情感分类问题实战的代码（layer方式）
"""

import os

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, losses, optimizers, Sequential

tf.random.set_seed(22)
np.random.seed(22)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def load_dataset(batchsz, total_words, max_review_len):
    # 加载IMDB数据集，此处的数据采用数字编码，一个数字代表一个单词
    (x_train, y_train), (x_test, y_test) = keras.datasets.imdb.load_data(num_words=total_words)
    print(x_train.shape, len(x_train[0]), y_train.shape)
    print(x_test.shape, len(x_test[0]), y_test.shape)

    # x_train:[b, 80]
    # x_test: [b, 80]
    # 截断和填充句子，使得等长，此处长句子保留句子后面的部分，短句子在前面填充
    x_train = keras.preprocessing.sequence.pad_sequences(x_train, maxlen=max_review_len)
    x_test = keras.preprocessing.sequence.pad_sequences(x_test, maxlen=max_review_len)
    # 构建数据集，打散，批量，并丢掉最后一个不够batchsz的batch
    db_train = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    db_train = db_train.shuffle(1000).batch(batchsz, drop_remainder=True)
    db_test = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    db_test = db_test.batch(batchsz, drop_remainder=True)
    print('x_train shape:', x_train.shape, tf.reduce_max(y_train), tf.reduce_min(y_train))
    print('x_test shape:', x_test.shape)
    return db_train, db_test


class MyRNN(keras.Model):
    # Cell方式构建多层网络
    def __init__(self, units, total_words, embedding_len, max_review_len):
        super(MyRNN, self).__init__()
        # 词向量编码 [b, 80] => [b, 80, 100]
        self.embedding = layers.Embedding(total_words, embedding_len,
                                          input_length=max_review_len)
        # 构建RNN
        self.rnn = Sequential([
            layers.GRU(units, dropout=0.5, return_sequences=True),
            layers.GRU(units, dropout=0.5)
        ])

        # 构建分类网络，用于将CELL的输出特征进行分类，2分类
        # [b, 80, 100] => [b, 64] => [b, 1]
        self.outlayer = Sequential([
            layers.Dense(32),
            layers.Dropout(rate=0.5),
            layers.ReLU(),
            layers.Dense(1)])

    def call(self, inputs, training=None):
        x = inputs  # [b, 80]
        # embedding: [b, 80] => [b, 80, 100]
        x = self.embedding(x)
        # rnn cell compute,[b, 80, 100] => [b, 64]
        x = self.rnn(x)
        # 末层最后一个输出作为分类网络的输入: [b, 64] => [b, 1]
        x = self.outlayer(x, training)
        # p(y is pos|x)
        prob = tf.sigmoid(x)

        return prob


def main():
    batchsz = 128  # 批量大小
    total_words = 10000  # 词汇表大小N_vocab
    embedding_len = 100  # 词向量特征长度f
    max_review_len = 80  # 句子最大长度s，大于的句子部分将截断，小于的将填充

    db_train, db_test = load_dataset(batchsz, total_words, max_review_len)

    units = 32  # RNN状态向量长度f
    epochs = 20  # 训练epochs

    model = MyRNN(units, total_words, embedding_len, max_review_len)
    # 装配
    model.compile(optimizer=optimizers.RMSprop(0.001),
                  loss=losses.BinaryCrossentropy(),
                  metrics=['accuracy'],
                  experimental_run_tf_function=False)
    # 训练和验证
    model.fit(db_train, epochs=epochs, validation_data=db_test)
    # 测试
    model.evaluate(db_test)


if __name__ == '__main__':
    main()
