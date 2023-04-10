# 神经网络的基本数据结构是层
# 简单的向量数据存储在形状为 (samples, features) 的 2 阶张量中通常用
# 密集连接层［densely connected layer，也叫全连接层（fully connected layer）或密集层（dense layer），对应于 Keras 的 Dense 类
# 序列数据存储在形状为 (samples, timesteps, features) 的 3 阶张量中，通常用循环层（recurrent layer）来处理，比如 LSTM 层或一维卷积层（Conv1D）。
# 图像数据存储在 4 阶张量中，通常用二维卷积层（Conv2D）来处理

# 本页中的所有inputs与targets继承TensorFlow_Basic

from Chapter_3.TensorFlow_basic import *
from tensorflow import keras
import tensorflow as tf
import numpy as np


class SimpleDense(keras.layers.Layer):
    def __init__(self, units, activation=None):
        super().__init__()
        self.units = units
        self.activation = activation

    def build(self, input_shape):  # 重建权重
        input_dim = input_shape[-1]
        self.W = self.add_weight(shape=(input_dim, self.units),  # 形状
                                 initializer="random_normal")  # 随机初始化
        self.b = self.add_weight(shape=(self.units,),
                                 initializer="zeros")
        # add_weight()是创建权重的快捷方法。你也可以创建独立变量，并指定其作为层属性，比如：
        # self.W = tf.Variable(tf.
        #                      random.uniform(w_shape))

    def __call__(self, inputs):  # 注意：这里会调用build
        if not self.built:
            self.build(inputs.shape)
        self.built = True
        return self.call(inputs)

    def call(self, inputs):
        y = tf.matmul(inputs, self.W) + self.b  # 前向传播
        if self.activation is not None:
            y = self.activation(y)
        return y


# 上述相当于简述了Keras的Dense

# 层兼容性（layer compatibility）
# 的概念具体指的是，每一层只接收特定形状的输入张量，并返回特定形状的输出张量。

from keras import layers  # 自动推断层的形状

layer = layers.Dense(32, activation="relu")  # 第一维形状32*32，故后一层只能接受32为行度的层

from keras import models

model = models.Sequential([
    layers.Dense(32, activation="relu"),
    layers.Dense(32)
])

# 遇到的第一个形状就是输入形状


from Chapter_2.naive_tensorflow import NaiveSequential
from Chapter_2.naive_tensorflow import NaiveDense

model = NaiveSequential([
    NaiveDense(input_size=784, output_size=32, activation="relu"),
    NaiveDense(input_size=32, output_size=64, activation="relu"),
    NaiveDense(input_size=64, output_size=32, activation="relu"),
    NaiveDense(input_size=32, output_size=10, activation="softmax")
])

# 自动形状判断，对于上述代码的简化
# 传入每个 Dense 层的第一个参数是该层的单元（unit）个数，即该层表示空间的维数。
model = keras.Sequential([
    SimpleDense(32, activation="relu"),
    SimpleDense(64, activation="relu"),
    SimpleDense(32, activation="relu"),
    SimpleDense(10, activation="softmax")
])

print(model)

# 3.6.2 从layer到model
# model是由layers构成的
# 常见结构：双分支( two - branch)、多头( multihead)、残差连接

# 简述transformer架构：MultiheadAttention，通过可能存在的上一层中的LayerNormalization的附加——1
# 给这一层的LayerNormalization来进行权重调整给这一层的Dense赋值而后再调整之后的LayerNormalization
# 重复步骤1

# 3.6.3
# LossFunction 损失函数( 目标函数)——需要被最小化，即减小误差
# Optimizer 优化器——决定通过什么方法来对损失函数来进行更新，一般实行的是SGD(随机梯度下降法的某个变体)
# Label 指标——衡量是否成功的指标，训练与验证中需要进行的监控，如分类精度；指标不需要可微，因为其不需要被优化

# 确定了以上的三个重点：就可以用compile() 以及fit() 进行适用
# 实例：
model = keras.Sequential([keras.layers.Dense(1)])
model.compile(optimizer="rmsprop",
              loss="mean_squared_error",
              metrics=["accuracy"])
# compile()函数的接收参数：optimizer、loss、metrics
# 可以通过“”的方式来进行传递，也可以通过keras.optimizer."优化器名字"()
# 大考点！！！
# 优化器常用有：SGD、RMSprop、Adam、Adgard、……

# 随时函数常有：CategoricalCrossentropy(分类交叉熵)、SparseCategoricalCrossentropy(稀疏分类交叉熵)、
# BinaryCrossentropy(二进制交叉熵)、MeanSquaredError(均方差)、CosineSimilarity(余弦相似性)

# fit()的实例：
history = model.fit(
    inputs,
    targets,
    epochs=5,
    batch_size=128
)
# 需要传入的参数是：样本、目标、训练循环次数epoch、批量大小batch_size
model = keras.Sequential([keras.layers.Dense(1)])  # 1层的DENSE
model.compile(optimizer=keras.optimizers.RMSprop(learning_rate=0.1),  # 可以在optimizer中添加learning_rate
              loss=keras.losses.MeanSquaredError(),
              metrics=[keras.metrics.BinaryAccuracy()])

# 避免验证数据的类别一致性，需要通过标签来进行打乱
indices_permutation = np.random.permutation(len(inputs))
shuffled_inputs = inputs[indices_permutation]
shuffled_targets = targets[indices_permutation]

# 保留30%的训练数据用于验证，相当于前70%是训练的数据
num_validation_samples = int(0.3 * len(inputs))
val_inputs = shuffled_inputs[:num_validation_samples]
val_targets = shuffled_targets[:num_validation_samples]
training_inputs = shuffled_inputs[num_validation_samples:]
training_targets = shuffled_targets[num_validation_samples:]

model.fit(
    training_inputs,
    training_targets,
    epochs=5,
    batch_size=16,
    validation_data=(val_inputs, val_targets)
)

loss_and_metrics = model.evaluate(val_inputs, val_targets, batch_size=128)
predictions = model.predict(val_inputs, batch_size=128)
print(predictions[:10])
# 2022.12.5 binary_accuracy: 0.9983，末predict
# 书上的原始实验
# [[ 1.146817  ]
#  [-0.0751701 ]
#  [ 0.7428402 ]
#  [-0.16093576]
#  [ 1.0438516 ]
#  [-0.2601635 ]
#  [-0.07389385]
#  [-0.34415895]
#  [ 1.0352826 ]
#  [ 1.119683  ]]
