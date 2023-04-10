import keras
from keras.datasets import imdb
import numpy as np
from keras import layers


def vectorize_sequences(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension))  # 创建一个形状为(len(sequences), dimension) 的零矩阵
    for i, sequence in enumerate(sequences):  # 将 results[i] 某些索引对应的值设为 1
        for j in sequence:
            results[i, j] = 1.
    return results


(train_data, train_labels), (test_data, test_labels) = imdb.load_data(
    num_words=10000)

x_train = vectorize_sequences(train_data)  # 训练数据向量化
x_test = vectorize_sequences(test_data)  # 测试数据向量化
print(x_train)
print(np.shape(x_test))
y_train = np.asarray(train_labels).astype("float32")
y_test = np.asarray(test_labels).astype("float32")  # 同样地，标签也向量化
print(np.shape(y_train))
print(np.shape(y_test))

model = keras.Sequential([
    layers.Dense(16, activation="relu"),
    layers.Dense(16, activation="relu"),
    layers.Dense(16, activation="relu"),
    layers.Dense(1, activation="sigmoid")
])
model.compile(optimizer="rmsprop",
              loss="binary_crossentropy",
              metrics=["accuracy"])
model.fit(x_train, y_train, epochs=4, batch_size=512)
results = model.evaluate(x_test, y_test)
print(results)
# my tested 2022/12/8 上午: [0.31183281540870667, 0.8740800023078918] “16 + 16 + 1”
# my tested 2022/12/8 下午1: [0.2894369661808014, 0.8846799731254578] “16 + 16 + 1”
# my tested 2022/12/8 下午1: [0.2894369661808014, 0.8846799731254578] “16 + 16 + 1”

# 易见，数据是有一定次数波动的，基于其原来的基础随机数据的影响，但是几乎不会超过 learning_rate/ 0.01 也许固定初始值会削弱变量的上下限？？？

# book tested: [0.2929924130630493, 0.88327999999999995] “16 + 16 + 1”
predict = model.predict(x_test)
print(predict)

# # 通过以下实验，可以确信前面选择的神经网络架构是非常合理的，不过仍有改进的空间。
# #  我们在最后的分类层之前使用了两个表示层。可以尝试使用一个或三个表示层，然后观
# # 察这么做对验证精度和测试精度的影响。
# #  尝试使用更多或更少的单元，比如 32 个或 64 个。
# #  尝试使用 mse 损失函数代替 binary_crossentropy。
# #  尝试使用 tanh 激活函数（这种激活函数在神经网络早期非常流行）代替 relu。
