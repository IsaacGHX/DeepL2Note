# 数据集加载
from tensorflow import keras
from keras.datasets import reuters
import numpy as np
from keras import layers

(train_data, train_labels), (test_data, test_labels) = reuters.load_data(
    num_words=10000)


def vectorize_sequences(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension))  # 创建一个形状为(len(sequences), dimension) 的零矩阵
    for i, sequence in enumerate(sequences):  # 将 results[i] 某些索引对应的值设为 1
        for j in sequence:
            results[i, j] = 1.
    return results


x_train = vectorize_sequences(train_data)
x_test = vectorize_sequences(test_data)


def to_one_hot(labels, dimension=46):
    results = np.zeros((len(labels), dimension))
    for i, label in enumerate(labels):
        results[i, label] = 1.
    return results


# 模型一：
y_train = to_one_hot(train_labels)
y_test = to_one_hot(test_labels)

x_val = x_train[:1000]
partial_x_train = x_train[1000:]
y_val = y_train[:1000]
partial_y_train = y_train[1000:]

model = keras.Sequential([
    layers.Dense(64, activation="relu"),
    layers.Dense(64, activation="relu"),
    layers.Dense(46, activation="softmax")
])

model.compile(optimizer="rmsprop",
              loss="categorical_crossentropy",
              metrics=["accuracy"])
model.fit(x_train,
          y_train,
          epochs=9,
          batch_size=512)
results = model.evaluate(x_test, y_test)
print(results)
# 书上原本[0.9565213431445807, 0.79697239536954589]
# 2022/12/12 尝试1 [0.9117308855056763, 0.7947462201118469]
import copy

test_labels_copy = copy.copy(test_labels)
np.random.shuffle(test_labels_copy)
hits_array = np.array(test_labels) == np.array(test_labels_copy)
hits_array.mean()
print(hits_array)
# 0.18655387355298308

# 新数据预测：
predictions = model.predict(x_test)
print(predictions[0].shape)
print(np.sum(predictions[0]))
print(np.argmax(predictions[0]))

# 模型二：
# 处理标签与损失的另外方法
# 将其转换为整数张量
y_train = np.array(train_labels)
y_test = np.array(test_labels)

x_val = x_train[:1000]
partial_x_train = x_train[1000:]
y_val = y_train[:1000]
partial_y_train = y_train[1000:]

model = keras.Sequential([
    layers.Dense(64, activation="relu"),
    layers.Dense(64, activation="relu"),
    layers.Dense(46, activation="softmax")
])

# 对于整数标签，你应该使用 sparse_categorical_crossentropy（稀疏分类交叉熵）损失函数。
model.compile(optimizer="rmsprop",
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])

model.fit(x_train,
          y_train,
          epochs=9,
          batch_size=512)
results = model.evaluate(x_test, y_test)
# 运算结果：[0.9717336297035217, 0.792965292930603] tensorflow-gpu的3060laptop


print(results)

# **************************************************
# 拥有足够大的中间层的重要性
# 具有信息瓶颈的模型，会造成验证精度的下降
# model = keras.Sequential([
#     layers.Dense(64, activation="relu"),
#     layers.Dense(4, activation="relu"),
#     layers.Dense(46, activation="softmax")
# ])
# model.compile(optimizer="rmsprop",
#               loss="categorical_crossentropy",
#               metrics=["accuracy"])
# model.fit(partial_x_train,
#           partial_y_train,
#           epochs=20,
#           batch_size=128,
#           validation_data=(x_val, y_val))
# 模型能够将大部分必要信息塞进这个 4 维表示中，但并不是全部信息。

# 进一步实验：
#  尝试使用更小或更大的层，比如 32 个单元、128 个单元等。
#  你在最终的 softmax 分类层之前使用了两个中间层。现在尝试使用一个或三个中间层。

# 小结：
#  如果要对 N 个类别的数据点进行分类，那么模型的最后一层应该是大小为 N 的 Dense 层。
#  对于单标签、多分类问题，模型的最后一层应该使用 softmax 激活函数，这样可以输出一个在 N 个输出类别上的概率分布。
#  对于这种问题，损失函数几乎总是应该使用分类交叉熵。它将模型输出的概率分布与目标的真实分布之间的距离最小化。
#  处理多分类问题的标签有两种方法：
#        通过分类编码（也叫 one-hot 编码）对标签进行编码，然后使用 categorical_crossentropy 损失函数；
#        将标签编码为整数，然后使用 sparse_categorical_crossentropy 损失函数。
#  如果你需要将数据划分到多个类别中，**那么应避免使用太小的中间层**，以免在模型中造成信息瓶颈。
