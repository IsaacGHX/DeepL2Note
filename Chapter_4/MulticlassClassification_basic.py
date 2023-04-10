# 4.2 多分类问题示例：新闻分类
# 本节将构建一个模型，把路透社新闻划分到 46 个互斥的主题中。
# 这是一个多分类（multiclass classification）问题。
# 由于每个数据点只能划分到一个类别中，因此更具体地说，这是一个单标签、多分类（single-label, multiclass classification）问题。
# 如果每个数据点可以划分到多个类别（主题）中，那就是多标签、多分类（multilabel, multiclass classification）问题。
# model 的最后一层一定是分类的个数(+1)，最后一层使用了 softmax 激活函数。
# 对于这个例子，最好的损失函数是 categorical_crossentropy（分类交叉熵）


# 数据集加载
from tensorflow import keras
from keras.datasets import reuters
import numpy as np
from keras import layers

(train_data, train_labels), (test_data, test_labels) = reuters.load_data(
    num_words=10000)

# 我们有 8982 个训练样本和 2246 个测试样本。
# print( len(train_data))
# 8982
# print( len(test_data))
# 2246

# 与 IMDB 影评一样，每个样本都是一个整数列表（表示单词索引）。
# print(train_data[10])
# [1, 245, 273, 207, 156, 53, 74, 160, 26, 14, 46, 296, 26, 39, 74, 2979, 3554,
# 14, 46, 4689, 4329, 86, 61, 3499, 4795, 14, 61, 451, 4329, 17, 12]

# 将数据集解码为可读文字
word_index = reuters.get_word_index()
reverse_word_index = dict(
    [(value, key) for (key, value) in word_index.items()])
decoded_newswire = " ".join(
    [reverse_word_index.get(i - 3, "?") for i in train_data[0]])


# print(decoded_newswire)
# 注意，索引减去了 3，因为 0、1、2 分别是为“padding”（填充）、“start of sequence”（序列开始）、“unknown”（未知词）保留的索引

# 训练标签的意义：分类答案
# print(train_labels[10])  # 样本对应的标签是一个介于 0 和 45 之间的整数，即话题索引编号。

# 训练数据向量化
def vectorize_sequences(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension))  # 创建一个形状为(len(sequences), dimension) 的零矩阵
    for i, sequence in enumerate(sequences):  # 将 results[i] 某些索引对应的值设为 1
        for j in sequence:
            results[i, j] = 1.
    return results


x_train = vectorize_sequences(train_data)
x_test = vectorize_sequences(test_data)


# 标签向量化
def to_one_hot(labels, dimension=46):
    results = np.zeros((len(labels), dimension))
    for i, label in enumerate(labels):
        results[i, label] = 1.
    return results


y_train = to_one_hot(train_labels)
y_test = to_one_hot(test_labels)
# print(y_train)
# print(y_test)

# *******************************************
# Keras自带的编码方法
# from keras.utils import to_categorical
#
# y_train = to_categorical(train_labels)
# y_test = to_categorical(test_labels)
# *******************************************

# 模型定义
model = keras.Sequential([
    layers.Dense(64, activation="relu"),
    layers.Dense(64, activation="relu"),
    layers.Dense(46, activation="softmax")
])

model.compile(optimizer="rmsprop",
              loss="categorical_crossentropy",
              metrics=["accuracy"])

# 留出的验证数据
x_val = x_train[:1000]
partial_x_train = x_train[1000:]
y_val = y_train[:1000]
partial_y_train = y_train[1000:]

history = model.fit(partial_x_train,
                    partial_y_train,
                    epochs=20,
                    batch_size=512,  # 一般log2后会差4左右
                    validation_data=(x_val, y_val))
# 验证图像：
import matplotlib.pyplot as plt

loss = history.history["loss"]
val_loss = history.history["val_loss"]
epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, "bo", label="Training loss")
plt.plot(epochs, val_loss, "b", label="Validation loss")
plt.title("Training and validation loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()

plt.clf()
acc = history.history["accuracy"]
val_acc = history.history["val_accuracy"]
plt.plot(epochs, acc, "bo", label="Training accuracy")
plt.plot(epochs, val_acc, "b", label="Validation accuracy")
plt.title("Training and validation accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.show()
