# 二分类(binary classification) —— 影评分类——大数据分类

# 二分类的最后label往往是两个不相关的答案如，True/False, Cat/Dog, 这类

# 模型输出是一个概率值（模型最后一层只有一个单元并使用 sigmoid 激活函数），所以最好使用 binary_crossentropy（二元交叉熵）损失函数
# 你还可以使用 mean_squared_error（均方误差）。但对于输出概率值的模型，交叉熵（crossentropy）通常是最佳选择。
# 本节将使用 IMDB 数据集，它包含来自互联网电影数据库（IMDB）的 50 000 条
# //严重两极化//的评论。
# 数据集被分为 25 000 条用于训练的评论与 25 000 条用于测试的评论，
# 训练集和测试集都包含 50% 的正面评论与 50% 的负面评论。

# 初始化：ASCII转整数
from keras.datasets import imdb

(train_data, train_labels), (test_data, test_labels) = imdb.load_data(
    num_words=10000)  # train数据初始化，提取最多出现的10000个词汇，剔除无频度词汇
# imdb.load_data会全部提取后重建，并且会显示时间、进度


print(train_data[0])
print(train_labels)
# 列表中 0 代表负面（negative），1 代表正面（positive）
print(max([max(sequence) for sequence in train_data]))
# 原数据组别按照出现从高到低，由0开始排布


# 评论解码为文本
word_index = imdb.get_word_index()  # 从word_index里面找被编码为数字的词语
reverse_word_index = dict(
    [(value, key) for (key, value) in word_index.items()])  # 字典字符的data与数组的index位置互换
decoded_review = " ".join(
    [reverse_word_index.get(i - 3, "?") for i in train_data[0]])
# 注意-3
print(decoded_review)

from HEAD import *

def vectorize_sequences(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension))  # 创建一个形状为(len(sequences), dimension) 的零矩阵
    for i, sequence in enumerate(sequences):  # 将 results[i] 某些索引对应的值设为 1
        for j in sequence:
            results[i, j] = 1.
    return results


x_train = vectorize_sequences(train_data)  # 训练数据向量化
x_test = vectorize_sequences(test_data)  # 测试数据向量化

y_train = np.asarray(train_labels).astype("float32")
y_test = np.asarray(test_labels).astype("float32")  # 同样地，标签也向量化

# 4.1.3构建模型
# 带有激活函数Relu的密集链接层的简单堆叠
# 决定：多少层，每层多少单元

from tensorflow import keras
from keras import layers

model = keras.Sequential([
    layers.Dense(16, activation="relu"),
    layers.Dense(16, activation="relu"),  # 两个中间层，每层 16 个单元
    layers.Dense(1, activation="sigmoid")  # 最后一个统计，来输出一个情感预测
])
# 每一层都是output = relu(dot(input, W) + b)

# compile函数
model.compile(optimizer="rmsprop",
              loss="binary_crossentropy",
              metrics=["accuracy"])

x_val = x_train[:10000]
partial_x_train = x_train[10000:]
y_val = y_train[:10000]
partial_y_train = y_train[10000:]

history = model.fit(partial_x_train,
                    partial_y_train,
                    epochs=20,
                    batch_size=512,
                    validation_data=(x_val, y_val),
                    callbacks = [tensorboard_callback])

history_dict = history.history
print(history_dict.keys())
# dict_keys(['loss', 'accuracy', 'val_loss', 'val_accuracy'])

# 数据绘制：
import matplotlib.pyplot as plt

history_dict = history.history
loss_values = history_dict["loss"]
val_loss_values = history_dict["val_loss"]
epochs = range(1, len(loss_values) + 1)
plt.plot(epochs, loss_values, "bo", label="Training loss")
plt.plot(epochs, val_loss_values, "b", label="Validation loss")
plt.title("Training and validation loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()

# 绘制训练精度与验证精度
plt.clf()  # 清空原来的图像
acc = history_dict["accuracy"]
val_acc = history_dict["val_accuracy"]
plt.plot(epochs, acc, "bo", label="Training acc")
plt.plot(epochs, val_acc, "b", label="Validation acc")
plt.title("Training and validation accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.show()
