# 机器学习的核心难题：过拟合。
# 两点的重要性：准确的模型评估，以及训练与泛化之间的平衡。
# 如果在训练数据中进行评估，则在几轮之后他的性能就会发生偏离，称为过拟合

# 机器学习的根本问题在于优化与泛化的矛盾
# 优化：(Optimization)是指调节模型使其在训练数据上得到最佳性能的过程
# 泛化：(Generalization)是指在前所未见的数据上的性能。
# 但是无法控制泛化，只能通过调整训练数据模型来实现更优的泛化。

# 5.1.1 欠拟合与过拟合
# 损失值-训练时间函数中的训练曲线与验证曲线开始出现明显偏离的时间电脑称为稳健拟合
# 而关注训练曲线后，左侧是损失值下降欠拟合，右侧损失值上升是过拟合

# 1.嘈杂数据：噪音、噪声
# 某些性能不佳的训练单项：例如出现完全没有实际的判断意义（即对于人类无法判断），
# 抑或是标签与人的认识出现严重偏差的train项
# 如果这些都全部考量，则泛化的性能会下降

# 2.模糊特征：
# 即分类标准对于人来说都是模糊的，那么也会产生噪声——问题本身就包含不确定性与模糊性

# 3.罕见的特征与虚假相关性：
# 例如：猫的颜色与性格并没有太大的相关性，
# 如果训练数据中出现的数据类别‘中的包含数据量很少，则他很容易被归类成一个特定的类。

# MNIST 数据集添加白噪声通道或全零通道
from keras.datasets import mnist
import numpy as np
from tensorflow import keras
from keras import layers

(train_images, train_labels), _ = mnist.load_data()
train_images = train_images.reshape((60000, 28 * 28))
train_images = train_images.astype("float32") / 255
print(len(train_images))

train_images_with_noise_channels = np.concatenate(
    [train_images, np.random.random((len(train_images), 784))], axis=1)
train_images_with_zeros_channels = np.concatenate(
    [train_images, np.zeros((len(train_images), 784))], axis=1)


# 对于带有噪声通道或全零通道的 MNIST 数据，训练相同的模型
def get_model():
    model = keras.Sequential([
        layers.Dense(512, activation="relu"),
        layers.Dense(10, activation="softmax")
    ])
    model.compile(optimizer="rmsprop",
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    return model


model = get_model()
history_noise = model.fit(
    train_images_with_noise_channels, train_labels,
    epochs=10,
    batch_size=128,
    validation_split=0.2)
model = get_model()
history_zeros = model.fit(
    train_images_with_zeros_channels, train_labels,
    epochs=10,
    batch_size=128,
    validation_split=0.2)

# 看看两个模型的变化
import matplotlib.pyplot as plt

val_acc_noise = history_noise.history["val_accuracy"]
val_acc_zeros = history_zeros.history["val_accuracy"]
epochs = range(1, 11)
plt.plot(epochs, val_acc_noise, "b-",
         label="Validation accuracy with noise channels")
plt.plot(epochs, val_acc_zeros, "b--",
         label="Validation accuracy with zeros channels")
plt.title("Effect of noise channels on validation accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.show()

# 所以突出训练前的特征选择的重要性质(feature selection)
# 特征选择的常用方法是对每个特征计算有用性分数，并且只保留那些分数高于某个阈值的特征。
# 例如说通过heatmap来进行筛选
# 有用性分数（usefulness score）是用于衡量特征对于任务来说所包含信息量大小的指标，
# 比如特征与标签之间的互信息。这么做可以过滤前面例子中的白噪声通道。



